# serve_packing_api.py
import os, sys, time, uuid
from typing import List, Optional, Dict, Any
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from types import SimpleNamespace as NS
import uuid

from tianshou.data import Batch
from tianshou.data import Batch, to_torch

# project imports (light at top, heavy under startup)
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

from tools import registration_envs, set_seed, CategoricalMasked
import arguments

# ---------- Models for requests/responses ----------
class StartSessionReq(BaseModel):
    container_size: List[int] = [10, 10, 10]
    scheme: str = "EMS"
    rot: bool = True
    k_placement: int = 80
    box_size_set: Optional[List[List[int]]] = None
    seed: Optional[int] = None
    render: bool = False  # server-side VTK off by default
    ckp: Optional[str] = None    

class PlaceReq(BaseModel):
    session_id: str
    box: List[int]  # [w, l, h] to place NOW

class PlaceResp(BaseModel):
    session_id: str
    placed_dim: List[int]          # actual (maybe rotated) dims used for placement
    pos: List[int]                 # (x, y, z)
    rot: int                       # 0 or 1 (your env uses 0/1 to swap w<->l)
    filled_ratio: float
    count: int
    done: bool
    boxes: List[Dict[str, Any]]    # all placed boxes so far: {"dim":[w,l,h], "pos":[x,y,z]}

class StateResp(BaseModel):
    session_id: str
    filled_ratio: float
    count: int
    boxes: List[Dict[str, Any]]

# ---------- Selectable creator (so we can inject the current box) ----------
class SelectableBoxCreator:
    def __init__(self, container_size, catalog=None, can_rotate=False, seed=None):
        self.container_size = np.array(container_size, dtype=np.int32)
        self.catalog = np.array(catalog if catalog else [(2,2,2)], dtype=np.int32)
        self.can_rotate = can_rotate
        self.rng = np.random.default_rng(seed)
        self.box_list: List[List[int]] = []
        self._forced_next: Optional[List[int]] = None
        self.autofill = False

    def preview(self, n):
        # Only auto-fill when autofill is enabled (training mode).
        if self.autofill:
            while len(self.box_list) < n:
                self.generate_box_size()
        # If caller disabled autofill (API mode), return whatever is in the list.
        # If empty, return [] and let env handle that case.
        return [b[:] for b in self.box_list[:n]] if self.box_list else []

    def drop_box(self): self.box_list.pop(0)
    def reset(self):
        self.box_list.clear()
        self._forced_next = None
        # leave self.autofill as-is; server will set it to False for API mode

    def force_next(self, size):
        size = np.minimum(np.array(size, dtype=np.int32), self.container_size).tolist()
        self._forced_next = size

    def override_current(self, size):
        size = np.minimum(np.array(size, dtype=np.int32), self.container_size).tolist()
        self.box_list = [size]    # replace current queue with exactly this box

    def _sample_one(self):
        size = self.catalog[self.rng.integers(len(self.catalog))].copy()
        if self.can_rotate:
            perms = ((0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0))
            size = size[list(perms[self.rng.integers(6)])]
        return np.minimum(size, self.container_size).tolist()

    def generate_box_size(self, **kwargs):
        if self._forced_next is not None:
            self.box_list.append(self._forced_next)
            self._forced_next = None
        else:
            self.box_list.append(self._sample_one())

# ---------- FastAPI app & global state ----------
app = FastAPI(title="Packing Placement API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # or ["http://127.0.0.1:5500", "http://localhost:5173", ...]
    allow_credentials=True,
    allow_methods=["*"],           # includes OPTIONS, GET, POST, etc.
    allow_headers=["*"],           # e.g., Content-Type, Authorization
)
SESS: Dict[str, Dict[str, Any]] = {}   # session_id -> {"env", "sel", "args", "policy", "device"}

# ---------- Lazy heavy imports at startup ----------
def _startup_load():
    import gymnasium as gym
    from ts_train import build_net
    from masked_ppo import MaskedPPOPolicy
    from tianshou.utils.net.common import ActorCritic
    return gym, build_net, MaskedPPOPolicy, ActorCritic

def _boxes_list(env):
    """
    Returns a list of {"dim":[w,l,h], "pos":[x,y,z]} for all placed boxes.
    Works whether env.container.boxes stores tuples or Box objects.
    """
    out = []
    if not hasattr(env, "container") or not hasattr(env.container, "boxes"):
        return out

    def to_int_list(x):
        # numpy array / tensor / list / tuple -> list[int]
        try:
            if hasattr(x, "tolist"):
                x = x.tolist()
        except Exception:
            pass
        return [int(v) for v in x]

    # helpers to extract attributes by common names
    size_keys = ("size", "dim", "dims", "dimension", "shape")
    pos_keys  = ("pos", "position", "xyz", "origin", "corner", "start")

    for b in env.container.boxes:
        if isinstance(b, (list, tuple)) and len(b) >= 6:
            dim = to_int_list(b[:3])
            pos = to_int_list(b[3:6])
        else:
            # object path
            dim_val = None
            for k in size_keys:
                if hasattr(b, k):
                    dim_val = getattr(b, k)
                    break
            pos_val = None
            for k in pos_keys:
                if hasattr(b, k):
                    pos_val = getattr(b, k)
                    break

            # some implementations store attributes as separate scalars
            if dim_val is None:
                # try individual fields
                w = getattr(b, "w", None)
                l = getattr(b, "l", getattr(b, "length", None))
                h = getattr(b, "h", getattr(b, "height", None))
                if None not in (w, l, h):
                    dim_val = [w, l, h]

            if pos_val is None:
                x = getattr(b, "x", None)
                y = getattr(b, "y", None)
                z = getattr(b, "z", None)
                if None not in (x, y, z):
                    pos_val = [x, y, z]

            if dim_val is None or pos_val is None:
                # last resort: try to format nicely for debugging, but skip entry
                # print(f"Warning: cannot extract dim/pos from Box: {type(b)} -> {dir(b)}")
                continue

            dim = to_int_list(dim_val)
            pos = to_int_list(pos_val)

        out.append({"dim": dim, "pos": pos})

    return out


@app.on_event("startup")
def on_startup():
    registration_envs()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

@app.post("/session/start", response_model=StateResp)
def start_session(cfg: StartSessionReq):
    gym, build_net, MaskedPPOPolicy, ActorCritic = _startup_load()

    seed = cfg.seed or int(time.time())
    set_seed(seed, cuda=True, cuda_deterministic=False)

    # arguments
    args = NS(
        cuda=True,
        cuda_deterministic=False,
        env=NS(
            id="OnlinePack-v1",
            container_size=tuple(cfg.container_size),
            rot=cfg.rot,
            scheme=cfg.scheme,
            k_placement=int(cfg.k_placement),
            box_type="random",
            box_size_set=cfg.box_size_set or [(2,2,2),(2,3,4),(3,3,2),(4,2,2),(5,3,2),(3,5,4),(6,2,3)],
            box_big=max(max(w,l,h) for (w,l,h) in (cfg.box_size_set or [(2,2,2),(2,3,4),(3,3,2),(4,2,2),(5,3,2),(3,5,4),(6,2,3)])),
        ),
        train=NS(
            gamma=1.0,
            clip_param=0.3,
            gae_lambda=0.96,
            reward_type=None,
        ),
        opt=NS(lr=7e-5, eps=1e-5),
        loss=NS(value=0.5, entropy=0.001),
        model=NS(embed_dim=128, num_layers=3, forward_expansion=2, heads=1, dropout=0.0, padding_mask=False),
    )
    args.env.id = "OnlinePack-v1"
    args.env.container_size = tuple(cfg.container_size)
    args.env.rot = cfg.rot
    args.env.scheme = cfg.scheme
    args.env.k_placement = int(cfg.k_placement)
    args.env.box_type = "random"
    args.env.box_size_set = cfg.box_size_set or [(2,2,2),(2,3,4),(3,3,2),(4,2,2),(5,3,2),(3,5,4),(6,2,3)]
    args.env.box_big = max(max(w,l,h) for (w,l,h) in args.env.box_size_set)

    # device + policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, critic = build_net(args, device)
    actor_critic = ActorCritic(actor, critic).to(device)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.opt.lr, eps=args.opt.eps)
    policy = MaskedPPOPolicy(
        actor=actor, critic=critic, optim=optim,
        dist_fn=CategoricalMasked,
        discount_factor=args.train.gamma, eps_clip=args.train.clip_param,
        advantage_normalization=False, vf_coef=args.loss.value,
        ent_coef=args.loss.entropy, gae_lambda=args.train.gae_lambda,
        action_space=None  # set after env created
    ).to(device)
    policy.eval()

    # load checkpoint from args.ckp env var or config path if you like
    ckp = cfg.ckp or os.environ.get("GOPT_CHECKPOINT")
    if not ckp or not os.path.exists(ckp):
        raise HTTPException(400, detail="Checkpoint not provided or path not found. "
                                        "Pass 'ckp' in /session/start body or set env GOPT_CHECKPOINT.")
    policy.load_state_dict(torch.load(ckp, map_location=device))

    # env
    env = gym.make(
        args.env.id,
        container_size=args.env.container_size,
        enable_rotation=args.env.rot,
        data_type="random",
        item_set=args.env.box_size_set,
        reward_type=args.train.reward_type,
        action_scheme=args.env.scheme,
        k_placement=args.env.k_placement,
        is_render=cfg.render,
    )

    # tie action space now that env exists
    policy.action_space = env.action_space

    # swap in selectable creator
    sel = SelectableBoxCreator(args.env.container_size, args.env.box_size_set, args.env.rot, seed)
    sel.autofill = False 
    env.box_creator = sel
    env.reset(seed=seed)

    print("[START] env.box_creator type:", type(env.box_creator).__name__)
    print("[START] sel is env.box_creator ?", sel is env.box_creator)

    session_id = str(uuid.uuid4())
    SESS[session_id] = {"env": env, "sel": sel, "args": args, "policy": policy, "device": device}
    ratio = env.container.get_volume_ratio() if hasattr(env.container, "get_volume_ratio") else 0.0
    return StateResp(session_id=session_id, filled_ratio=float(ratio), count=len(_boxes_list(env)), boxes=_boxes_list(env))

@app.post("/place", response_model=PlaceResp)
def place_one(req: PlaceReq):
    
    if req.session_id not in SESS:
        raise HTTPException(404, detail="Unknown session_id")
    env = SESS[req.session_id]["env"]
    sel = SESS[req.session_id]["sel"]
    policy = SESS[req.session_id]["policy"]
    device = SESS[req.session_id]["device"]  # cuda or cpu

    env._override_next = [int(req.box[0]), int(req.box[1]), int(req.box[2])]

    sel.autofill = False
    sel.override_current(req.box)
    print(f"[API] requested={tuple(req.box)} env.next_box={tuple(env.next_box)}")

    # 1) Force THIS box for both candidate generation and step
    env._override_next = [int(req.box[0]), int(req.box[1]), int(req.box[2])]
    # Do NOT reset or write box_creator here

    # 2) Build observation for the forced box
    # 2) build a single-sample Batch (with correct dtypes + device)
    cur = env.cur_observation

    # obs: float32 on model device
    obs_t = torch.as_tensor(cur["obs"], dtype=torch.float32, device=device).unsqueeze(0)

    # mask: CPU float32 (IMPORTANT: keep it on CPU so FloatTensor(...) is happy)
    # Option A: numpy
    mask_np = np.asarray(cur["mask"], dtype=np.float32)[None, ...]
    # Option B: CPU torch
    # mask_cpu = torch.as_tensor(cur["mask"], dtype=torch.float32).unsqueeze(0)  # note: no device=...

    batch = Batch(obs={"obs": obs_t, "mask": mask_np})  # or mask_cpu
    with torch.no_grad():
        out = policy.forward(batch, state=None)

    a = int(out.act.view(-1)[0].detach().cpu().item())

    # 4) Decode + step (step() will clear _override_next afterwards)
    # pos, rot, dim = env.idx2pos(a)
    # _, reward, done, truncated, info = env.step(a)

    pos, rot, dim = env.idx2pos_for(a, req.box)
    _, reward, done, truncated, info = env.step_with_box(a, req.box)

    return PlaceResp(
        session_id=req.session_id,
        placed_dim=[int(d) for d in dim],
        pos=[int(p) for p in pos],
        rot=int(rot),
        filled_ratio=float(info.get("ratio", 0.0)),
        count=int(info.get("counter", 0)),
        done=bool(done or truncated),
        boxes=_boxes_list(env),
    )


@app.get("/state/{session_id}", response_model=StateResp)
def get_state(session_id: str):
    if session_id not in SESS:
        raise HTTPException(404, detail="Unknown session_id")
    env = SESS[session_id]["env"]
    ratio = env.container.get_volume_ratio() if hasattr(env.container, "get_volume_ratio") else 0.0
    return StateResp(session_id=session_id, filled_ratio=float(ratio), count=len(_boxes_list(env)), boxes=_boxes_list(env))
