# serve_packing_api.py
import os, sys, time, uuid
from typing import List, Optional, Dict, Any
import io, base64, numpy as np
import io, base64
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


from types import SimpleNamespace as NS
import uuid

from tianshou.data import Batch
from tianshou.data import Batch, to_torch

from threading import Lock
SESS_LOCKS: dict[str, Lock] = {}

def _get_lock(session_id: str) -> Lock:
    if session_id not in SESS_LOCKS:
        SESS_LOCKS[session_id] = Lock()
    return SESS_LOCKS[session_id]


def _heightmap_png_b64(env, title="Heightmap"):
    
    hm = getattr(env.container, "heightmap", None)
    if hm is None:
        return None
    hm = np.array(hm)
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(hm, cmap="viridis", origin="lower")
    fig.colorbar(im, ax=ax, label="Height (Z)")
    ax.set_title(title); ax.set_xlabel("X (width)"); ax.set_ylabel("Y (depth)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def visualize_ems_colored(env, box_size, mask_1d, picked=None, title="EMS before placement"):
    """
    Draw EMS rectangles color-coded by feasibility given the CURRENT mask.
    - Green: any valid action for that EMS (no-rot or rot)
    - Red:   no valid action
    - Gold outline + ★: picked action (if provided)

    Args:
      env: your PackingEnv
      box_size: [w,l,h] of current box
      mask_1d:  1D numpy array (len K or 2K) with 1/0 valid flags
      picked:   optional action index a (0..len(mask)-1)
    Returns:
      base64 PNG string
    """
    hm = np.array(env.container.heightmap)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(hm, cmap="viridis", origin="lower")
    fig.colorbar(im, ax=ax, label="Height (Z)")
    ax.set_title(f"{title} (current box: {box_size[0]}×{box_size[1]}×{box_size[2]})")
    ax.set_xlabel("X"); ax.set_ylabel("Y")

    # Pull EMS rectangles. If explicit list missing, fall back to env.candidates (non-zero rows).
    ems_list = []
    if hasattr(env.container, "ems_list"):
        ems_list = list(env.container.ems_list)
    elif hasattr(env.container, "ems"):
        ems_list = list(env.container.ems)
    if not ems_list and hasattr(env, "candidates"):
        K = getattr(env, "k_placement", len(env.candidates))
        for row in np.array(env.candidates[:K], dtype=int):
            x1,y1,z1,x2,y2,z2 = row.tolist()
            if (x1|y1|z1|x2|y2|z2) == 0:  # zero padding
                continue
            ems_list.append([x1,y1,z1,x2,y2,z2])

    K = getattr(env, "k_placement", len(ems_list))
    mask = np.asarray(mask_1d, dtype=float).reshape(-1)
    two_halves = (mask.shape[0] == 2 * K)

    # helper: does EMS i have any valid action?
    def has_valid(i):
        if i < mask.shape[0] and mask[i] > 0.5:
            return True
        if two_halves:
            j = i + K
            if j < mask.shape[0] and mask[j] > 0.5:
                return True
        return False

    for i, e in enumerate(ems_list):
        x1, y1, z1, x2, y2, z2 = map(int, e)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue

        feasible = has_valid(i)
        face = (0.0, 1.0, 0.0, 0.25) if feasible else (1.0, 0.0, 0.0, 0.20)  # green/red with alpha
        edge = "none"

        ax.add_patch(Rectangle((x1, y1), w, h, fill=True, facecolor=face, ec=edge, lw=1.0))
        # label EMS index in contrasting color
        ax.text(x1 + w/2, y1 + h/2, str(i),
                ha="center", va="center",
                color="white" if feasible else "yellow",
                fontsize=7, weight="bold")

    # Highlight the picked action (if any)
    if picked is not None:
        try:
            pos, rot, dim = env.idx2pos_for(int(picked), box_size)
            px, py, pz = map(int, pos)
            dw, dl, dh = map(int, dim)
            ax.add_patch(Rectangle((px, py), dw, dl, fill=False, ec="gold", lw=2.0))
            ax.text(px + dw/2, py + dl/2, "★", ha="center", va="center", color="gold", fontsize=10)
        except Exception:
            pass

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

    
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
    debug_hmap_png: Optional[str] = None
    debug_valid_actions: Optional[List[Dict[str, Any]]] = None
    debug_mask_len: Optional[int] = None
    debug_k_placement: Optional[int] = None
    debug_raw_candidates: Optional[List[List[int]]] = None  # careful: large
    debug_mask: Optional[List[float]] = None                # careful: large
    debug_ems_png: Optional[str] = None

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

    env.api_mode = True

    # tie action space now that env exists
    policy.action_space = env.action_space

    # swap in selectable creator
    sel = SelectableBoxCreator(args.env.container_size, args.env.box_size_set, args.env.rot, seed)
    sel.autofill = False 
    env.box_creator = sel
    env.reset(seed=seed, defer_box=True)

    print("[START] env.box_creator type:", type(env.box_creator).__name__)
    print("[START] sel is env.box_creator ?", sel is env.box_creator)

    session_id = str(uuid.uuid4())
    SESS[session_id] = {"env": env, "sel": sel, "args": args, "policy": policy, "device": device}
    ratio = env.container.get_volume_ratio() if hasattr(env.container, "get_volume_ratio") else 0.0
    return StateResp(session_id=session_id, filled_ratio=float(ratio), count=len(_boxes_list(env)), boxes=_boxes_list(env))

@app.post("/place", response_model=PlaceResp)
def place_one(req: PlaceReq):
    # --- session lookup ---
    if req.session_id not in SESS:
        raise HTTPException(404, detail="Unknown session_id")
    
    lock = _get_lock(req.session_id)
    with lock:
        env    = SESS[req.session_id]["env"]
        sel    = SESS[req.session_id]["sel"]
        policy = SESS[req.session_id]["policy"]
        device = SESS[req.session_id]["device"]

        forced = (int(req.box[0]), int(req.box[1]), int(req.box[2]))
        env.force_next_box(forced)

        # --- force this exact box for observation + step ---
        env._override_next = [int(req.box[0]), int(req.box[1]), int(req.box[2])]
        sel.autofill = False
        sel.override_current(req.box)
        
        

        # print(f"[API] requested={tuple(req.box)} env.next_box={tuple(env.next_box)}")

        # nb = tuple(env.next_box)
        # assert nb == tuple(req.box), f"next_box desync: {nb} vs {tuple(req.box)}"

        # --- build current observation (for THIS box) ---
        cur = env.cur_observation

        # === SNAPSHOT candidates + mask BEFORE policy/step (this is the key) ===
        K          = int(env.k_placement)
        cands_snap = np.array(env.candidates, copy=True)                 # (K,6)
        mask_now   = np.asarray(cur["mask"], dtype=np.float32).copy()    # (2K,) or (K,)

        # --- policy forward using the snapshot mask ---
        obs_t = torch.as_tensor(cur["obs"], dtype=torch.float32, device=device).unsqueeze(0)
        batch = Batch(obs={"obs": obs_t, "mask": mask_now[None, ...]})
        with torch.no_grad():
            out = policy.forward(batch, state=None)
        a = int(out.act.view(-1)[0].detach().cpu().item())

        # --- validate chosen action against the SNAPSHOT mask ---
        if not (0 <= a < mask_now.shape[0]) or mask_now[a] <= 0.5:
            valid = np.flatnonzero(mask_now > 0.5)
            if valid.size == 0:
                raise HTTPException(409, detail="No feasible placement (mask has no valid indices).")
            if hasattr(out, "logits"):
                logits = out.logits.detach().cpu().numpy().reshape(-1)
                a = int(valid[np.argmax(logits[valid])])
            else:
                a = int(valid[0])

        # --- decode using the SNAPSHOT (no dependence on env after step) ---
        ems_idx      = int(a % K)
        rot_expected = 1 if a >= K else 0
        if ems_idx >= cands_snap.shape[0]:
            raise HTTPException(500, detail="EMS index out of range (snapshot).")

        x1, y1, z1, x2, y2, z2 = map(int, cands_snap[ems_idx])
        w, l, h = map(int, req.box)
        if rot_expected:
            dw, dl, dh = l, w, h
        else:
            dw, dl, dh = w, l, h
        pos_dec = (x1, y1, z1)
        dim_dec = (dw, dl, dh)

        # --- hard bounds guard using container dims ---
        X, Y, Z = map(int, env.container.dimension)
        px, py, pz = pos_dec
        if not (0 <= px and 0 <= py and 0 <= pz and px+dw <= X and py+dl <= Y and pz+dh <= Z):
            # quick rescue: try the other half if valid in SNAPSHOT
            other = a + K if a < K else a - K
            if 0 <= other < mask_now.shape[0] and mask_now[other] > 0.5:
                a = int(other)
                ems_idx      = int(a % K)
                rot_expected = 1 if a >= K else 0
                x1, y1, z1, x2, y2, z2 = map(int, cands_snap[ems_idx])
                dw, dl = (l, w) if rot_expected else (w, l)
                pos_dec = (x1, y1, z1)
                dim_dec = (dw, dl, h)
                px, py, pz = pos_dec
            if not (0 <= px and 0 <= py and 0 <= pz and px+dw <= X and py+dl <= Y and pz+dh <= Z):
                raise HTTPException(409, detail="Decoded placement out of bounds (snapshot).")

        # --- optional cross-check: idx2pos_for agrees with snapshot decode? ---
        pos_chk, rot_chk, dim_chk = env.idx2pos_for(a, req.box)
        if (tuple(map(int, pos_chk)) != pos_dec) or (tuple(map(int, dim_chk)) != tuple(map(int, dim_dec))):
            print(f"[WARN] idx2pos_for disagrees with snapshot decode: "
                f"snapshot pos={pos_dec} dim={dim_dec} | "
                f"idx2pos pos={tuple(map(int,pos_chk))} dim={tuple(map(int,dim_chk))}, a={a}, K={K}")

        # --- step the env exactly once (env will rebuild cands AFTER this) ---
        _, reward, done, truncated, info = env.step_with_box(a, req.box)

        # --- EMS BEFORE placement (color-coded) built with the SNAPSHOT mask ---
        ems_png = visualize_ems_colored(env, req.box, mask_now, picked=a,
                                        title="EMS before placement")

        # --- heightmap AFTER placement ---
        png = _heightmap_png_b64(env, title=f"Heightmap after placement #{info.get('counter','?')}")

        # --- debug prints for alignment (use SNAPSHOT EMS; not post-step) ---
        print(f"[EMS DEBUG] Using SNAPSHOT EMS #{ems_idx}: ({x1},{y1},{z1})–({x2},{y2},{z2}), "
            f"picked a={a}, rot_expected={rot_expected}")
        print(f"[EMS DEBUG] Placed box pos={pos_dec}, dim={dim_dec}")

        # --- response ---
        return PlaceResp(
            session_id=req.session_id,
            placed_dim=[int(d) for d in dim_dec],         # from snapshot decode
            pos=[int(p) for p in pos_dec],                # from snapshot decode
            rot=int(rot_expected),
            filled_ratio=float(info.get("ratio", 0.0)),
            count=int(info.get("counter", 0)),
            done=bool(done or truncated),
            boxes=_boxes_list(env),
            debug_hmap_png=png,
            debug_k_placement=int(K),
            debug_mask_len=int(mask_now.shape[0]),
            debug_raw_candidates=cands_snap.tolist(),     # snapshot, not post-step
            debug_mask=mask_now.astype(float).tolist(),   # snapshot, not post-step
            debug_ems_png=ems_png,
            debug_valid_actions=None,
        )

@app.get("/state/{session_id}", response_model=StateResp)
def get_state(session_id: str):
    if session_id not in SESS:
        raise HTTPException(404, detail="Unknown session_id")
    env = SESS[session_id]["env"]
    ratio = env.container.get_volume_ratio() if hasattr(env.container, "get_volume_ratio") else 0.0
    return StateResp(session_id=session_id, filled_ratio=float(ratio), count=len(_boxes_list(env)), boxes=_boxes_list(env))
