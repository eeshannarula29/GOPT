import os, sys, time
import numpy as np
import gymnasium as gym
import torch

# project imports
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

from tools import registration_envs, set_seed, CategoricalMasked
from ts_train import build_net
import arguments
from masked_ppo import MaskedPPOPolicy
from tianshou.utils.net.common import ActorCritic
from tianshou.data import Batch

# ---- selectable catalog creator (replaces RandomBoxCreator) ----
class SelectableBoxCreator:
    def __init__(self, container_size, catalog, can_rotate=False, seed=None):
        self.container_size = np.array(container_size, dtype=np.int32)
        self.catalog = np.array(catalog, dtype=np.int32)
        assert self.catalog.ndim == 2 and self.catalog.shape[1] == 3
        self.can_rotate = can_rotate
        self.rng = np.random.default_rng(seed)
        self.box_list = []
        self._forced_next = None

    def preview(self, n):
        while len(self.box_list) < n:
            self.generate_box_size()
        return [b[:] for b in self.box_list[:n]]

    def drop_box(self): self.box_list.pop(0)

    def generate_box_size(self, **kwargs):
        if self._forced_next is not None:
            self.box_list.append(self._forced_next); self._forced_next = None
        else:
            self.box_list.append(self._sample_one())

    def reset(self):
        self.box_list.clear(); self._forced_next = None

    def force_next(self, size_triplet):
        size = np.minimum(np.array(size_triplet, dtype=np.int32), self.container_size)
        self._forced_next = size.tolist()

    def override_current(self, size_triplet):
        size = np.minimum(np.array(size_triplet, dtype=np.int32), self.container_size).tolist()
        if self.box_list: self.box_list[0] = size
        else: self.box_list.append(size)

    def _sample_one(self):
        size = self.catalog[self.rng.integers(len(self.catalog))].copy()
        if self.can_rotate:
            perms = ((0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0))
            size = size[list(perms[self.rng.integers(6)])]
        return np.minimum(size, self.container_size).tolist()

def choose_from_catalog(catalog):
    print("\nChoose NEXT incoming box:")
    for i, s in enumerate(catalog):
        print(f"  {i}: {tuple(s)}")
    print("  r: random   |   q: quit")
    while True:
        s = input("your choice: ").strip().lower()
        if s == "q": return "quit", None
        if s == "r": return "random", None
        if s.isdigit() and 0 <= int(s) < len(catalog): return "index", int(s)
        print("  -> invalid input, try again.")

def main():
    registration_envs()
    args = arguments.get_args()

    # IMPORTANT: keep your catalog small here (6–7 sizes)
    # If you prefer YAML, put the same list under env.box_size_set there.
    if not getattr(args.env, "box_size_set", None):
        args.env.box_size_set = [
            (2, 2, 2),
            (2, 3, 4),
            (3, 3, 2),
            (4, 2, 2),
            (5, 3, 2),
            (3, 5, 4),
            (6, 2, 3),
        ]

    # seed & device
    if not hasattr(args, "seed") or args.seed is None:
        args.seed = int(time.time())
    set_seed(args.seed, args.cuda, args.cuda_deterministic)
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")

    # env (pass render flag to open VTK window)
    env = gym.make(
        args.env.id,
        container_size=args.env.container_size,
        enable_rotation=args.env.rot,
        data_type="random",
        item_set=args.env.box_size_set,
        reward_type=args.train.reward_type,
        action_scheme=args.env.scheme,
        k_placement=args.env.k_placement,
        is_render=args.render,          # <-- set --render to visualize
    )

    # swap in selectable catalog
    sel = SelectableBoxCreator(
        container_size=args.env.container_size,
        catalog=args.env.box_size_set,
        can_rotate=args.env.rot,
        seed=args.seed,
    )
    env.box_creator = sel

    # build & load pretrained policy
    actor, critic = build_net(args, device)
    actor_critic = ActorCritic(actor, critic).to(device)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.opt.lr, eps=args.opt.eps)
    policy = MaskedPPOPolicy(
        actor=actor, critic=critic, optim=optim,
        dist_fn=CategoricalMasked,
        discount_factor=args.train.gamma,
        eps_clip=args.train.clip_param,
        advantage_normalization=False,
        vf_coef=args.loss.value, ent_coef=args.loss.entropy,
        gae_lambda=args.train.gae_lambda,
        action_space=env.action_space,
    ).to(device)
    policy.eval()
    if not getattr(args, "ckp", None):
        print("ERROR: pass --ckp <path_to_policy_step_best.pth>"); return
    policy.load_state_dict(torch.load(args.ckp, map_location=device))

    # start episode
    ob, info = env.reset(seed=args.seed)
    print(f"\nCurrent first box (before step 0): {env.next_box}")
    if input("Override the current first box? (y/N): ").strip().lower() == "y":
        mode, idx = choose_from_catalog(args.env.box_size_set)
        if mode == "index":
            sel.override_current(args.env.box_size_set[idx])
            print(f"Overrode current to: {env.next_box}")
        elif mode == "quit":
            print("bye!"); return

    step_i = 0
    while True:
        print(f"\n[step {step_i}] current next_box: {env.next_box}")
        mode, idx = choose_from_catalog(args.env.box_size_set)
        if mode == "quit":
            print("bye!"); break
        elif mode == "index":
            sel.force_next(args.env.box_size_set[idx])
        elif mode == "random":
            sel._forced_next = None

        # ==== FIX: add batch dimension ====
        cur = env.cur_observation
        batch = Batch(obs={
            "obs": np.expand_dims(cur["obs"], 0),   # (1, obs_dim)
            "mask": np.expand_dims(cur["mask"], 0)  # (1, k_placement)
        })

        with torch.no_grad():
            out = policy.forward(batch, state=None)

        act = out.act
        if torch.is_tensor(act): a = int(act.view(-1)[0].item())
        else: a = int(np.array(act).reshape(-1)[0])

        ob, reward, done, truncated, info = env.step(a)

        # visualize this placement
        if args.render: env.render()

        step_i += 1
        ratio = info.get("ratio", float("nan"))
        placed = info.get("counter", -1)
        print(f" -> action={a}, reward={reward:.4f}, filled={ratio:.4f}, count={placed}")

        if done or truncated:
            print(f"\nEpisode finished after {step_i} steps. "
                  f"packed={placed}, space utilization={ratio:.4f}")
            break

if __name__ == "__main__":
    main()
