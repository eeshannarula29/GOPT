# serve_packing_api.py
import os, sys, time, uuid
from typing import List, Optional, Dict, Any
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import collections
from enum import IntEnum

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

class Rotation(IntEnum):
    XYZ = 1  # (l,w,h)
    XZY = 2  # (l,h,w)
    YXZ = 3  # (w,l,h)
    YZX = 4  # (w,h,l)
    ZXY = 5  # (h,l,w)
    ZYX = 6  # (h,w,l)

def apply_rotation(dim, rot: int):
    l,w,h = dim
    if rot == Rotation.XYZ: return [l,w,h]
    if rot == Rotation.XZY: return [l,h,w]
    if rot == Rotation.YXZ: return [w,l,h]
    if rot == Rotation.YZX: return [w,h,l]
    if rot == Rotation.ZXY: return [h,l,w]
    if rot == Rotation.ZYX: return [h,w,l]
    return [l,w,h]

class OfflineItem(BaseModel):
    id: str
    dim: List[int]                 # [l,w,h]
    count: int = 1
    weight: float = 0.0
    allowed_rots: Optional[List[int]] = None  # list of 1..6 (Rotation)
    must_upright: bool = False      

class OfflinePackReq(BaseModel):
    bin_size: List[int]                     # [L,W,H]
    max_load: Optional[float] = None       # per-bin weight cap
    items: List[OfflineItem]

    # --- GA params ---
    use_ga: bool = False
    pop_size: int = 120
    iters: int = 250
    cx_prob: float = 0.75
    mut_prob: float = 0.15
    tabu_len: int = 120
    tournament_k: int = 3

    # --- Cargo constraints (paper-inspired; all optional) ---
    # keep CoG of the whole bin inside a window (fractions of L/W)
    cog_window_x: Optional[List[float]] = None   # e.g., [0.2, 0.8]
    cog_window_y: Optional[List[float]] = None   # e.g., [0.2, 0.8]
    # per-item “upright” convenience already supported via must_upright

class OfflinePlacedBox(BaseModel):
    item_id: str
    dim: List[int]          # placed dims (after rotation)
    pos: List[int]          # [x,y,z] LLB
    rot: int                # 1..6 (Rotation)
    weight: float

class OfflineBin(BaseModel):
    boxes: List[OfflinePlacedBox]
    filled_ratio: float
    load: float

class OfflinePackResp(BaseModel):
    bins: List[OfflineBin]
    total_bins: int

# ---- Heuristic loader (COHLA-style) ----
class _Space:
    __slots__ = ("x","y","z","L","W","H")
    def __init__(self, x,y,z,L,W,H):
        self.x,self.y,self.z = int(x),int(y),int(z)
        self.L,self.W,self.H = int(L),int(W),int(H)
    def vol(self): return self.L*self.W*self.H

class _BinState:
    def __init__(self, bin_size, max_load=None):
        self.L,self.W,self.H = bin_size
        self.max_load = max_load if max_load is not None else float("inf")
        self.load = 0.0
        self.boxes: List[OfflinePlacedBox] = []
        # start with one full space
        self.spaces: List[_Space] = [_Space(0,0,0,self.L,self.W,self.H)]

    def utilization(self):
        used = sum(b.dim[0]*b.dim[1]*b.dim[2] for b in self.boxes)
        return used / float(self.L*self.W*self.H)

    def can_accept_weight(self, w): return (self.load + w) <= self.max_load + 1e-9

class CornerOccupyHeuristic:
    """
    Minimal COHLA scaffold:
    - pick smallest usable space (min volume) that fits the rotated box
    - place at LLB corner
    - split remaining space into up/front/right (choose vertical vs horizontal by area diff heuristic)
    - merge adjacent 'abandoned' small spaces (very simple pass)
    """
    @staticmethod
    def _fits(space:_Space, dim:List[int]) -> bool:
        l,w,h = dim
        return (l <= space.L and w <= space.W and h <= space.H)

    @staticmethod
    def _choose_space(spaces:List[_Space], dim:List[int]) -> Optional[int]:
        usable = [ (i,s) for i,s in enumerate(spaces) if CornerOccupyHeuristic._fits(s, dim) ]
        if not usable: return None
        # Minimum-space rule to reduce fragmentation
        usable.sort(key=lambda t: t[1].vol())
        return usable[0][0]

    @staticmethod
    def _split(space:_Space, placed_pos:List[int], dim:List[int]) -> List[_Space]:
        """Return new spaces after placing 'dim' at (x,y,z) in 'space'."""
        x,y,z = placed_pos
        l,w,h = dim
        # upper space directly above placed box (same x,y, but reduced H)
        up = _Space(x, y, z+h, l, w, max(0, space.H - h))
        # right/front splitting decision by area diff (avoid skinny slivers)
        # compute areas if split vertical vs horizontal
        # vertical right area = (space.L - l) * space.W
        # horizontal front area = space.L * (space.W - w)
        d = (space.L - l)*space.W - space.L*(space.W - w)
        # right space
        right = _Space(x+l, y, z, max(0, space.L - l), w, h)
        # front space
        front = _Space(x, y+w, z, l, max(0, space.W - w), h)
        # also the remainder slab beyond up/right/front along height/length/width
        # extend right/front to full height available at that slab level
        right.H = h
        front.H = h
        # The leftover 'ceiling' chunk to the right-front-up:
        # split preference: if d > 0 choose vertical-first; we already created both; keep both
        new_spaces = []
        if up.H > 0: new_spaces.append(up)
        if right.L > 0 and right.W > 0 and right.H > 0: new_spaces.append(right)
        if front.L > 0 and front.W > 0 and front.H > 0: new_spaces.append(front)

        # add the remainder of original space not covered by placed volume bands:
        # band along +X beyond l, +W beyond w, +H beyond h
        # slab A: (x+l, y, z+h) size (space.L-l, w, space.H-h)
        if space.L - l > 0 and w > 0 and space.H - h > 0:
            new_spaces.append(_Space(x+l, y, z+h, space.L - l, w, space.H - h))
        # slab B: (x, y+w, z+h) size (l, space.W-w, space.H-h)
        if l > 0 and space.W - w > 0 and space.H - h > 0:
            new_spaces.append(_Space(x, y+w, z+h, l, space.W - w, space.H - h))
        # slab C: (x+l, y+w, z) size (space.L-l, space.W-w, h)
        if space.L - l > 0 and space.W - w > 0 and h > 0:
            new_spaces.append(_Space(x+l, y+w, z, space.L - l, space.W - w, h))
        return new_spaces

    @staticmethod
    def _merge_adjacent(spaces:List[_Space]) -> List[_Space]:
        # very simple O(n^2) pass: merge coplanar, same size along two axes, contiguous along the third
        merged = True
        S = spaces[:]
        while merged:
            merged = False
            out = []
            used = [False]*len(S)
            for i,a in enumerate(S):
                if used[i]: continue
                merged_once = False
                for j,b in enumerate(S):
                    if i==j or used[j]: continue
                    # try merge along X (left-right adjacency)
                    if (a.y==b.y and a.z==b.z and a.W==b.W and a.H==b.H):
                        if a.x + a.L == b.x:  # a before b
                            out.append(_Space(a.x, a.y, a.z, a.L+b.L, a.W, a.H))
                            used[i]=used[j]=True; merged = merged_once = True; break
                        if b.x + b.L == a.x:  # b before a
                            out.append(_Space(b.x, b.y, b.z, b.L+a.L, b.W, b.H))
                            used[i]=used[j]=True; merged = merged_once = True; break
                    # try merge along Y (front-back adjacency)
                    if (a.x==b.x and a.z==b.z and a.L==b.L and a.H==b.H):
                        if a.y + a.W == b.y:
                            out.append(_Space(a.x, a.y, a.z, a.L, a.W+b.W, a.H))
                            used[i]=used[j]=True; merged = merged_once = True; break
                        if b.y + b.W == a.y:
                            out.append(_Space(b.x, b.y, b.z, b.L, b.W+a.W, b.H))
                            used[i]=used[j]=True; merged = merged_once = True; break
                    # try merge along Z (stacked)
                    if (a.x==b.x and a.y==b.y and a.L==b.L and a.W==b.W):
                        if a.z + a.H == b.z:
                            out.append(_Space(a.x, a.y, a.z, a.L, a.W, a.H+b.H))
                            used[i]=used[j]=True; merged = merged_once = True; break
                        if b.z + b.H == a.z:
                            out.append(_Space(b.x, b.y, b.z, b.L, b.W, b.H+a.H))
                            used[i]=used[j]=True; merged = merged_once = True; break
                if not merged_once and not used[i]:
                    out.append(a)
            S = out
        return S

    def place_one(self, bin_state:_BinState, item_id:str, base_dim:List[int],
                  allowed_rots:List[int], weight:float) -> Optional[OfflinePlacedBox]:
        # try all allowed rotations, prefer tightest (min leftover volume space)
        best = None
        for rot in allowed_rots:
            dim = apply_rotation(base_dim, rot)
            idx = self._choose_space(bin_state.spaces, dim)
            if idx is None: continue
            space = bin_state.spaces[idx]
            # place at (space.x, space.y, space.z)
            leftover = (space.L*space.W*space.H) - (dim[0]*dim[1]*dim[2])
            cand = (leftover, idx, rot, dim, [space.x, space.y, space.z])
            if best is None or cand < best:
                best = cand
        if best is None: return None
        _, idx, rot, dim, pos = best
        if not bin_state.can_accept_weight(weight): return None

        # split space and update space list
        old = bin_state.spaces.pop(idx)
        new_spaces = self._split(old, pos, dim)
        bin_state.spaces.extend(new_spaces)
        bin_state.spaces = self._merge_adjacent(bin_state.spaces)

        placed = OfflinePlacedBox(item_id=item_id, dim=[int(x) for x in dim],
                                  pos=[int(x) for x in pos], rot=int(rot), weight=float(weight))
        bin_state.boxes.append(placed)
        bin_state.load += float(weight)
        return placed


# ---- GA optimizer (TSRNEM + crossover + mutation + tabu) ----
class OfflineGAOptimizer:
    """
    Offline 3D bin packing optimizer:
    - Two-segment random key chromosome:
        * first N keys -> order (argsort)
        * next N keys -> rotation choice per item (index into allowed_rots)
    - Fitness: prioritizes fewer bins, then higher utilization.
    - Selection schedule: roulette for early generations; elitism + tournament later.
    - Crossover: arithmetic blend for order-keys; uniform / blend for rot-keys
    - Mutation: gaussian jitter on a subset of keys
    - Tabu: hashes of decoded (order+rot) to avoid cycling; plus a light local-search tweak.
    """

    def __init__(self, req: OfflinePackReq):
        self.req = req
        self.rng = np.random.default_rng(int(time.time()))
        self._items_expanded = None  # filled by seed_sequence()
        self._N = 0

    # ---------- helpers ----------
    def build_allowed_rots(self, it: OfflineItem) -> List[int]:
        if it.allowed_rots:
            return [int(r) for r in it.allowed_rots]
        if it.must_upright:
            # preserve Z "up" (no flipping Z). Allow (XYZ, YXZ)
            return [int(Rotation.XYZ), int(Rotation.YXZ)]
        return [1,2,3,4,5,6]

    def seed_sequence(self) -> List[Dict[str,Any]]:
        # Expand counts, attach allowed_rots and volume for sorting
        expanded = []
        for it in self.req.items:
            rots = self.build_allowed_rots(it)
            for k in range(int(it.count)):
                vol = it.dim[0]*it.dim[1]*it.dim[2]
                expanded.append({
                    "id": f"{it.id}#{k+1}",
                    "base_dim": it.dim,
                    "weight": it.weight,
                    "allowed_rots": rots,
                    "vol": vol
                })
        expanded.sort(key=lambda x: (-x["vol"], x["id"]))  # FFD seed
        self._items_expanded = expanded
        self._N = len(expanded)
        return expanded

    def _decode(self, chrom: np.ndarray) -> List[Dict[str,Any]]:
        """
        chrom: 2N-length vector in [0,1)
        -> decode order and per-item rotation
        """
        N = self._N
        assert chrom.shape[0] == 2*N
        order_keys = chrom[:N]
        rot_keys   = chrom[N:]

        idx_sorted = np.argsort(order_keys, kind="stable")
        seq = []
        for rank, item_idx in enumerate(idx_sorted):
            job = self._items_expanded[int(item_idx)]
            rots = job["allowed_rots"]
            # map key to a rotation index
            ridx = int(np.floor(rot_keys[item_idx] * len(rots))) % len(rots)
            seq.append({
                "id": job["id"],
                "base_dim": job["base_dim"],
                "weight": job["weight"],
                "allowed_rots": rots,
                "rot_choice": rots[ridx],  # fixed rotation for this decode
                "vol": job["vol"]
            })
        return seq

    def _evaluate_sequence(self, seq_fixed_rot: List[Dict[str,Any]]) -> Dict[str,Any]:
        """
        Pack the fixed (order, rotation) plan using CornerOccupyHeuristic and
        return fitness info. CoG window constraints penalize if violated.
        """
        L, W, H = self.req.bin_size
        bins: List[_BinState] = []
        loader = CornerOccupyHeuristic()

        for job in seq_fixed_rot:
            placed = False
            for b in bins:
                p = loader.place_one(
                    b, job["id"], job["base_dim"],
                    allowed_rots=[job["rot_choice"]],  # fixed rotation
                    weight=job["weight"]
                )
                if p is not None:
                    placed = True
                    break
            if not placed:
                nb = _BinState([L,W,H], max_load=self.req.max_load)
                p = loader.place_one(
                    nb, job["id"], job["base_dim"],
                    allowed_rots=[job["rot_choice"]],
                    weight=job["weight"]
                )
                if p is None:
                    # does not fit empty bin under fixed rotation -> hard penalty
                    return {"bins": None, "fitness": -1e9, "viol": 1}
                bins.append(nb)

        # compute fitness
        num_bins = len(bins)
        util_sum = sum(b.utilization() for b in bins)
        util_avg = util_sum / max(1, num_bins)

        # cargo CoG window check (soft penalty)
        penalty = 0.0
        if self.req.cog_window_x or self.req.cog_window_y:
            for b in bins:
                # mass center in xy (z ignored for the window check in many cargo settings)
                total_w = sum(bb.weight for bb in b.boxes) or 1.0
                cx = sum((bb.pos[0] + 0.5*bb.dim[0]) * bb.weight for bb in b.boxes) / total_w
                cy = sum((bb.pos[1] + 0.5*bb.dim[1]) * bb.weight for bb in b.boxes) / total_w
                # normalize to [0,1] across L/W
                rx = cx / b.L
                ry = cy / b.W
                if self.req.cog_window_x:
                    lo, hi = self.req.cog_window_x
                    if rx < lo: penalty += (lo - rx)
                    if rx > hi: penalty += (rx - hi)
                if self.req.cog_window_y:
                    lo, hi = self.req.cog_window_y
                    if ry < lo: penalty += (lo - ry)
                    if ry > hi: penalty += (ry - hi)

        # Fitness: larger is better.
        # Prioritize fewer bins; tie-break by average utilization; penalize CoG deviation.
        fitness = 1e6 * (-num_bins) + 1e3 * util_avg - 1e2 * penalty
        return {"bins": bins, "fitness": float(fitness), "util": util_avg, "pen": penalty}

    def _hash_plan(self, seq_fixed_rot: List[Dict[str,Any]]) -> int:
        # light-weight plan hash: order of ids + rot choices
        tup = tuple( (j["id"], int(j["rot_choice"])) for j in seq_fixed_rot )
        return hash(tup)

    # ---------- GA ops ----------
    def _init_population(self, seed_ffd: List[Dict[str,Any]]) -> np.ndarray:
        N = self._N
        pop = []
        # Seed individual: order keys = ranks of FFD; rot keys = 0.5 (neutral)
        ranks = np.zeros(N, dtype=np.float32)
        # map item id order to ranks
        id2rank = {seed_ffd[i]["id"]: i for i in range(N)}
        for i in range(N):
            ranks[i] = id2rank[self._items_expanded[i]["id"]] / max(1, N-1)
        seed_keys = np.concatenate([ranks, np.full(N, 0.5, dtype=np.float32)])
        pop.append(seed_keys)

        # Rest random
        for _ in range(self.req.pop_size - 1):
            chrom = self.rng.random(2*N, dtype=np.float32)
            pop.append(chrom)
        return np.stack(pop, axis=0)  # (P, 2N)

    def _roulette_pick(self, fits: np.ndarray, k: int) -> np.ndarray:
        # shift to positive
        fmin = fits.min()
        probs = (fits - fmin + 1e-6)
        probs = probs / probs.sum()
        idxs = self.rng.choice(len(fits), size=k, replace=True, p=probs)
        return idxs

    def _tournament_pick(self, fits: np.ndarray, k: int, tsize: int) -> np.ndarray:
        sel = []
        P = len(fits)
        for _ in range(k):
            S = self.rng.choice(P, size=tsize, replace=False)
            best = S[np.argmax(fits[S])]
            sel.append(best)
        return np.array(sel, dtype=np.int64)

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # arithmetic blend on first N (order keys), uniform on last N (rot keys)
        N = self._N
        alpha = self.rng.random()
        child = np.empty_like(a)
        # order keys
        child[:N] = alpha*a[:N] + (1-alpha)*b[:N]
        # rotation keys
        mask = self.rng.random(N) < 0.5
        child[N:][mask]  = a[N:][mask]
        child[N:][~mask] = b[N:][~mask]
        return np.clip(child, 0.0, 1.0, out=child)

    def _mutate(self, c: np.ndarray, sigma=0.07, rate=0.15):
        N2 = c.shape[0]
        mask = self.rng.random(N2) < rate
        noise = self.rng.normal(0.0, sigma, size=N2).astype(np.float32)
        c[mask] = np.clip(c[mask] + noise[mask], 0.0, 1.0)

    def _local_search_rotate_flip(self, chrom: np.ndarray):
        """Tiny LS: nudge a few rotation keys to neighbors."""
        N = self._N
        idxs = self.rng.choice(N, size=max(1, N//20), replace=False)
        for i in idxs:
            # push rotation key a bit to change discrete bucket
            delta = self.rng.choice([-0.2, 0.2])
            chrom[N+i] = float(np.clip(chrom[N+i] + delta, 0.0, 1.0))

    def optimize_sequence(self, seed: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        if not self.req.use_ga or self._N == 0:
            # deterministic baseline: FFD + per-item best rotation (try all rots per item greedily)
            # To keep it simple and fast, return seed with rotations = pick first allowed (can be improved later).
            fixed = []
            for j in seed:
                fixed.append({**j, "rot_choice": j["allowed_rots"][0]})
            return fixed

        P = self.req.pop_size
        G = self.req.iters
        pop = self._init_population(seed)
        tabu = collections.deque(maxlen=self.req.tabu_len)
        best_fit = -np.inf
        best_plan = None

        for g in range(G):
            # Evaluate population
            fits = np.zeros(P, dtype=np.float64)
            plans = [None]*P
            for i in range(P):
                seq = self._decode(pop[i])
                h = self._hash_plan(seq)
                # tabu check: if exact plan recently seen, shake rot keys slightly before eval
                if h in tabu:
                    self._local_search_rotate_flip(pop[i])
                    seq = self._decode(pop[i])
                    h = self._hash_plan(seq)

                res = self._evaluate_sequence(seq)
                plans[i] = (seq, res)
                fits[i] = res["fitness"]

            # track best
            i_best = int(np.argmax(fits))
            if fits[i_best] > best_fit:
                best_fit = float(fits[i_best])
                best_plan = plans[i_best][0]
                # record tabu on improvement too
                tabu.append(self._hash_plan(best_plan))

            # Selection schedule: early roulette, later elitism + tournament
            if g < (2*G)//3:
                parent_idx = self._roulette_pick(fits, P)
            else:
                # keep top K elites
                elites_k = max(2, P//10)
                elite_idx = np.argsort(-fits)[:elites_k]
                rest_idx  = self._tournament_pick(fits, P - elites_k, self.req.tournament_k)
                parent_idx = np.concatenate([elite_idx, rest_idx], axis=0)

            # Breed next generation
            next_pop = []
            # elitism (carry elites if in late stage)
            if g >= (2*G)//3:
                for ei in elite_idx:
                    next_pop.append(pop[ei].copy())

            while len(next_pop) < P:
                i, j = self.rng.choice(parent_idx, size=2, replace=False)
                a, b = pop[i], pop[j]
                child = a.copy()
                if self.rng.random() < self.req.cx_prob:
                    child = self._crossover(a, b)
                if self.rng.random() < self.req.mut_prob:
                    self._mutate(child)
                # occasional local search & tabu avoid
                if self.rng.random() < 0.1:
                    self._local_search_rotate_flip(child)
                # avoid tabu exact duplicate
                seq_child = self._decode(child)
                if self._hash_plan(seq_child) in tabu:
                    self._mutate(child, sigma=0.15, rate=0.4)
                    seq_child = self._decode(child)
                next_pop.append(child)

            pop = np.stack(next_pop, axis=0)

            # push a few best hashes into tabu to encourage exploration
            elite_hash = self._hash_plan(plans[i_best][0])
            tabu.append(elite_hash)

        # return best decoded plan (order + fixed rotation)
        return best_plan


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


# ---- API entrypoint ----
@app.post("/offline/pack", response_model=OfflinePackResp)
def offline_pack(req: OfflinePackReq):
    # Prepare GA/sequence
    optimizer = OfflineGAOptimizer(req)
    seed = optimizer.seed_sequence()
    plan = optimizer.optimize_sequence(seed)

    # Greedy packing using heuristic; open new bin as needed
    bins: List[_BinState] = []
    loader = CornerOccupyHeuristic()
    L,W,H = req.bin_size

    for job in plan:
        placed = False
        for b in bins:
            p = loader.place_one(b, job["id"], job["base_dim"], job["allowed_rots"], job["weight"])
            if p is not None:
                placed = True
                break
        if not placed:
            nb = _BinState([L,W,H], max_load=req.max_load)
            p = loader.place_one(nb, job["id"], job["base_dim"], job["allowed_rots"], job["weight"])
            if p is None:
                raise HTTPException(422, detail=f"Item {job['id']} does not fit in an empty bin with allowed rotations.")
            bins.append(nb)

    resp_bins: List[OfflineBin] = []
    for b in bins:
        util = b.utilization()
        resp_bins.append(OfflineBin(
            boxes=b.boxes,
            filled_ratio=float(util),
            load=float(b.load)
        ))
    return OfflinePackResp(bins=resp_bins, total_bins=len(resp_bins))