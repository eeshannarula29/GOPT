from typing import Optional

from .container import Container
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .cutCreator import CuttingBoxCreator
# from .mdCreator import MDlayerBoxCreator
from .binCreator import RandomBoxCreator, LoadBoxCreator, BoxCreator

from render import VTKRender


class PackingEnv(gym.Env):
    def __init__(
        self,
        container_size=(10, 10, 10),
        item_set=None, 
        data_name=None, 
        load_test_data=False,
        enable_rotation=False,
        data_type="random",
        reward_type=None,
        action_scheme="heightmap",
        k_placement=100,
        is_render=False,
        is_hold_on=False,
        **kwags
    ) -> None:
        self.bin_size = container_size
        self.area = int(self.bin_size[0] * self.bin_size[1])
        # packing state
        self.container = Container(*self.bin_size, rotation=enable_rotation)
        self.can_rotate = enable_rotation
        self.reward_type = reward_type
        self.action_scheme = action_scheme
        self.k_placement = k_placement
        if action_scheme == "EMS":
            self.candidates = np.zeros((self.k_placement, 6), dtype=np.int32)  # (x1, y1, z1, x2, y2, H)
        else:
            self.candidates = np.zeros((self.k_placement, 3), dtype=np.int32)  # (x, y, z)

        # Generator for train/test data
        if not load_test_data:
            assert item_set is not None
            if data_type == "random":
                print(f"using items generated randomly")
                self.box_creator = RandomBoxCreator(item_set)  
            if data_type == "cut":
                print(f"using items generated through cutting method")
                low = list(item_set[0])
                up = list(item_set[-1])
                low.extend(up)
                self.box_creator = CuttingBoxCreator(container_size, low, self.can_rotate)
            assert isinstance(self.box_creator, BoxCreator)
        if load_test_data:
            print(f"use box dataset: {data_name}")
            self.box_creator = LoadBoxCreator(data_name)

        self.test = load_test_data

        # for rendering
        if is_render:
            self.renderer = VTKRender(container_size, auto_render=not is_hold_on)
        self.render_box = None
        
        self._set_space()

        self._override_next = None  

    def _set_space(self) -> None:
        obs_len = self.area + 3  # the state of bin + the dimension of box (l, w, h)
        obs_len += self.k_placement * 6
        self.action_space = spaces.Discrete(self.k_placement)
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(low=0, high=max(self.bin_size), shape=(obs_len, )),
                "mask": spaces.Discrete(self.k_placement)
            }
        )

    def get_box_ratio(self):
        coming_box = self.next_box
        return (coming_box[0] * coming_box[1] * coming_box[2]) / (
                self.container.dimension[0] * self.container.dimension[1] * self.container.dimension[2])

    # box mask (W x L x 3)
    def get_box_plain(self):
        coming_box = self.next_box
        x_plain = np.ones(self.container.dimension[:2], dtype=np.int32) * coming_box[0]
        y_plain = np.ones(self.container.dimension[:2], dtype=np.int32) * coming_box[1]
        z_plain = np.ones(self.container.dimension[:2], dtype=np.int32) * coming_box[2]
        return x_plain, y_plain, z_plain

    @property
    def cur_observation(self):
        """
            get current observation and action mask
        """
        hmap = self.container.heightmap
        size = list(self.next_box)
        placements, mask = self.get_possible_position(size)
        self.candidates = np.zeros_like(self.candidates)
        if len(placements) != 0:
            # print("candidates:")
            # for c in placements:
            #     print(c)
            self.candidates[0:len(placements)] = placements

        size.extend([size[1], size[0], size[2]])
        obs = np.concatenate((hmap.reshape(-1), np.array(size).reshape(-1), self.candidates.reshape(-1)))
        mask = mask.reshape(-1)
        return {
            "obs": obs, 
            "mask": mask
        }

    @property
    def next_box(self) -> list:
        # If API forced a specific current box, use it
        if self._override_next is not None:
            return list(self._override_next)

        # Otherwise rely on the creator's queue (API may have injected one)
        prev = self.box_creator.preview(1)
        if prev:                       # non-empty queue → use it
            return prev[0]

        # No forced box and no queued box: API mode expects client to set it
        raise RuntimeError(
            "No current box available. In API mode, call override_current(...) before reading next_box."
        )

    def get_possible_position(self, next_box):
        """
            get possible actions for next box
        Args:
            scheme: the scheme how to generate candidates

        Returns:
            candidate action mask, i.e., the position where the current item should be placed
        """
        if self.action_scheme == "heightmap":
            candidates = self.container.candidate_from_heightmap(next_box, self.k_placement)
        elif self.action_scheme == "EP":
            candidates, mask = self.container.candidate_from_EP(next_box, self.k_placement)
        elif self.action_scheme == "EMS":
            candidates, mask = self.container.candidate_from_EMS(next_box, self.k_placement)
        elif self.action_scheme == "FC": # full coordinate space
            candidates, mask = self.container.candidate_from_FC(next_box)
        else:
            raise NotImplementedError("action scheme not implemented")

        return candidates, mask 

    def idx2pos(self, idx):
        if idx >= self.k_placement:# - 1:
            idx = idx - self.k_placement
            rot = 1
        else:
            rot = 0

        pos = self.candidates[idx][:3]

        if rot == 1:
            dim = [self.next_box[1], self.next_box[0], self.next_box[2]]
        else:
            dim = list(self.next_box)
        self.render_box = [dim, pos]

        return pos, rot, dim

    def step(self, action):
        pos, rot, size = self.idx2pos(action)
        succeeded = self.container.place_box(self.next_box, pos, rot)

        if not succeeded:
            reward = self.container.get_volume_ratio() if self.reward_type == "terminal" else 0.0
            done = True
            self.render_box = [[0, 0, 0], [0, 0, 0]]
            info = {'counter': len(self.container.boxes), 'ratio': self.container.get_volume_ratio()}
            return self.cur_observation, reward, done, False, info

        box_ratio = self.get_box_ratio()

        # Consume any queued current box (if any)
        try:
            self.box_creator.drop_box()
        except Exception:
            pass

        # ✅ Only auto-generate for training mode
        if getattr(self.box_creator, "autofill", False):
            self.box_creator.generate_box_size()

        # ✅ Clear forced override after a successful placement
        self._override_next = None

        reward = 0.01 if self.reward_type == "terminal" else box_ratio
        done = False
        info = {'counter': len(self.container.boxes), 'ratio': self.container.get_volume_ratio()}
        return self.cur_observation, reward, done, False, info

    def idx2pos_for(self, idx, size):
        if idx >= self.k_placement:
            idx -= self.k_placement
            rot = 1
        else:
            rot = 0

        pos = self.candidates[idx][:3]
        dim = [size[1], size[0], size[2]] if rot == 1 else list(size)
        self.render_box = [dim, pos]
        return pos, rot, dim

    def step_with_box(self, action, size):
        pos, rot, dim = self.idx2pos_for(action, size)
        succeeded = self.container.place_box(size, pos, rot)

        if not succeeded:
            reward = self.container.get_volume_ratio() if self.reward_type == "terminal" else 0.0
            done = True
            self.render_box = [[0, 0, 0], [0, 0, 0]]
            info = {'counter': len(self.container.boxes), 'ratio': self.container.get_volume_ratio()}
            return self.cur_observation, reward, done, False, info

        box_ratio = (size[0]*size[1]*size[2])/(self.container.dimension[0]*self.container.dimension[1]*self.container.dimension[2])

        # Consume queued item if present (safe in API mode)
        try:
            self.box_creator.drop_box()
        except Exception:
            pass

        if getattr(self.box_creator, "autofill", False):
            self.box_creator.generate_box_size()

        self._override_next = None

        reward = 0.01 if self.reward_type == "terminal" else box_ratio
        done = False
        info = {'counter': len(self.container.boxes), 'ratio': self.container.get_volume_ratio()}
        return self.cur_observation, reward, done, False, info


    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)
        self.box_creator.reset()
        self.container = Container(*self.bin_size, rotation=self.can_rotate)
        # self._override_next = None
        self.candidates = np.zeros_like(self.candidates)

        # Only auto-generate in training mode (autofill==True)
        if getattr(self.box_creator, "autofill", False):
            self.box_creator.generate_box_size()

        return self.cur_observation, {}
    
    def seed(self, s=None):
        np.random.seed(s)

    def render(self):
        self.renderer.add_item(self.render_box[0], self.render_box[1])
        # self.renderer.save_img()
