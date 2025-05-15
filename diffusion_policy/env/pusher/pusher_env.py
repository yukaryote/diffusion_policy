from pathlib import Path
from typing import List, Tuple
import gymnasium

import hydra
import numpy as np
from gymnasium import spaces
from mujoco_sim.env.push_env import PushEnv, PushEnvCfg

from gymnasium.utils import seeding


class PusherEnv(gymnasium.Env):

    def __init__(
        self,
        pusher_cfg: PushEnvCfg,
        **kwargs,
    ):
        super().__init__(
            **kwargs
        )
        self._seed = None
        self.seed()
        self.mujoco_env = PushEnv(pusher_cfg)
        self.goal = np.array([0.8, -0.8])

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            dict(
                image=spaces.Box(
                    low=0,
                    high=1,
                    shape=(3, 256, 256),
                    dtype=np.float32
                ),
                agent_pos=spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(2,),
                    dtype=np.float32
                )
            )
        )

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def reset(
        self,
        *,
        seed: int = None,
        options: dict = None,
    ):
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)
        self.mujoco_env.reset()
        obs = self._get_obs()

        return obs, {}
        
    def render(self, segmentation=False):
        """Render a frame of the MuJoCo simulation.

        Returns:
            rgb image (np.ndarray)
        """
        return self.mujoco_env.render(
            "birdview",
            render_segmentation=segmentation,
        )
    
    def _get_info(self):
        qpos_robot, qvel_robot = self._get_robot_obs()
        info = {
            'pos_agent': qpos_robot,
            'vel_agent': qvel_robot,
        }
        return info
    
    def _get_robot_obs(self):
        robot_qpos = self.mujoco_env.data.get_body_xpos("pusher_main")[:2]
        robot_qvel = self.mujoco_env.data.get_body_xvelp("pusher_main")[:2]

        return robot_qpos, robot_qvel

    def _get_obs(self) -> dict:
        qpos_obs = self.mujoco_env.data.get_body_xpos("pusher_main")[:2]

        img = self.mujoco_env.render("birdview")
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)

        obs = {
            'image': img_obs,
            'agent_pos': qpos_obs
        }

        return obs
    
    def _mujoco_step(self, action):
        return self.mujoco_env.step(action)
    
    def step(self, action):
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (np.ndarray): Control action to be applied to the agent and update the simulation. Should be of shape :attr:`action_space`.

        Returns:
            observation (dictionary): Next observation due to the agent actions .It should satisfy the `GoalEnv` :attr:`observation_space`.
            reward (integer): The reward as a result of taking the action. This is calculated by :meth:`compute_reward` of `GoalEnv`.
            terminated (boolean): Whether the agent reaches the terminal state. This is calculated by :meth:`compute_terminated` of `GoalEnv`.
            truncated (boolean): Whether the truncation condition outside the scope of the MDP is satisfied. Timically, due to a timelimit, but
            it is also calculated in :meth:`compute_truncated` of `GoalEnv`.
            info (dictionary): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging). In this case there is a single
            key `is_success` with a boolean value, True if the `achieved_goal` is the same as the `desired_goal`.
        """
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._mujoco_step(action)

        obs = self._get_obs()

        info = self._get_info()

        achieved_goal_qpos, achieved_goal_qvel = self._get_robot_obs()

        reward = np.e ** -(np.linalg.norm(achieved_goal_qpos - self.goal)) + 0.5 * np.e ** -np.linalg.norm(achieved_goal_qvel)
        done = np.allclose(achieved_goal_qpos, self.goal, atol=1e-2)

        return obs, reward, done, done, info


if __name__ == "__main__":

    import os
    from omegaconf import DictConfig, OmegaConf
    from mujoco_sim.env.push_env import PushEnvCfg
    from jacobian.config import get_typed_root_config
    import mediapy as media

    config_path = "/data/scene-rep/u/iyu/scene-jacobian-discovery/assets/config/dataset/push_env_cfg/pusher_with_rod.yaml"
    config_name = "pusher_only"
    cfg_dict = OmegaConf.load(config_path)
    push_env_cfg = get_typed_root_config(cfg_dict=cfg_dict, cfg_type=PushEnvCfg)
    pusher_env = PusherEnv(push_env_cfg)
    vid = [pusher_env.render()]
    actions_y = np.linspace(0.36, -1.0, 50)
    for i, y in enumerate(actions_y):
        action = np.array([0., y])
        print(i, pusher_env.mujoco_env.data.get_body_xpos("pusher_main"), action)
        obs, reward, done, _, info = pusher_env.step(action)
        img = pusher_env.render()
        vid.append(img)
    media.write_video("/data/scene-rep/u/iyu/scene-jacobian-discovery/data/test_diff_policy_pusher_env.mp4", vid)
