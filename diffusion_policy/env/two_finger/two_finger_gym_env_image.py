from pathlib import Path
from typing import List, Tuple

import hydra
import numpy as np
from gym_wrapper.envs.base_gym_env import ActionType, GoalType, ObsType
from gymnasium import error, logger, spaces
from gymnasium_robotics.utils import rotations
from hydra import compose, initialize
from jacobian.config.common import get_typed_root_config
from mujoco_sim.env.two_finger_env import TwoFingerEnv, TwoFingerEnvCfg

from gym_wrapper.envs.base_gym_env import BaseGoalEnv
from gym_wrapper.envs.two_finger_gym_env import TwoFingerGoalEnv
from gym_wrapper.envs.planar_arm.manipulate_block_shadow import ShadowArmBlockEnv
from gymnasium.utils import seeding


class TwoFingerGoalEnvImage(ShadowArmBlockEnv):

    def __init__(
        self,
        state_type: str = "qpos",
        fixed_goal: np.ndarray = None,
        **kwargs,
    ):
        if state_type == "qpos":
            self.state_dim = 12
        elif state_type == "qvel":
            self.state_dim = 18
        elif state_type == "stateless":
            self.state_dim = 0
        else:
            raise ValueError(f"state_type {state_type} not supported")
        action_dim = 4
        super().__init__(
            **kwargs
        )
        self.fixed_goal = fixed_goal
        if self.fixed_goal is not None:
            self.goal = self.fixed_goal

        self.action_space = spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        if self.state_dim > 0:
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
                        shape=(self.state_dim,),
                        dtype=np.float32
                    )
                )
            )
        else:
            self.observation_space = spaces.Dict(
                dict(
                    image=spaces.Box(
                        low=0,
                        high=1,
                        shape=(3, 256, 256),
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
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        if self.fixed_goal is None:
            self.goal = self._sample_goal().copy()
        else:
            self.goal = self.fixed_goal
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        return obs, {}
        
    def render(self, mode="rgb_array", segmentation=False):
        """Render a frame of the MuJoCo simulation.

        Returns:
            rgb image (np.ndarray): if render_mode is "rgb_array", return a 3D image array.
        """
        return self.mujoco_renderer.render(
            self.render_mode,
            camera_name=self.camera_name,
            segmentation=segmentation,
        )

    def get_qpos_qvel_of_list(self, joint_names):
        state = self.mujoco_env.get_state()
        qpos = state.qpos.copy()
        qvel = state.qvel.copy()

        qpos_subset = []
        qvel_subset = []
        for name in joint_names:
            qpos_addr = self.mujoco_env.sim.model.get_joint_qpos_addr(name)
            qvel_addr = self.mujoco_env.sim.model.get_joint_qvel_addr(name)
            if isinstance(qpos_addr, tuple):
                start, end = qpos_addr
                qpos_subset.append(qpos[start:end])
            else:
                qpos_subset.append(qpos[qpos_addr])

            if isinstance(qvel_addr, tuple):
                qvel_subset.append(qvel[start:end])
            else:
                qvel_subset.append(qvel[qpos_addr])

        return np.array(qpos_subset).ravel(), np.array(qvel_subset).ravel()
    
    def _get_info(self):
        qpos_robot, qvel_robot = self._get_robot_obs()
        qpos_goal, _ = self._get_achieved_goal()
        info = {
            'pos_agent': qpos_robot,
            'vel_agent': qvel_robot,
            'block_pose': qpos_goal,
            'goal_pose': self.goal
        }
        return info
    
    def _get_robot_obs(self):
        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )

        return robot_qpos, robot_qvel

    def _get_achieved_goal(self):
        object_qpos = self._utils.get_joint_qpos(self.model, self.data, "object:joint")
        object_qvel = self._utils.get_joint_qvel(self.model, self.data, "object:joint")
        assert object_qpos.shape == (7,), f"object_qpos shape not 7: {object_qpos.shape}"
        return object_qpos, object_qvel

    def _get_obs(self) -> dict:
        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        # print(robot_qpos, robot_qvel)

        assert (
            len(robot_qpos) > 0 and len(robot_qvel) > 0
        ), "robot_qpos and robot_qvel should not be empty"

        img = self.render()
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)

        def encode_angle(x):
            return np.concatenate([np.cos(x), np.sin(x)])

        qpos_obs = np.concatenate(
            [
                # robot state
                np.sin(robot_qpos),
                np.cos(robot_qpos),
                robot_qvel
            ]
        )[:self.state_dim]

        if self.state_dim > 0:
            obs = {
                'image': img_obs,
                'agent_pos': qpos_obs
            }
        else:
            obs = {
                'image': img_obs
            }

        return obs
    
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
        self._set_action(action)

        self._mujoco_step(action)

        self._step_callback()

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()

        info = self._get_info()

        achieved_goal_qpos, achieved_goal_qvel = self._get_achieved_goal()

        terminated = self.compute_terminated(achieved_goal_qpos, self.goal, info)
        truncated = self.compute_truncated(achieved_goal_qpos, self.goal, info)

        reward = self.compute_reward(achieved_goal_qpos, self.goal, info)

        return obs, reward, terminated, truncated, info
