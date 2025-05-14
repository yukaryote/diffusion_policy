import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
from omegaconf import OmegaConf
import wandb.sdk.data_types.video as wv
#from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from mujoco_sim.env.push_env import PushEnvCfg
from diffusion_policy.env.pusher.pusher_env import PusherEnv
from jacobian.config import get_typed_root_config


class PusherImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            train_start_seed=0,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            fps=10,
            crf=22,
            render_size=256,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=1,
            legacy_test=False,
            fixed_goal=True,
            push_env_cfg_path="/data/scene-rep/u/iyu/scene-jacobian-discovery/assets/config/dataset/push_env_cfg/pusher_only.yaml",
        ):
        super().__init__(output_dir)
        cfg_dict = OmegaConf.load(push_env_cfg_path)
        push_env_cfg = get_typed_root_config(cfg_dict=cfg_dict, cfg_type=PushEnvCfg)

        steps_per_render = max(10 // fps, 1)
        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PusherEnv(push_env_cfg),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env = env_fn()

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')

        self.env = env
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
    
    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # start rollout
        obs, info = env.reset()
        past_action = None
        policy.reset()

        pbar = tqdm.tqdm(total=self.max_steps, desc="Eval PusherImageRunner", 
            leave=False, mininterval=self.tqdm_interval_sec)
        done = False
        all_videos = []
        all_rewards = []
        while not done:
            # create obs dict
            np_obs_dict = dict(obs)
            if self.past_action and (past_action is not None):
                # TODO: not tested
                np_obs_dict['past_action'] = past_action[
                    :,-(self.n_obs_steps-1):].astype(np.float32)
            
            # device transfer
            obs_dict = dict_apply(np_obs_dict, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(
                    device=device))

            # run policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            # device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.squeeze().detach().to('cpu').numpy())

            action = np_action_dict['action']
            # step env
            obs, reward, terminated, truncated, info = env.step(action)
            done = np.all(terminated)
            past_action = action
            
            # update pbar
            pbar.update(action.shape[1])
            for o in env.obs:
                all_videos.append(o["image"] * 255)
            all_rewards.append(reward)
        pbar.close()

        # clear out video buffer
        _ = env.reset()

        # log
        max_rewards = collections.defaultdict(list)
        end_rewards = collections.defaultdict(list)
        log_data = dict()

        prefix = "test_"
        max_reward = np.max(all_rewards)
        max_rewards[prefix].append(max_reward)
        log_data[prefix+'sim_max_reward'] = max_reward
        end_reward = all_rewards[-1]
        end_rewards[prefix].append(end_reward)
        log_data[prefix+'sim_end_reward'] = end_reward

        # visualize sim
        if all_videos is not None:
            sim_video = wandb.Video(np.array(all_videos))
            log_data[prefix+'sim_video'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

    def playback(self, actions: np.ndarray):
        env = self.env

        # start rollout
        obs, info = env.reset()

        print("Playing back pusher training actions")
        all_videos = []
        all_rewards = []
        # step env
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # update pbar
        for o in env.obs:
            all_videos.append(o["image"] * 255)
        all_rewards.append(reward)

        # clear out video buffer
        _ = env.reset()

        # log
        max_rewards = collections.defaultdict(list)
        end_rewards = collections.defaultdict(list)
        log_data = dict()

        prefix = "playback_"
        max_reward = np.max(all_rewards)
        max_rewards[prefix].append(max_reward)
        log_data[prefix+'sim_max_reward'] = max_reward
        end_reward = all_rewards[-1]
        end_rewards[prefix].append(end_reward)
        log_data[prefix+'sim_end_reward'] = end_reward

        # visualize sim
        if all_videos is not None:
            sim_video = wandb.Video(np.array(all_videos))
            log_data[prefix+'sim_video'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
