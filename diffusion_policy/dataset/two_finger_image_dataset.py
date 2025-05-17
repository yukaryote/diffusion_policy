from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)

class TwoFingerImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            shape_meta,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            state_type='qpos'
            ):
        
        super().__init__()
        self.state_type = state_type
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                if key == "agent_pos":
                    lowdim_keys.append("state")

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('pos') or key.endswith("state"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            if key == "state":
                normalizer["agent_pos"] = this_normalizer
            else:
                normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer
        # if self.state_type != "stateless":
        #     data = {
        #         'action': self.replay_buffer['action'],
        #         'agent_pos': self.replay_buffer['state']
        #     }
        # else:
        #     data = {
        #         'action': self.replay_buffer['action'],
        #     }
        # normalizer = LinearNormalizer()
        # normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # normalizer['image'] = get_image_range_normalizer()
        # return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'].astype(np.float32) # (agent_posx2, block_posex3)
        image = np.moveaxis(sample['img'],-1,1)/255

        if self.state_type != 'stateless':
            data = {
                'obs': {
                    'image': image, # T, 3, 256, 256
                    'agent_pos': agent_pos, # T, state_dim
                },
                'action': sample['action'].astype(np.float32) # T, action_dim
            }
        else:
            data = {
                'obs': {
                    'image': image, # T, 3, 256, 256
                },
                'action': sample['action'].astype(np.float32) # T, action_dim
            }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )


def test():
    import os
    zarr_path = os.path.expanduser('~/scene-jacobian-discovery/diff-policy/diffusion_policy/data/two_finger/shadow_finger_box_1traj_pos0_rotz0.zarr')
    shape_meta = {
        "action": 
        {
            "shape": [4]
        },
        "obs": 
        {
            "agent_pos":
            {
                "shape": [12],
                "type": "low_dim"
            },
            "image":
            {
                "shape": [3, 256, 256],
                "type": "rgb"
            }
        }

    }
    dataset = TwoFingerImageDataset(zarr_path, shape_meta=shape_meta, horizon=16)

    # from matplotlib import pyplot as plt
    normalizer = dataset.get_normalizer()
    nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    #diff = np.diff(nactions, axis=0)
    #dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)


if __name__ == "__main__":
    test()
