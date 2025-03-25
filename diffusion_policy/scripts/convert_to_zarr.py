if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import click
import pathlib
from pathlib import Path
import numpy as np
import io
import shutil
import zarr
import torch
import pickle
import gzip
import cv2 as cv
from tqdm import tqdm

from diffusion_policy.common.replay_buffer import ReplayBuffer


# load everything onto cpu
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        # elif module == "jax.interpreters.xla" and name == "DeviceArray":
        #     return lambda b: jax.device_put(io.BytesIO(b), jax.devices("cpu")[0])
        else:
            return super().find_class(module, name)
     
        
def load_gzip_file(file_name):
    with gzip.open(file_name, "rb") as f:
        traj = CPU_Unpickler(f).load()
    return traj


def move(destination, depth=None):
    if not depth:
        depth = []
    for file_or_dir in os.listdir(os.path.join([destination] + depth, os.sep)):
        if os.path.isfile(file_or_dir):
            shutil.move(file_or_dir, destination)
        else:
            move(destination, os.path.join(depth + [file_or_dir], os.sep))


def convert_to_replay_buffer_zarr(dataset_root, zarr_store_path, state_type='qpos', remove_existing=False):
    dataset_root = Path(dataset_root)
    # remove existing zarr store if it exists, remove everything in the store
    if remove_existing and os.path.exists(zarr_store_path):
        print(f"Removing existing Zarr store at {zarr_store_path}")
        shutil.rmtree(zarr_store_path)
    zarr_store = zarr.open(zarr_store_path, mode='w')

    # Initialize lists to hold aggregated data
    all_actions = []
    all_images = []
    all_states = []
    episode_ends = []
    current_index = 0

    for traj_dir in tqdm(sorted(dataset_root.iterdir())):
        if not traj_dir.is_dir():
            continue  # Skip non-directory files

        # Load images
        img_files = sorted(traj_dir.glob("*.jpg"))
        images = np.stack([cv.imread(img_file) for img_file in img_files])
        all_images.append(images)

        # Load states
        traj_data = load_gzip_file(traj_dir / "traj_data.pkl")
        states = np.array(traj_data["states"], dtype='f4')
        if state_type == 'qpos':
            states = states[:, :12]
        elif state_type == 'qvel':
            states = states[:, :18]
        all_states.append(states)

        # Load actions
        actions = np.array(traj_data["actions"], dtype='f4')
        all_actions.append(actions)

        # Update episode end index
        current_index += len(actions)
        # print("images shape:", images.shape)
        # print("states shape:", states.shape)
        # print("actions shape:", actions.shape)
        # print("Current index:", current_index)
        episode_ends.append(current_index)

    # Concatenate all data along the time dimension
    all_actions = np.concatenate(all_actions, axis=0)
    all_images = np.concatenate(all_images, axis=0)
    all_states = np.concatenate(all_states, axis=0)
    episode_ends = np.array(episode_ends, dtype='int64')

    # Create datasets in the Zarr store
    data_group = zarr_store.create_group('data')
    meta_group = zarr_store.create_group('meta')

    actions_arr = data_group.create_array("action", shape=all_actions.shape, chunks=(1, all_actions.shape[1]), dtype='f4')
    img_arr = data_group.create_array("img", shape=all_images.shape, chunks=(1, *all_images.shape[1:]), dtype='f4')
    states_arr = data_group.create_array("state", shape=all_states.shape, chunks=(1, all_states.shape[1]), dtype='f4')
    ends_arr = meta_group.create_array("episode_ends", shape=episode_ends.shape, chunks=(1,), dtype='int64')

    actions_arr[:] = all_actions
    img_arr[:] = all_images
    states_arr[:] = all_states
    ends_arr[:] = episode_ends

    # move all data out of the 'c' directory and into each 'action', 'img', 'state' directory
    for key in data_group.keys():
        dst_path = zarr_store_path + '/' + key
        move(dst_path)
    print(f"Dataset converted to ReplayBuffer Zarr format at {zarr_store_path}")


@click.command()
@click.option('-i', '--input', default="/data/scene-rep/u/sizheli/03-02-2025", help='input dir contains npy files')
@click.option('-o', '--output', default="/data/scene-rep/u/iyu/scene-jacobian-discovery/diff-policy/diffusion_policy/data/two_finger/shadow_finger_box_qpos_1traj.zarr", help='output zarr path')
@click.option('--state_type', default='qpos', help='state type to use for replay buffer')
@click.option('--num_traj', default=-1, help='number of trajectories to convert, -1 for all')
def main(input, output, state_type, num_traj):
    data_directory = pathlib.Path(input)
    # if input already exists, remove it
    if os.path.exists(output):
        print(f"Removing existing Zarr store at {output}")
        shutil.rmtree(output)

    buffer = ReplayBuffer.create_empty_numpy()
    traj_dirs = sorted(data_directory.iterdir())[:num_traj]
    print("Number of trajectories to convert:", len(traj_dirs))
    for traj_dir in tqdm(traj_dirs):
        # each traj_dir is a directory containing images and states and actions
        if not traj_dir.is_dir():
            continue  # Skip non-directory files

        # Load images
        img_files = sorted(traj_dir.glob("*.jpg"))
        images = np.stack([cv.cvtColor(cv.imread(str(img_file)), cv.COLOR_BGR2RGB) for img_file in img_files]).astype(np.float32)

        # Load states
        traj_data = load_gzip_file(traj_dir / "traj_data.pkl")
        states = np.array(traj_data["states"], dtype='f4')
        if state_type == 'qpos':
            states = states[:, :12]
        elif state_type == 'qvel':
            states = states[:, :18]

        # Load actions
        actions = np.array(traj_data["actions"], dtype='f4')
        data = {
            "img": images,
            "state": states,
            "action": actions
        }
        buffer.add_episode(data)
    buffer.save_to_path(zarr_path=output, chunk_length=-1)


if __name__ == '__main__':
    main()
    # Example usage
    # convert_to_replay_buffer_zarr("/data/scene-rep/u/sizheli/03-02-2025", 
    #                             "/data/scene-rep/u/iyu/diffusion_policy/data/two_finger/shadow_finger_box_qpos.zarr",
    #                             state_type='qpos',
    #                             remove_existing=True)

