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
import multiprocessing as mp


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


def process_trajectory(args):
    """Function to load and process a single trajectory."""
    traj_dir, state_type = args
    try:
        traj_data = load_gzip_file(os.path.join(traj_dir, "traj_data.pkl"))
        images = np.stack([
            cv.cvtColor(cv.imread(str(img_file)), cv.COLOR_BGR2RGB)
            for img_file in sorted(traj_dir.glob("*.jpg"))
        ])
        states = np.array(traj_data["states"], dtype='f4')

        if state_type == 'qpos':
            states = states[:, :12]
        elif state_type == 'qvel':
            states = states[:, :18]

        actions = np.array(traj_data["actions"], dtype='f4')

        return {
            "img": images,
            "state": states,
            "action": actions
        }
    except Exception as e:
        print(f"Error processing {traj_dir}: {e}")
        return None


@click.command()
@click.option('-i', '--input', default="/data/scene-rep/u/iyu/data/ShadowFinger/lester/03-31-2025", help='input dir contains npy files')
@click.option('-o', '--output', default="/data/scene-rep/u/iyu/scene-jacobian-discovery/diff-policy/diffusion_policy/data/two_finger/shadow_finger_box_qvel.zarr", help='output zarr path')
@click.option('--state_type', default='qvel', help='state type to use for replay buffer')
@click.option('--num_traj', default=-1, help='number of trajectories to convert, -1 for all')
@click.option('--num_workers', default=8, help='number of parallel workers')
def main(input, output, state_type, num_traj, num_workers):
    data_directory = pathlib.Path(input)
    
    # If output already exists, remove it
    if os.path.exists(output):
        print(f"Removing existing Zarr store at {output}")
        shutil.rmtree(output)

    buffer = ReplayBuffer.create_empty_numpy()
    # only get directories, not files
    traj_dirs = sorted([d for d in data_directory.iterdir() if d.is_dir()])[:num_traj]
    print("Number of trajectories to convert:", len(traj_dirs))

    # Use multiprocessing Pool
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(process_trajectory, [(traj_dir, state_type) for traj_dir in traj_dirs]), total=len(traj_dirs)))

    # Add valid episodes to the replay buffer
    for data in results:
        if data is not None:
            buffer.add_episode(data)

    buffer.save_to_path(zarr_path=output, chunk_length=-1)

if __name__ == '__main__':
    main()
    # Example usage
    # convert_to_replay_buffer_zarr("/data/scene-rep/u/sizheli/03-02-2025", 
    #                             "/data/scene-rep/u/iyu/diffusion_policy/data/two_finger/shadow_finger_box_qpos.zarr",
    #                             state_type='qpos',
    #                             remove_existing=True)

