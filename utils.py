import sys,os
import warnings
import json
warnings.filterwarnings("ignore")

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

blockPrint()
import torch
import numpy as np
import multiprocess as mp

import h5py
import plotly.graph_objects as go

from copy import deepcopy
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robosuite.utils.camera_utils import CameraMover
from scipy.spatial.transform import Rotation as R

from tqdm import tqdm
import matplotlib.pyplot as plt

from robomimic.scripts.train import train
from robomimic.config import config_factory

sys.path.append(os.path.join("./", 'robomimic', 'robomimic', 'scripts'))
from dataset_states_to_obs import dataset_states_to_obs

enablePrint()

class MyCameraMover(CameraMover):
    def __init__(self, env, camera_name):
        super().__init__(env, camera_name)
    
    def rotate_camera_around_point(self, point, axis, angle_deg):
        camera_pos = np.array(self.env.sim.data.get_mocap_pos(self.mover_body_name))
        camera_quat = self.env.sim.data.get_mocap_quat(self.mover_body_name)
        # camera_quat from mujoco is in [w, x, y, z] format
        camera_quat = [camera_quat[1], camera_quat[2], camera_quat[3], camera_quat[0]]
        new_pos, new_quat = self.rotate_vector(camera_pos, camera_quat, point, axis, angle_deg)
        new_quat = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]
        self.set_camera_pose(pos=new_pos, quat=new_quat)
        return new_pos
    

    def set_camera_pose(self, pos=None, quat=None):
        """
        Sets the camera pose, which optionally includes position and / or quaternion

        Args:
            pos (None or 3-array): If specified, should be the (x,y,z) global cartesian pos to set camera to
            quat (None or 4-array): If specified, should be the (w,x,y,z) global quaternion orientation to set camera to
        """
        if pos is not None:
            self.env.sim.data.set_mocap_pos(self.mover_body_name, pos)
        if quat is not None:
            self.env.sim.data.set_mocap_quat(self.mover_body_name, quat)

        # Make sure changes propagate in sim
        self.env.sim.forward()


    def rotate_vector(self, camera_pos, camera_quat, point, axis, angle_deg):
        """
        Rotate the camera around a given point and axis.

        Args:
            camera_pos (np.array): Camera's position as [x, y, z].
            camera_quat (np.array): Camera's orientation as [x, y, z, w].
            point (np.array): The point to rotate around as [x, y, z].
            axis (np.array): The axis of rotation as [x, y, z].
            angle_deg (float): Rotation angle in degrees.

        Returns:
            new_pos (np.array): New camera position as [x, y, z].
            new_quat (np.array): New camera quaternion as [x, y, z, w].
        """
        # Convert angle to radians
        angle_rad = np.radians(angle_deg)

        # Normalize the axis of rotation
        axis = axis / np.linalg.norm(axis)

        # Translate the camera position relative to the rotation point
        translated_pos = camera_pos - point

        # Create the rotation matrix
        rotation = R.from_rotvec(angle_rad * axis)
        # print('rotation:', rotation.as_rotvec())
        # input()

        # Rotate the camera position
        rotated_pos = rotation.apply(translated_pos)

        # Translate back to the original position
        new_pos = rotated_pos + point

        # Rotate the camera's orientation (quaternion)
        camera_rotation = R.from_quat(camera_quat)
        new_rotation = rotation * camera_rotation
        new_quat = new_rotation.as_quat()
        # print('camera quat before and after:', camera_quat, new_quat)
        # print('camera pos before and after:', camera_pos, new_pos)

        return new_pos, new_quat

def get_succes_rate_for_given_camera_angle_semi_sphere(angle,n_rollouts,agent,horizon=200,render=False): 
    env_name=None
    video_path=None
    camera_names=['agentview']

    success_rate_per_rollout = []
    
    for _ in tqdm(range(n_rollouts), desc="Processing rollouts"):
    # for _ in range(n_rollouts):
        blockPrint()
        ckpt_path = agent

        # device
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)

        # restore policy
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=False)

        # read rollout settings
        rollout_horizon = horizon
        if rollout_horizon is None:
            # read horizon from config
            config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
            rollout_horizon = config.experiment.rollout.horizon

        # create environment from saved checkpoint
        env1, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt_dict, 
            env_name=env_name, 
            render=render, 
            render_offscreen=(video_path is not None), 
            verbose=False,
        )


        policy.start_episode()

        state_dict = env1.get_state()
        obs = env1.reset_to(state_dict)

        try:
            camera_mover = MyCameraMover(env1.env.env, 'agentview')
        except:
            camera_mover = MyCameraMover(env1.env, 'agentview')
        camera_mover.rotate_camera_around_point((0, 0, 0.85), axis=(0, 1, 0), angle_deg=angle[1])
        camera_mover.rotate_camera_around_point((0, 0, 1.35), axis=(0, 0, 1), angle_deg=angle[0])

        total_reward = 0.

        for step_i in range(rollout_horizon):

            # get action from policy
            act = policy(ob=obs)
            
            # play action
            next_obs, r, done, _ = env1.step(act)

            # compute reward
            total_reward += r
            success = env1.is_success()["task"]

            # visualization
            if render:
                env1.render(mode="human", camera_name=camera_names[0])

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            st = env1.get_state()

                

        stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))
        
        if render:
            try:
                env1.env.env.close_renderer()
            except:
                env1.env.close_renderer()

        success_rate_per_rollout.append(stats['Success_Rate'])

    succes_rate_for_given_angle = np.mean(success_rate_per_rollout)
    enablePrint()
    return succes_rate_for_given_angle

def get_success_rate_in_parallel(angle_list, n_rollouts, agent,save_path, horizon,render=False, num_workers=2):
    # Create a context with the 'spawn' start method.
    blockPrint()
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_workers) as pool:
        # Note: Ensure that get_succes_rate_for_given_camera_angle_semi_sphere is importable.
        results = pool.starmap(
            get_succes_rate_for_given_camera_angle_semi_sphere,
            [(angle, n_rollouts, agent,horizon, render) for angle in angle_list]
        )
    enablePrint()
    model_name = agent.split('/')[-3]
    np.save(f"{save_path}/{model_name}.npy",np.array(results))
    return torch.tensor(results,dtype=torch.float32).unsqueeze(-1)

def overall_dynamic_succes_rate(n_rollouts,agent,horizon=200,render=False):
    sr_array = np.zeros(6)
    for i in range(6):
        sr_array[i] = get_succes_rate_for_given_camera_angle_dynamic_camera(0, i, n_rollouts, agent, horizon, render)
    return sr_array

def get_succes_rate_for_given_camera_angle_dynamic_camera(angle,path,n_rollouts,agent,horizon=200,render=False): 
    env_name=None
    video_path=None
    camera_names=['agentview']

    success_rate_per_rollout = []
    
    for _ in tqdm(range(n_rollouts), desc="Processing rollouts"):
    # for _ in range(n_rollouts):
        # print(device)
        blockPrint()
        ckpt_path = agent

        # device
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        # device = "cuda:1"

        # restore policy
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=False)

        # read rollout settings
        rollout_horizon = horizon
        if rollout_horizon is None:
            # read horizon from config
            config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
            rollout_horizon = config.experiment.rollout.horizon

        # create environment from saved checkpoint
        env1, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt_dict, 
            env_name=env_name, 
            render=render, 
            render_offscreen=(video_path is not None), 
            verbose=False,
        )


        policy.start_episode()

        state_dict = env1.get_state()
        obs = env1.reset_to(state_dict)

        try:
            camera_mover = MyCameraMover(env1.env.env, 'agentview')
        except:
            camera_mover = MyCameraMover(env1.env, 'agentview')
        # camera_mover.rotate_camera_around_point((0, 0, 1.35), axis=(0, 0, 1), angle_deg=angle[0])

        total_reward = 0.

        for step_i in range(rollout_horizon):

            if path == 0:
                camera_mover.rotate_camera_around_point((0, 0, 0.85), axis=(0, 1, 0), angle_deg=0.1)
                # camera_mover.rotate_camera_around_point((0, 0, 0.85), axis=(0, 0, 1), angle_deg=0.2)
            if path == 1:
                # camera_mover.rotate_camera_around_point((0, 0, 0.85), axis=(0, 1, 0), angle_deg=0.1)
                camera_mover.rotate_camera_around_point((0, 0, 0.85), axis=(0, 0, 1), angle_deg=0.2)
            if path == 2:
                camera_mover.rotate_camera_around_point((0, 0, 0.85), axis=(0, 1, 0), angle_deg=0.1)
                camera_mover.rotate_camera_around_point((0, 0, 0.85), axis=(0, 0, 1), angle_deg=0.2)
            if path == 3:
                camera_mover.rotate_camera_around_point((0, 0, 0.85), axis=(0, 1, 0), angle_deg=-0.1)
                camera_mover.rotate_camera_around_point((0, 0, 0.85), axis=(0, 0, 1), angle_deg=0.2)
            if path == 4:
                camera_mover.rotate_camera_around_point((0, 0, 0.85), axis=(0, 1, 0), angle_deg=0.1)
                camera_mover.rotate_camera_around_point((0, 0, 0.85), axis=(0, 0, 1), angle_deg=-0.2)
            if path == 5:
                camera_mover.rotate_camera_around_point((0, 0, 0.85), axis=(0, 1, 0), angle_deg=-0.1)
                camera_mover.rotate_camera_around_point((0, 0, 0.85), axis=(0, 0, 1), angle_deg=-0.2)


            # get action from policy
            act = policy(ob=obs)
            
            # play action
            next_obs, r, done, _ = env1.step(act)

            # compute reward
            total_reward += r
            success = env1.is_success()["task"]

            # visualization
            if render:
                env1.render(mode="human", camera_name=camera_names[0])

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            st = env1.get_state()

                

        stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))
        
        if render:
            try: 
                env1.env.env.close_renderer()
            except:
                env1.env.close_renderer()


        success_rate_per_rollout.append(stats['Success_Rate'])

    succes_rate_for_given_angle = np.mean(success_rate_per_rollout)
    enablePrint()
    return succes_rate_for_given_angle

def merge_groups(group1, group2, output_group):
    """Merge two HDF5 groups into an output group."""
    # Copy attributes of group1 to the output group
    for attr in group1.attrs:
        output_group.attrs[attr] = group1.attrs[attr]
    
    # Get all keys from both groups
    keys1 = list(group1.keys())
    keys2 = list(group2.keys())
    
    for key in keys1:
        # Copy keys from group1 as is
        group1.copy(key, output_group)
    
    for key in keys2:
        # Prefix keys from group2 with 'a_'
        group2.copy(key, output_group, name=f'{key.split("_")[0]}_{int(key.split("_")[-1])+200}')

def combine_hdf5(file1, file2, output_file):
    # Open the input HDF5 files
    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        with h5py.File(output_file, 'w') as out_f:
            # Create the 'mask' group and double its dataset values
            mask_group = out_f.create_group('mask')
            for key in f1['mask'].keys():
                data = f1['mask'][key][:]  # Read dataset
                mask_group.create_dataset(key, data=np.tile(data,2))  # Double values
            
            # Create the 'data' group in the output file
            output_data_group = out_f.create_group('data')
            merge_groups(f1['data'], f2['data'], output_data_group)


def generate_dataset_for_angle(angle, raw_dataset, dataset_path, default_dataset):
    args_generate_data = type('', (), {})()  # Create a simple object to hold attributes as variables
    args_generate_data.dataset = raw_dataset
    args_generate_data.camera_names = ['agentview']
    args_generate_data.camera_height = 84
    args_generate_data.camera_width = 84
    args_generate_data.done_mode = 2

    #TODO check what these do
    args_generate_data.depth = False
    args_generate_data.n = None
    args_generate_data.shaped = False
    args_generate_data.copy_rewards = False
    args_generate_data.copy_dones = False
    args_generate_data.exclude_next_obs = False
    args_generate_data.compress = False
    """ Process a single angle: generate data and combine datasets. """
    args_generate_data.output_name = f"training_angle_{int(angle[0])}and{int(angle[1])}.hdf5"
    args_generate_data.angle = angle
    default_dataset_name =  default_dataset.split('/')[-1][:-5]
    training_path = f'{dataset_path}/training_angle_{int(angle[0])}and{int(angle[1])}.hdf5'
    combined_path = f'{dataset_path}/{default_dataset_name}_{int(angle[0])}and{int(angle[1])}.hdf5'

    if not os.path.exists(training_path):
        dataset_states_to_obs(args_generate_data)

    if not os.path.exists(combined_path):
        combine_hdf5(default_dataset, training_path, combined_path)
    os.remove(training_path) 
    return combined_path  # Return the new dataset path

def parallel_generate_dataset_for_angles(angle_list, raw_dataset, dataset_path, default_dataset,num_workers=2):
    """ Parallel execution of dataset processing per angle. """
    mp.set_start_method('spawn', force=True)  # Ensure CUDA compatibility
    with mp.Pool(processes=num_workers) as pool:
        new_datasets = pool.starmap(
            generate_dataset_for_angle, [(angle, raw_dataset, dataset_path, default_dataset) for angle in angle_list]
        )

    return new_datasets

def train_model_for_dataset(dataset, alg, batch_size, num_epochs, device, model_path,base_model_path):
    # config = config_factory(algo_name=alg)
    ext_cfg = json.load(open(alg, 'r'))
    config = config_factory(ext_cfg["algo_name"])
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base algo config
    with config.values_unlocked():
        config.update(ext_cfg)
    config.train.batch_size = batch_size
    config.train.num_epochs = num_epochs
    dataset_name =  dataset.split('/')[-1][:-5]
    config.train.starting_point = base_model_path
    # angle =  dataset_name.split('_')[-1]
    if not os.path.exists(f'{model_path}'):
        os.makedirs(model_path)
    if not os.path.exists(f'{model_path}/{dataset_name}/models/model_epoch_{num_epochs}.pth'): 
        config.experiment.name = dataset_name
        config.train.data = dataset
        config.train.output_dir = model_path

        # launch training run
        # blockPrint()
        train(config, device=device)
        # enablePrint()
    return f'{model_path}/{dataset_name}/models/model_epoch_{num_epochs}.pth'

def parallel_train_agents(datasets, alg, batch_size, num_epochs, device, model_path,base_model_path, num_workers=2):
    """ Train models in parallel for multiple datasets. """
    

    mp.set_start_method('spawn', force=True)  # Ensure CUDA compatibility
    with mp.Pool(processes=num_workers) as pool:
        agents = pool.starmap(
            train_model_for_dataset, [(dataset, alg, batch_size, num_epochs, device, model_path,base_model_path) for dataset in datasets]
        )

    return agents

def plot_3d(success_rate_flattened,positions,title):
    positions = np.array(positions)
    # success_rate_flattened = sr.numpy().flatten()

    # ----- Create Interactive 3D Scatter Plot with Plotly -----
    fig = go.Figure(data=[go.Scatter3d(
        x=positions[:, 0],         # x-coordinates
        y=positions[:, 1],         # y-coordinates
        z=positions[:, 2],         # z-coordinates
        mode='markers',
        marker=dict(
            size=10,
            color=success_rate_flattened,              # Set color based on success rate values
            colorscale='Viridis',  # Choose a colorscale
            colorbar=dict(title=title),
            line=dict(width=1),
            opacity=0.8
        )
    )])

    # Update layout to add axis titles and a plot title
    fig.update_layout(
        title='Interactive 3D Scatter Plot of Initial Positions with Success Rate Coloring',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=1200,   # Adjust the width (in pixels) as desired
        height=800    # Adjust the height (in pixels) as desired
    )

    # Display the interactive plot
    fig.show()

