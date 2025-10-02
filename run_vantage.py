import torch
import sys,os
import warnings
import numpy as np
import time
import argparse

from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.acquisition import qUpperConfidenceBound
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, PeriodicKernel


from tqdm import tqdm

import utils as ut

warnings.filterwarnings("ignore")

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class UncertainRBFKernel(RBFKernel):
  """
  RBF kernel convolved with Gaussian input noise.
  Supports isotropic noise with variance sigma^2.
  """
  def __init__(self, sigma=0.1, ard_num_dims=None, **kwargs):
      super().__init__(ard_num_dims=ard_num_dims, **kwargs)
      self.sigma = sigma  # fixed noise std, can also make this learnable

  def forward(self, x1, x2, diag=False, **params):
      # effective lengthscale^2 = ell^2 + 2*sigma^2
      eff_lengthscale2 = self.lengthscale.pow(2) + 2 * self.sigma**2

      if diag:
          diff = (x1 - x2)
          sq_dist = diff.pow(2).sum(dim=-1)
      else:
          diff = (x1.unsqueeze(-2) - x2.unsqueeze(-3))
          sq_dist = diff.pow(2).sum(dim=-1)

      cov = torch.exp(-0.5 * sq_dist / eff_lengthscale2)

      # prefactor shrinks amplitude
      d = x1.shape[-1]
      prefactor = (self.lengthscale.pow(2) / eff_lengthscale2).pow(d/2)
      cov = prefactor * cov

      return cov

class CustomGP(SingleTaskGP):
    def __init__(self, train_X, train_Y, kernel="rbf", fixed_lengthscale=None):
        mean_module = ConstantMean()

        if kernel == "my_uncertain_rbf":
            covar_module = ScaleKernel(UncertainRBFKernel(sigma=0.1))
        elif kernel == "rbf":
            covar_module = ScaleKernel(RBFKernel())
        elif kernel == "matern":
            covar_module = ScaleKernel(MaternKernel(nu=2.5))
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        super().__init__(train_X, train_Y,
                         covar_module=covar_module,
                         mean_module=mean_module)

def get_next_ips_ucb_custom(angles_tensor, success_rates_tensor, bounds, n_points):
    single_task_model = SingleTaskGP(angles_tensor,success_rates_tensor)
    # single_task_model = CustomGP(angles_tensor,success_rates_tensor, kernel="my_uncertain_rbf", fixed_lengthscale=None) #used for experimental setup error
    mll = ExactMarginalLogLikelihood(single_task_model.likelihood,single_task_model)
    fit_gpytorch_mll(mll)
    UCB = qUpperConfidenceBound(model = single_task_model, beta=0.3) #beta = low means exploit otherwise explore
    candidates, _ = optimize_acqf(acq_function=UCB,bounds=bounds,q=n_points,num_restarts=200,raw_samples=512,options={"batch_limit":5,"maxiter":200})

    return candidates

def get_next_ips_ucb(angles_tensor, success_rates_tensor, bounds, n_points):
    single_task_model = SingleTaskGP(angles_tensor,success_rates_tensor)
    mll = ExactMarginalLogLikelihood(single_task_model.likelihood,single_task_model)
    fit_gpytorch_mll(mll)
    UCB = qUpperConfidenceBound(model = single_task_model, beta=0.3) #beta = low means exploit otherwise explore
    candidates, _ = optimize_acqf(acq_function=UCB,bounds=bounds,q=n_points,num_restarts=200,raw_samples=512,options={"batch_limit":5,"maxiter":200})

    return candidates

def normalized_bo_op_to_camera_angles_90_90_horizontal_45_45_vertical(bo_op_list):
    angle_list = []
    for val in bo_op_list:
        angle_h = (val[0]-0.5)*2*90 #convert 0-1 to -1 to 1, then -90 to 90
        angle_v = (val[1]-0.5)*2*45#convert 0-1 to -1 to 1, then -45 to 45
        angle_list.append([angle_h,angle_v])

    return angle_list

def normalized_bo_op_to_camera_angles_90_90_horizontal_45_45_vertical_uncertain(bo_op_list):
    angle_list = []
    for val in bo_op_list:
        r_h = (np.random.random() - 0.5)*2*2.29
        r_v = (np.random.random() - 0.5)*2*2.29
        angle_h = (val[0]-0.5)*2*90 + r_h#convert 0-1 to -1 to 1, then -90 to 90
        angle_v = (val[1]-0.5)*2*45 + r_v#convert 0-1 to -1 to 1, then -45 to 45
        angle_list.append([angle_h,angle_v])

    return angle_list


def generate_initial_data(algo, batch_size, num_epochs, device, model_path,base_model_path,test_angle_list,raw_dataset, dataset_path, default_dataset, horizon=200,initial_points = 4, n_rollouts=10,uncertain_camera=False):
    new_inputs = torch.rand(initial_points,2)
    if uncertain_camera:
        angle_list = normalized_bo_op_to_camera_angles_90_90_horizontal_45_45_vertical_uncertain(new_inputs)
    else:
        angle_list = normalized_bo_op_to_camera_angles_90_90_horizontal_45_45_vertical(new_inputs)

    # Create a dictionary to keep track of unique angles and their corresponding inputs
    unique_data = {}
    for angle, value in zip(angle_list, new_inputs):
        angle_key = tuple(angle)  # Convert the list to a tuple
        if angle_key not in unique_data:
            unique_data[angle_key] = value


    # Extract the filtered angle_list and new_inputs
    angle_list = list(unique_data.keys())
    new_inputs = torch.stack(list(unique_data.values()))
    datasets =  ut.parallel_generate_dataset_for_angles(np.round(np.array(angle_list)),raw_dataset, dataset_path, default_dataset,1)
    agents = ut.parallel_train_agents(datasets, algo, batch_size, num_epochs, device, model_path,base_model_path, num_workers=1)

    results_overall = []
    results_array = []
    if not os.path.exists(f'{model_path}/save_arrays'):
        os.makedirs(f'{model_path}/save_arrays')
    for agent in agents:
        results_array.append(np.array(ut.get_success_rate_in_parallel(test_angle_list, n_rollouts, agent,f'{model_path}/save_arrays', horizon,render=False, num_workers=1)))
        results_overall.append(np.mean(results_array[-1]))

    return new_inputs,torch.tensor(results_overall,dtype = torch.float32).unsqueeze(-1)

def bo_loop( algo, batch_size, num_epochs, device, model_path,base_model_path, raw_dataset, dataset_path, default_dataset,n_runs = 8, horizon=200, initial_points = 4, n_rollouts = 10,uncertain_camera=False):
    #All anlges to test inside Theta
    test_angle_list = [[-90,-40],[0,-45],[90,-40],
                        [-90,-30],[-45,-30],[0,-30],[45,-30],[90,-30],
                        [-90,-15],[-60,-15],[-30,-15],[0,-15],[30,-15],[60,-15],[90,-15],
                        [-90,0],[-75,0],[-60,0],[-45,0],[-30,0],[-15,0],[0,0],[15,0],[30,0],[45,0],[60,0],[75,0],[90,0],
                        [-90,15],[-75,15],[-60,15],[-45,15],[-30,15],[-15,15],[0,15],[15,15],[30,15],[45,15],[60,15],[75,15],[90,15],
                        [-90,30],[-75,30],[-60,30],[-45,30],[-30,30],[-15,30],[0,30],[15,30],[30,30],[45,30],[60,30],[75,30],[90,30],
                        [-90,45],[-75,45],[-60,45],[-45,45],[-30,45],[-15,45],[0,45],[15,45],[30,45],[45,45],[60,45],[75,45],[90,45],
                        ]
    # blockPrint()
    normalized_angles_tensor, success_rates_tensor= generate_initial_data(algo, batch_size, num_epochs, device, model_path,base_model_path,test_angle_list,raw_dataset, dataset_path, default_dataset,horizon,initial_points, n_rollouts=10,uncertain_camera=uncertain_camera)
    # enablePrint()
    print("initial_data_done")

    bounds = torch.tensor([[0.,0.], [1.,1.]])

    results_array = []

    for i in tqdm(range(n_runs)):
        if uncertain_camera:
            new_inputs = get_next_ips_ucb_custom(normalized_angles_tensor, success_rates_tensor, bounds, n_points=1) #for exp setup error experiment
        else:
            new_inputs = get_next_ips_ucb(normalized_angles_tensor, success_rates_tensor, bounds, n_points=1)
        angle_list = normalized_bo_op_to_camera_angles_90_90_horizontal_45_45_vertical(new_inputs)

        # Create a dictionary to keep track of unique angles and their corresponding inputs
        unique_data = {}
        for angle, value in zip(angle_list, new_inputs):
            angle_key = tuple(angle)  # Convert the list to a tuple
            if angle_key not in unique_data:
                unique_data[angle_key] = value

        # Extract the filtered angle_list and new_inputs
        angle_list = list(unique_data.keys())
        new_inputs = torch.stack(list(unique_data.values()))

        datasets =  ut.parallel_generate_dataset_for_angles(np.round(np.array(angle_list)),raw_dataset, dataset_path, default_dataset,1)

        agents = ut.parallel_train_agents(datasets, algo, batch_size, num_epochs, device, model_path,base_model_path, num_workers=1)

        results_overall = []
        if not os.path.exists(f'{model_path}/save_arrays'):
            os.makedirs(f'{model_path}/save_arrays')
        for agent in agents:
            results_array.append(np.array(ut. get_success_rate_in_parallel(test_angle_list, n_rollouts, agent,f'{model_path}/save_arrays', horizon,render=False, num_workers=1)))
            results_overall.append(np.mean(results_array[-1]))

        normalized_angles_tensor = torch.cat([normalized_angles_tensor,new_inputs]) 
        success_rates_tensor       = torch.cat([success_rates_tensor,torch.tensor(results_overall,dtype = torch.float32).unsqueeze(-1)])

        np.save(f'{model_path}/full_array.npy',np.array(results_array))
        np.save(f'{model_path}/normalized_angles.npy', np.array(normalized_angles_tensor))
        np.save(f'{model_path}/succes_rates.npy', np.array(success_rates_tensor))

    return  normalized_angles_tensor,success_rates_tensor


def main():
    parser = argparse.ArgumentParser(
        description="Run bo_loop with all inputs provided as command-line arguments."
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        help='Mode (e.g., "bc")'
    )
    parser.add_argument(
        "--batch_size", type=int, required=True,
        help="Second parameter (e.g., 256)"
    )
    parser.add_argument(
        "--num_epochs", type=int, required=True,
        help="Third parameter (e.g., 50)"
    )
    parser.add_argument(
        "--device", type=str, default='cuda:0',
        help='Device to use (default: "cuda:0")'
    )
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Path to the model directory"
    )
    parser.add_argument(
        "--base_model", type=str, required=True,
        help="Path to the model file (e.g., model_epoch_1000.pth)"
    )
    parser.add_argument(
        "--raw_data", type=str, required=True,
        help="Path to the demo HDF5 file"
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True,
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--base_dataset", type=str, required=True,
        help="Path to the angle HDF5 file"
    )
    parser.add_argument(
        "--num_iterations", type=int, required=True,
        help="An integer parameter (e.g., 8)"
    )
    parser.add_argument(
        "--horizon", type=int, default=200,
        help="Horizon (default: 200)"
    )
    parser.add_argument(
        "--initial_points", type=int, default=4,
        help="Number of initial points (default: 4)"
    )
    parser.add_argument(
        "--n_rollouts", type=int, default=10,
        help="Number of rollouts (default: 10)"
    )
    parser.add_argument(
        "--uncertain_camera", type=bool, default=False,
        help="Is camera placement imprecise (default: False)"
    )

    args = parser.parse_args()

    start_time = time.time()
    
    normalized_angles_tensor_test, success_rates_tensor_test = bo_loop(
        args.mode,
        args.batch_size,
        args.num_epochs,
        args.device,
        args.model_dir,
        args.base_model,
        args.raw_data,
        args.dataset_dir,
        args.base_dataset,
        args.num_iterations,
        horizon=args.horizon,
        initial_points=args.initial_points,
        n_rollouts=args.n_rollouts,
        uncertain_camera=args.uncertain_camera
    )
    
    end_time = time.time()
    print("Time elapsed: {:.2f} seconds".format(end_time - start_time))

if __name__ == '__main__':
    main()

# python run_vantage.py --mode "vantage/robomimic/robomimic/exps/templates/bc_transformer.json" --batch_size 256 --num_epochs 50 --device "cuda:0" --model_dir "vantage/base_models/lift_bct" --base_model "vantage/base_models/lift_bct/models/model_epoch_1000.pth" --raw_data "vantage/datasets/lift/ph/demo_v141.hdf5" --dataset_dir "vantage/datasets/lift/ph" --base_dataset "vantage/datasets/lift/ph/angle_0_0.hdf5" --num_iterations 32 --horizon 200 --initial_points 4 --n_rollouts 10