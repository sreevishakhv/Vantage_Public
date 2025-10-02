# Vantage

This repository contains the official implementation of the paper: Viewpoint-Agnostic Manipulation Policies with Strategic Vantage Selection

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
* Python 3.9+
* PyTorch

You can install PyTorch by following the official instructions on their website: [pytorch.org](https://pytorch.org/get-started/locally/).

---

## ⚙️ Installation

Follow these steps to set up the project environment.

1.  **Clone the Repository**
    Clone this repository to your local machine:
    ```bash
    git clone https://github.com/sreevishakhv/Vantage_Public.git
    cd Vantage
    ```

2.  **Install Dependencies**
    First, install the specified version of `robosuite`:
    ```bash
    pip install robosuite==1.4.1
    ```

3.  **Install Project Package**
    Install this project in editable mode, which will also handle other dependencies listed in `setup.py`:
    ```bash
    pip install -e .
    ```

---

## 💾 Data Preparation

Before running the training, you need to download the necessary datasets and process them.

1.  **Download Robomimic Datasets**
    Run the `download_datasets.py` script (from the `robomimic` library) to download the low-dimensional proficient-human (`ph`) datasets.
    ```bash
    python -m robomimic.scripts.download_datasets --tasks sim --dataset_types ph --hdf5_types low_dim
    ```

2.  **Generate Image Datasets**
    Next, convert the downloaded low-dimensional dataset into an image-based dataset. This script will render observations from the `agentview` camera.
    ```bash
    python -m robomimic.scripts.dataset_states_to_obs --dataset /path/to/your/low_dim_dataset.hdf5 --output_name image_dataset.hdf5 --done_mode 2 --camera_names agentview --camera_height 84 --camera_width 84
    ```
    **Note**: Make sure to replace `/path/to/your/low_dim_dataset.hdf5` with the actual path to the dataset you downloaded in the previous step.

---

## 🚀 Usage

Once the installation and data preparation are complete, you can start the vantage process.

Execute the following command from the root directory of the project:
```bash
python run_vantage.py --mode "vantage/robomimic/robomimic/exps/templates/ bc_transformer.json" --batch_size 256 --num_epochs 50 --device "cuda:0" --model_dir "vantage/base_models/lift_bct" --base_model "vantage/base_models/lift_bct/models/model_epoch_1000.pth" --raw_data "vantage/datasets/lift/ph/demo_v141.hdf5" --dataset_dir "vantage/datasets/lift/ph" --base_dataset "vantage/datasets/lift/ph/angle_0_0.hdf5" --num_iterations 32 --horizon 200 --initial_points 4 --n_rollouts 10
```


## 🦾 Real Robot Experiments

Data collection and model training codes for the real robot (Unitree D1) are provided in the folder `unitree_d1_experiments`.
