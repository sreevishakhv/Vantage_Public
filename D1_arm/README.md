# Data Collection and Processing Pipeline

This project provides a pipeline for collecting and processing data for vision-based tasks using the Unitree Go2 robot and D1 arm. Follow the steps below to set up your environment and collect your dataset.

## 1. Install Dependencies

- Create a conda environment, this is where you install all the remaining dependancies

```bash
conda create --name d1 python=3.9
```
### Unitree Go2 SDK 2
- Download and install the [Unitree Go2 SDK 2](https://support.unitree.com/home/en/developer/Obtain%20SDK) from the official Unitree Robotics website.
- Follow the official installation instructions for your platform.

### D1 SDK
- Install the [D1 SDK](https://support.unitree.com/home/en/developer/D1Arm_services) as required for controlling the D1 arm.
- You may have received an executable, but a proper installation is recommended for full control.
- Refer to the D1 SDK documentation for installation steps.

## 2. Data Collection
### Control the arm with Joystick
- Run `control_with_joystick.py` to controll the arm using a joystick interface.
### Collect Data with Joystick
- Run `collect_with_joystick.py` to collect data using a joystick interface.
- This will generate a dataset (e.g., a JSON file) with recorded actions.

```bash
python collect_with_joystick.py
```

## 3. Data Interpolation
- Use `interpolate.py` to interpolate the collected data for smoother trajectories or denser sampling.
- This saves the data as JSON with interpolated joint angles.

```bash
python interpolate.py --input data_from_joystick.json --output interpolated.json
```

## 4. Get actions as a JSON file
### 4a. Actions as change in x,y and z coordinates
> [!IMPORTANT]
> This can be skipped if using ACT policy as it directly uses the joint angles
- Use `apply_forward_kinematics.py` to process the interpolated data and compute the forward kinematics.
- This step will save the actions part of the dataset as delta xyz. 
- Applying forward kinematics

```bash
python apply_forward_kinematics.py --input interpolated.json   --output interpolated_fk.json
```
### 4b. Actions as joint angles
- To get the action part of dataset as joint angles run:
```bash
python joint_angle_data.py --input interpolated.json --output interpolated_ja.json
```

## 5. Collect Image Data
- Use `collect_with_image.py` to collect the image part of the dataset.
- This will save images corresponding to the actions in your dataset.

```bash
python collect_with_image.py --json unitree_sdk2/d1_data/gen_interpolated.json --save-root unitree_sdk2/d1_data/default
```

## 6. Build Vision-Based Dataset
- After completing the above steps, you will have:
  - A JSON file with actions.
  - Images (from step 5)
- Use these together as your vision-based dataset for further training or analysis.

## 7. Train and test ACT policy
- Install requirements for ACT (Action Chunking Transformer) as described in [ACT](https://github.com/tonyzhaozh/act)
- To convert to ACT Dataset format (Images + joint angles JSON):
```bash
python3 act/convert_to_act_hdf5.py   --images_root "imgs"   --actions_json "actions.json"   --out_dir "data"   --camera_name top   --pad_action_to 14  
```
- Train ACT policy:
```bash
python3 act/imitate_episodes.py --task_name lift --ckpt_dir "act/model_0_0" --policy_class ACT --kl_weight 10 --chunk_size 1
0 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 50  --lr 1e-5 --seed 0
```
- Test ACT policy:
```bash
python3 act/test_on_real.py
```
---

**Note:**
- Ensure all SDKs and dependencies are properly installed and configured before running the scripts.
- Refer to the individual script files for additional arguments or configuration options.
