import os, sys, argparse, pickle, numpy as np, torch
from PIL import Image
from einops import rearrange
import pyrealsense2 as rs
from torchvision import transforms
import subprocess
import cv2
import time

STATE_DIM = 14
BACKBONE = 'resnet18'
ENC_LAYERS = 4
DEC_LAYERS = 7
NHEADS = 8
LR_BACKBONE = 1e-5

transform = transforms.Compose([
    transforms.Resize((640, 480)),
    transforms.ToTensor(),
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image_as_tensor(path, size_hw=None):
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device_rs = pipeline_profile.get_device()

    found_rgb = any(s.get_info(rs.camera_info.name) == 'RGB Camera' for s in device_rs.sensors)
    if not found_rgb:
        print("No RGB camera found.")
        exit(0)

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        print("No frame received.")
        exit(0)
    color_image = np.asanyarray(color_frame.get_data())

    # Convert to tensor

    np_img = np.array(color_image, dtype=np.uint8)
    t = torch.from_numpy(rearrange(np_img, 'h w c -> c h w')).float() / 255.0
    return t

def build_policy(policy_class, camera_names, num_queries, hidden_dim, dim_feedforward):
    policy_config = {
        'lr': 1e-4,
        'num_queries': num_queries,
        'kl_weight': 10,
        'hidden_dim': hidden_dim,
        'dim_feedforward': dim_feedforward,
        'lr_backbone': LR_BACKBONE,
        'backbone': BACKBONE,
        'enc_layers': ENC_LAYERS,
        'dec_layers': DEC_LAYERS,
        'nheads': NHEADS,
        'camera_names': camera_names,
    }
    if policy_class == 'ACT':
        from policy import ACTPolicy
        return ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        from policy import CNNMLPPolicy
        return CNNMLPPolicy(policy_config)
    else:
        raise ValueError("policy_class must be 'ACT' or 'CNNMLP'")

def send_arm_command(action_array):
    angle_dict = {i: int(a) for i, a in enumerate(action_array)}
    command = ["./multiple_motor_control"]
    for k, v in angle_dict.items():
        command.extend([str(k), str(v)])
    subprocess.run(command, capture_output=True, text=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--policy_class', required=True, choices=['ACT','CNNMLP'])
    ap.add_argument('--camera_names', nargs='+', required=True)
    ap.add_argument('--image_paths', nargs='+', required=True)
    ap.add_argument('--qpos', nargs='+', type=float, required=True)
    ap.add_argument('--chunk_size', type=int, default=100)
    ap.add_argument('--hidden_dim', type=int, default=512)
    ap.add_argument('--dim_feedforward', type=int, default=3200)
    ap.add_argument('--resize_h', type=int, default=None)
    ap.add_argument('--resize_w', type=int, default=None)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    # --- Spoof argv while importing policy to avoid foreign argparse side-effects ---
    saved_argv = sys.argv[:]
    sys.argv = [saved_argv[0], '--task_name','dummy','--seed','0','--num_epochs','1','--ckpt_dir','dummy','--policy_class','ACT']
    try:
        policy = build_policy(args.policy_class, args.camera_names, args.chunk_size,
                              args.hidden_dim, args.dim_feedforward)
    finally:
        sys.argv = saved_argv

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    state = torch.load(os.path.join(args.ckpt_dir, 'policy_best.ckpt'), map_location=device)
    policy.load_state_dict(state)
    policy.to(device).eval()

    # stats for (de)normalization
    with open(os.path.join(args.ckpt_dir, 'dataset_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)
    pre = lambda q: (q - stats['qpos_mean']) / stats['qpos_std']
    post = lambda a: a * stats['action_std'] + stats['action_mean']

    assert len(args.camera_names) == len(args.image_paths), "camera_names vs image_paths length mismatch"
    qpos_np = np.array(args.qpos, dtype=np.float32); assert qpos_np.size == STATE_DIM
    size_hw = (args.resize_h, args.resize_w) if args.resize_h and args.resize_w else None

    last_action = np.zeros(STATE_DIM, dtype=np.float32)  # init all 0s first time

    last_action = np.zeros(STATE_DIM, dtype=np.float32)  # first time = all zeros
    action_queue = []  # will hold postprocessed actions from the model

    while True:
        # When the queue is empty, replan a new chunk from the current observation + last_action
        if not action_queue:
            cams = [load_image_as_tensor(p, size_hw=size_hw) for p in args.image_paths]
            img = torch.stack(cams, dim=0).unsqueeze(0).to(device)  # (1, Cams, 3, H, W)
            qpos = torch.from_numpy(pre(last_action)).float().unsqueeze(0).to(device)  # (1, 14)

            with torch.inference_mode():
                if args.policy_class == 'ACT':
                    # Get the whole sequence: (1, num_queries, STATE_DIM)
                    all_actions = policy(qpos, img)
                    # Postprocess each step in the chunk and push into queue
                    seq = all_actions.squeeze(0).detach().cpu().numpy()  # (num_queries, STATE_DIM)
                    for a in seq:
                        a = post(a)
                        a = np.round(a).astype(int)
                        action_queue.append(a)
                else:
                    # CNNMLP -> single-step action only
                    raw = policy(qpos, img)  # (1, STATE_DIM)
                    a = post(raw.squeeze(0).detach().cpu().numpy())
                    a = np.round(a).astype(int)
                    action_queue.append(a)

        # Pop the next action from the queue and execute
        next_action = action_queue.pop(0)
        send_arm_command(next_action.tolist()[:7])
        print("Command sent:", next_action.tolist()[:7])

        # Update last_action so next replans use it as qpos
        last_action = next_action.astype(np.float32)

        # Small pause between commands (tune as needed)
        time.sleep(2)


if __name__ == "__main__":
    main()