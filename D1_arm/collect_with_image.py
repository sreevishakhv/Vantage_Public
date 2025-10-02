import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
import time
import subprocess
import argparse


def send_arm_command(angle_dict):
    command = ["./multiple_motor_control"]
    for angle_id, angle_value in angle_dict.items():
        command.extend([str(angle_id), str(angle_value)])
    subprocess.run(command, capture_output=True, text=True)


def run(json_path: str, save_root: str, min_demo: int):
    # --- Load JSON data ---
    with open(json_path, "r") as f:
        data = json.load(f)

    # --- Initialize RealSense pipeline ---
    pipeline = rs.pipeline()
    config = rs.config()

    # Check for RGB camera
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The device does not have an RGB camera.")
        return

    # Enable only color stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        for demo_name, demo_data in data.items():
            if int(demo_name.split('_')[1]) > min_demo:
                action_list = demo_data["action"]

                # Create directory if it doesn't exist
                save_dir = os.path.join(save_root, demo_name)
                os.makedirs(save_dir, exist_ok=True)

                print(f"Capturing images for {demo_name}...")

                for idx, action in enumerate(action_list):
                    # Wait for a new color frame
                    while True:
                        frames = pipeline.wait_for_frames()
                        color_frame = frames.get_color_frame()
                        if color_frame:
                            break

                    # Convert image to numpy array
                    color_image = np.asanyarray(color_frame.get_data())

                    # Save image (skip idx 0 to match your original)
                    if idx != 0:
                        filename = os.path.join(save_dir, f"{idx}.png")
                        cv2.imwrite(filename, color_image)
                        print(f"Saved {filename}")

                    # Send arm command
                    send_arm_command(action)

                    # Timing (kept as in your original)
                    time.sleep(1)
                    if idx == 0:
                        time.sleep(6)
                    if idx in [36, 37, 38, 39]:
                        time.sleep(1)

                # Run extra actions (no image capture)
                if "extra" in demo_data:
                    print(f"Running extra actions for {demo_name} (no image capture)...")
                    for extra_action_dict in demo_data["extra"]:
                        extra_action = {int(k): v for k, v in extra_action_dict.items()}
                        send_arm_command(extra_action)
                        time.sleep(3)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Finished capturing all images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture RealSense frames while sending arm commands.")
    parser.add_argument("--json", required=True, help="Path to gen_interpolated.json")
    parser.add_argument("--save-root", required=True, help="Directory to save captured frames")
    parser.add_argument("--min-demo", type=int, default=-1, help="Process demos with ID > min-demo (default: -1 (collects all demos))")
    args = parser.parse_args()

    run(json_path=args.json, save_root=args.save_root, min_demo=args.min_demo)


#python capture_and_move.py --json unitree_sdk2/d1_data/gen_interpolated.json --save-root unitree_sdk2/d1_data/default
