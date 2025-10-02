import subprocess
import json
import argparse
from evdev import InputDevice, ecodes


# ---- Arm Command ----
def send_arm_command(angle_dict):
    command = ["./multiple_motor_control"]
    for joint_id, angle in angle_dict.items():
        command.extend([str(joint_id), str(angle)])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print("Sent angles:", angle_dict)
    else:
        print("Error:", result.stderr)


def main():
    # ---- Parse arguments ----
    parser = argparse.ArgumentParser(description="Gamepad controlled Unitree D1 arm")
    parser.add_argument(
        "--device",
        type=str,
        default="/dev/input/event7",
        help="Path to gamepad device (default: /dev/input/event7)",
    )
    args = parser.parse_args()

    GAMEPAD_PATH = args.device
    gamepad = InputDevice(GAMEPAD_PATH)

    # ---- Configurable Mapping ----
    button_mapping = {
        ecodes.BTN_WEST:  (2, -1),   # Y
        ecodes.BTN_SOUTH: (2, +1),   # A
        ecodes.BTN_TL:    (6, +15),  # L1
        ecodes.BTN_TR:    (6, -10),  # R1
        ecodes.BTN_START: 'save',    # Start → save
        ecodes.BTN_SELECT: 'reset',  # Select → reset
        ecodes.BTN_MODE:  'exit',    # Home → exit
    }

    dpad_mapping = {
        (ecodes.ABS_HAT0X, -1): (0, +1),   # D-left
        (ecodes.ABS_HAT0X,  1): (0, -1),   # D-right
        (ecodes.ABS_HAT0Y, -1): (1, -1),   # D-up
        (ecodes.ABS_HAT0Y,  1): (1, +1),   # D-down
    }

    # ---- Initialization ----
    angles = {i: 0 for i in range(7)}
    angles[6] += 50
    send_arm_command(angles)
    print("Gamepad control active. Press Home to quit.")

    # ---- Main Loop ----
    count = 0
    for event in gamepad.read_loop():
        if event.type == ecodes.EV_KEY and event.value == 1:  # Button pressed
            code = event.code
            if code in button_mapping:
                action = button_mapping[code]
                if action == 'exit':
                    print("Exiting control.")
                    break
                elif action == 'reset':
                    print("Resetting...")
                    angles = {i: 0 for i in range(7)}
                    angles[6] += 50
                    send_arm_command(angles)
                else:
                    joint, delta = action
                    angles[joint] += delta
                    send_arm_command(angles)

        elif event.type == ecodes.EV_ABS:
            key = (event.code, event.value)
            if key in dpad_mapping:
                joint, delta = dpad_mapping[key]
                angles[joint] += delta
                send_arm_command(angles)


if __name__ == "__main__":
    main()


#python control_gamepad.py --save_path unitree_sdk2/d1_data/raw_data_sep_22.json --device /dev/input/event7
#run "python -m evdev.evtest" to see which event is gamepad connected to
