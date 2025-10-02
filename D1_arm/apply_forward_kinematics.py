import json
import argparse
import numpy as np
from fk import ForwardKinematicsSolver


def process_fk(
    input_path: str,
    output_path: str,
    label_threshold: float = 50.0,
    placeholder_joint6: float = 0.0,
):
    # Load input
    with open(input_path, "r") as f:
        input_data = json.load(f)

    fk_solver = ForwardKinematicsSolver()
    output_data = {}

    for demo, content in input_data.items():
        processed_actions = []
        for action_dict in content.get("action", []):
            # Extract joint angles for indices 0..5  (keys are strings)
            joint_angles_deg = [action_dict[str(i)] for i in range(6)]
            j6_value = action_dict["6"]  # preserve for labeling rule

            # Append placeholder for joint index 6 (kept as in your original)
            joint_angles_deg.append(placeholder_joint6)

            # Convert to radians
            joint_angles_rad = np.radians(joint_angles_deg)

            # Compute FK
            fk_result = fk_solver.compute_fk(joint_angles_rad)

            # Ensure fk_result is a Python list (JSON-serializable)
            if isinstance(fk_result, np.ndarray):
                fk_result = fk_result.tolist()
            else:
                fk_result = list(fk_result)

            # Add binary label: 1 if original "6" < threshold else 0
            label = 1 if j6_value < label_threshold else 0
            fk_result.append(label)

            processed_actions.append(fk_result)

        output_data[demo] = processed_actions

    # Save output
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved processed FK actions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute FK on interpolated demos and append a binary label."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to gen_interpolated.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write gen_fk.json",
    )
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=50.0,
        help='Threshold for action["6"] to set label=1 (default: 50.0).',
    )
    parser.add_argument(
        "--placeholder-joint6",
        type=float,
        default=0.0,
        help="Placeholder value appended as the 7th joint before FK (default: 0.0).",
    )
    args = parser.parse_args()

    process_fk(
        input_path=args.input,
        output_path=args.output,
        label_threshold=args.label_threshold,
        placeholder_joint6=args.placeholder_joint6,
    )


#python apply_forward_kinematics.py --input unitree_sdk2/d1_data/gen_interpolated.json   --output unitree_sdk2/d1_data/gen_fk.json --label-threshold 50 --placeholder-joint6 0
