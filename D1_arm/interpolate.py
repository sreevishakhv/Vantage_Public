import json
import argparse


def interpolate_dicts(d1, d2, steps):
    """Linear interpolation between two dicts with integer values."""
    result = []
    for i in range(1, steps + 1):
        ratio = i / (steps + 1)
        interpolated = {}
        for k in d1:
            if k == "6":  # preserve key "6" (gripper) as-is
                interpolated[k] = d1[k]
            else:
                interpolated[k] = int(round(d1[k] + (d2[k] - d1[k]) * ratio))
        result.append(interpolated)
    return result


def process_file(input_path: str, output_path: str):
    # Step 1: Load input JSON
    with open(input_path, "r") as f:
        input_data = json.load(f)

    # Flatten into {int_key: dict}
    flat_data = {int(list(d.keys())[0]): list(d.values())[0] for d in input_data}
    sorted_keys = sorted(flat_data.keys())

    # Step 2: group into demo blocks of 9
    demo_dict = {}
    for k in sorted_keys:
        demo_id = k // 9
        demo_dict.setdefault(demo_id, []).append((k, flat_data[k]))

    # Step 3–4: Build output
    final_output = {}
    for demo_id, items in demo_dict.items():
        actions, extras = [], []
        items.sort()

        for i in range(len(items) - 1):
            k1, d1 = items[i]
            k2, d2 = items[i + 1]

            # Assign to extras or actions
            if k1 % 9 in {5, 6, 7}:
                extras.append(d1)
            else:
                actions.append(d1)

            # Interpolation rules
            if k1 % 9 == 0 and k2 % 9 == 1:
                steps = 30
            elif k1 % 9 == 1 and k2 % 9 == 2:
                steps = 5
            elif k1 % 9 == 3 and k2 % 9 == 4:
                steps = 5
            else:
                steps = 0

            interpolated = interpolate_dicts(d1, d2, steps)
            actions.extend(interpolated)

        # Add last entry
        last_key, last_val = items[-1]
        if last_key % 9 in {5, 6, 7, 8}:
            extras.append(last_val)
        else:
            actions.append(last_val)

        final_output[f"demo_{demo_id}"] = {
            "action": actions,
            "extra": extras
        }

    # Step 5: Save to file
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and interpolate JSON demos.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input JSON file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output JSON file")
    args = parser.parse_args()

    process_file(args.input, args.output)


# python interpolate.py --inputunitree_sdk2/d1_sdk/build/generated_data.json --output unitree_sdk2/d1_data/gen_interpolated.json
