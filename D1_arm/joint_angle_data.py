import json
import argparse

def convert_interpolated_to_skipfirst(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    result = {}
    for demo, contents in data.items():
        # get action list
        actions = contents.get("action", [])
        # skip the first element
        actions = actions[1:]
        # convert each dict to list of values in numeric key order
        converted = []
        for action in actions:
            row = [action[str(i)] for i in range(len(action))]
            converted.append(row)
        result[demo] = converted

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

def main():
    parser = argparse.ArgumentParser(
        description="Convert interpolated JSON to skipfirst JSON format."
    )
    parser.add_argument("--input", help="Path to the input interpolated JSON file")
    parser.add_argument("--output", help="Path to save the converted skipfirst JSON file")
    
    args = parser.parse_args()
    convert_interpolated_to_skipfirst(args.input, args.output)
    print(f"Converted data saved to {args.output}")

if __name__ == "__main__":
    main()
