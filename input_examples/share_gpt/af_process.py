import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--input",
                        type=str,
                        required=True,
                        help="Path to the dataset.")
    parser.add_argument("--vllm", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    with open(args.input, 'r') as file:
        data = json.load(file)
    with open(args.vllm, 'r') as file:
        vllm_data = json.load(file)
    
    for i in range(len(data)):
        vllm_data[i]['input_len'] = len(data[i]['input_ids'])

    with open('vllm_output_2.json', 'w') as json_file:
        json.dump(vllm_data, json_file)
