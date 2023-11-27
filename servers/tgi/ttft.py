import argparse
from timeit import default_timer as timer
import requests
import json

url = "http://127.0.0.1:80/generate"
headers = {'Content-Type': 'application/json'}

def single_measure(prompt):
    data = {
        'inputs': prompt,
        'parameters': {
            'max_new_tokens': 1
        }
    }
    data_json = json.dumps(data)
    start = timer()
    _ = requests.post(url, data=data_json, headers=headers)
    return timer() - start

def measure(prompt, num_iterations):
    ttft_time = 0

    for i in range(num_iterations):
        ttft = single_measure(prompt)
        ttft_time += ttft
        print(f"Iteration {i + 1}: TTFT: {ttft} seconds")

    average_ttft_time = ttft_time / num_iterations
    print(f"Average for {num_iterations} runs: TTFT: {average_ttft_time} seconds")

def read_prompt_from_file(file_path):
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure time to first token (TTFT) for TGI")
    parser.add_argument("--prompt_file", type=str, help="Path to a file containing the prompt.")
    parser.add_argument("--iterations", type=int, default=10, help="The iterations parameter.")
    args = parser.parse_args()

    prompt = read_prompt_from_file(args.prompt_file)
    measure(prompt, args.iterations)
