import openai
import argparse
from timeit import default_timer as timer
import time

openai.api_key = "YOUR_API_KEY"
openai.api_base = "http://localhost:8000/v1"

models = openai.Model.list()
model = models["data"][0]["id"]

def single_measure(prompt, max_tokens, user):
    start = timer()
    completion = openai.Completion.create(
        model=model,
        echo=False,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0,
        n=1,
        stream=True,
        headers={"user": user}
    )

    ttft = 0
    i = 0
    tpot_start = 0
    for c in completion:
        i += 1
        if i == 1:
            ttft = timer() - start
            tpot_start = timer()
    return ttft, i - 1, timer() - tpot_start

def measure(prompt, max_tokens, num_iterations, user, pause):
    ttft_time = 0
    total_tpot_tokens = 0
    total_tpot_time = 0

    for i in range(num_iterations):
        ttft, tpot_tokens, tpot_time = single_measure(prompt, max_tokens, user)
        ttft_time += ttft
        total_tpot_tokens += tpot_tokens
        total_tpot_time += tpot_time
        print(f"Iteration {i + 1}: TTFT: {ttft} seconds, {tpot_tokens} TPOT tokens: {tpot_time} seconds")
        time.sleep(pause)

    average_ttft_time = ttft_time / num_iterations
    average_tpot_throughput = total_tpot_time / total_tpot_tokens
    print(f"Average for {num_iterations} runs: TTFT: {average_ttft_time} seconds, TPOT: {average_tpot_throughput} seconds")

def read_prompt_from_file(file_path):
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure time to first token (TTFT) and inter token latency (ITL) using OpenAI API")
    parser.add_argument("--file", type=str, help="Path to a file containing the prompt.")
    parser.add_argument("--max_tokens", type=int, default=128, help="The max_tokens parameter.")
    parser.add_argument("--iterations", type=int, default=10, help="The iterations parameter.")
    parser.add_argument("--pause", type=int, default=1, help="The pause parameter.")
    parser.add_argument("--user", type=str, default="default", help="The user")
    args = parser.parse_args()

    if args.file:
        prompt = read_prompt_from_file(args.file)
        measure(prompt, args.max_tokens, args.iterations, args.user, args.pause)
    else:
        print("Please specify a file containing the prompt using the --file argument.")
