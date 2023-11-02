import openai
import argparse
from timeit import default_timer as timer

openai.api_key = "YOUR_API_KEY"
openai.api_base = "http://localhost:8000/v1"

models = openai.Model.list()
model = models["data"][0]["id"]

def single_measure(prompt, max_tokens):
    start = timer()
    completion = openai.Completion.create(
        model=model,
        echo=False,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0,
        n=1,
        stream=True
    )

    ttft = 0
    i = 0
    total_tokens = 0
    itl_start = 0
    for c in completion:
        i += 1
        if i == 1:
            ttft = timer() - start
            itl_start = timer()
        else:
            total_tokens += c['choices'][0]['tokens']
    return ttft, total_tokens, itl_start - start

def measure(prompt, max_tokens, num_iterations):
    ttft_time = 0
    total_itl_tokens = 0
    total_itl_time = 0

    for i in range(num_iterations):
        ttft, itl_tokens, itl_time = single_measure(prompt, max_tokens)
        ttft_time += ttft
        total_itl_tokens += itl_tokens
        total_itl_time += itl_time
        print(f"Iteration {i + 1}: {ttft} seconds, {itl_tokens} ITL tokens in {itl_time} seconds")

    average_ttft_time = ttft_time / num_iterations
    average_itl_throughput = total_itl_tokens / total_itl_time
    print(f"Average for {num_iterations} runs: TTFT: {average_ttft_time} seconds, ITL throughput: {average_itl_throughput} tokens/seconds")

def read_prompt_from_file(file_path):
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure time to first token (TTFT) and inter token latency (ITL) using OpenAI API")
    parser.add_argument("--file", type=str, help="Path to a file containing the prompt.")
    parser.add_argument("--max_tokens", type=int, default=128, help="The max_tokens parameter.")
    parser.add_argument("--iterations", type=int, default=10, help="The iterations parameter.")
    args = parser.parse_args()

    if args.file:
        prompt = read_prompt_from_file(args.file)
        measure(prompt, args.max_tokens, args.iterations)
    else:
        print("Please specify a file containing the prompt using the --file argument.")
