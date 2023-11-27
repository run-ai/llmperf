import argparse
import asyncio
from timeit import default_timer as timer
from text_generation import Client

endpoint_url = "http://127.0.0.1:80/"
client = Client(endpoint_url)

async def single_measure(prompt, output_tokens):
    i = 0
    for res in client.generate_stream(prompt, max_new_tokens=output_tokens):
        if i == 0:
            start = timer()
        i += 1
    return (timer() - start) / (i - 1)

async def measure(prompt, num_iterations, output_tokens):
    total_tpot = 0

    for i in range(num_iterations):
        tpot = await single_measure(prompt, output_tokens)
        total_tpot += tpot
        print(f"Iteration {i + 1}: TPOT: {tpot} seconds")

    average_tpot_time = total_tpot / num_iterations
    print(f"Average for {num_iterations} runs: TPOT: {average_tpot_time} seconds")

def read_prompt_from_file(file_path):
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure time per output token (TPOT) for TGI")
    parser.add_argument("--prompt_file", type=str, help="Path to a file containing the prompt.")
    parser.add_argument("--output_tokens", type=int, default=128, help="Number of output tokens to generate")
    parser.add_argument("--iterations", type=int, default=10, help="The iterations parameter.")
    args = parser.parse_args()
    prompt = read_prompt_from_file(args.prompt_file)

    asyncio.run(measure(prompt, args.iterations, args.output_tokens))
    
