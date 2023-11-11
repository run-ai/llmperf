import argparse
from timeit import default_timer as timer
from vllm import LLM, SamplingParams

llm = 0

def single_measure(prompt, max_tokens, batch_size):
    sampling_params = SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=max_tokens,
        )
    for _ in range(batch_size):
        llm._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
            )
    start = timer()
    llm._run_engine(use_tqdm=True)
    tokens_count = batch_size * max_tokens
    duration = timer() - start
    return tokens_count / duration

def measure(prompt, max_tokens, batch_size, num_iterations):
    sum_throughput = 0

    for i in range(num_iterations):
        throghput = single_measure(prompt, max_tokens, batch_size)
        sum_throughput += throghput
        print(f"Iteration {i + 1} throughput: {throghput}")

    average_throughput = sum_throughput / num_iterations
    print(f"Average throughput: {average_throughput}")


def read_prompt_from_file(file_path):
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure maximal throughput for vLLM engine")
    parser.add_argument("--file", type=str, help="Path to a file containing the prompt.")
    parser.add_argument("--output_tokens", type=int, default=128, help="The max_tokens parameter.")
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size parameter.")
    parser.add_argument("--iterations", type=int, default=10, help="The iterations to perform the test.")
    parser.add_argument("--dtype", type=str, default="float16", help="The dtype.")
    parser.add_argument("--model", type=str, default="", help="The model.")
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype=args.dtype,
        gpu_memory_utilization=1,
    )

    if args.file:
        prompt = read_prompt_from_file(args.file)
        measure(prompt, args.output_tokens, args.batch_size, args.iterations)
    else:
        print("Please specify a file containing the prompt using the --file argument.")