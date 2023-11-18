import argparse
from timeit import default_timer as timer
from vllm import LLM, SamplingParams

def single_measure(llm, prompt, output_tokens, batch_size):
    sampling_params = SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=output_tokens,
        )
    tokenizer = llm.get_tokenizer()
    prompt_token_ids = tokenizer.encode(prompt)
    for _ in range(batch_size):
        llm._add_request(
            prompt=None,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            )
    start = timer()
    llm._run_engine(use_tqdm=True)
    tokens_count = batch_size * output_tokens
    duration = timer() - start
    return tokens_count / duration

def measure(llm, prompt, num_iterations, output_tokens, batch_size):
    total_results = 0

    for i in range(num_iterations):
        result = single_measure(llm, prompt, output_tokens, batch_size)
        total_results += result
        print(f"Iteration {i + 1}: Throughput: {result} tokens/second")

    average_result = total_results / num_iterations
    print(f"Average for {num_iterations} runs: Throughput: {average_result} tokens/second")

def read_prompt_from_file(file_path):
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure time to first token (TTFT) for vLLM")
    parser.add_argument("--model", type=str, default="", help="The model.")
    parser.add_argument("--dtype", type=str, default="float16", help="The dtype.")
    parser.add_argument("--prompt_file", type=str, help="Path to a file containing the prompt.")
    parser.add_argument("--iterations", type=int, default=10, help="The iterations parameter.")
    parser.add_argument("--output_tokens", type=int, default=128, help="Number of output tokens to generate")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of requests to batch")
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype=args.dtype,
    )

    prompt = read_prompt_from_file(args.prompt_file)
    measure(llm, prompt, args.iterations, args.output_tokens, args.batch_size)
