import argparse
import asyncio
from timeit import default_timer as timer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid

async def single_measure(llm, prompt, output_tokens):
    sampling_params = SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=output_tokens,
        )
    request_id = random_uuid()
    results_generator = llm.generate(prompt, sampling_params, request_id)
    i = 0
    async for _ in results_generator:
        if i == 0:
            start = timer()
        i += 1
    return (timer() - start) / (i - 1)

async def measure(llm, prompt, num_iterations, output_tokens):
    total_tpot = 0

    for i in range(num_iterations):
        tpot = await single_measure(llm, prompt, output_tokens)
        total_tpot += tpot
        print(f"Iteration {i + 1}: TPOT: {tpot} seconds")

    average_tpot_time = total_tpot / num_iterations
    print(f"Average for {num_iterations} runs: TPOT: {average_tpot_time} seconds")

def read_prompt_from_file(file_path):
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure time per output token (TPOT) for vLLM")
    parser.add_argument("--prompt_file", type=str, help="Path to a file containing the prompt.")
    parser.add_argument("--output_tokens", type=int, default=128, help="Number of output tokens to generate")
    parser.add_argument("--iterations", type=int, default=10, help="The iterations parameter.")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    llm = AsyncLLMEngine.from_engine_args(engine_args)

    prompt = read_prompt_from_file(args.prompt_file)

    asyncio.run(measure(llm, prompt, args.iterations, args.output_tokens))
    
