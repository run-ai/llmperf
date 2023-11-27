import argparse
import openai_perf
import vllm_perf
import tgi_perf
import triton_perf
import asyncio
import math
from vllm.engine.arg_utils import AsyncEngineArgs
from timeit import default_timer as timer

def read_prompt_from_file(file_path):
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt

def run_func_n_times(fn, iterations):
    total = 0
    for i in range(iterations):
        value = fn()
        total += value
        print(f"Iteration {i}: {value}")
    print(f"Average: {total/iterations}")

async def async_run_func_n_times(fn, iterations):
    total = 0
    for i in range(iterations):
        value = await fn()
        total += value
        print(f"Iteration {i}: {value}")
    print(f"Average: {total/iterations}")

async def run_func_periodically(fn, qps, total):
    tasks = []
    start = timer()
    for _ in range(math.floor(total/qps)):
        for _ in range(qps):
            task = asyncio.create_task(fn())
            tasks.append(task)
        await asyncio.sleep(1)
    results = await asyncio.gather(*tasks)
    total_tokens = sum(results)
    elapsed = timer() - start
    return total_tokens / elapsed

def run_ttft(args):
    prompt = read_prompt_from_file(args.prompt_file)
    measurer = None
    if args.engine == "vllm":
        measurer = vllm_perf.ttft_measurer(prompt, args)
    elif args.engine == "openai":
        measurer = openai_perf.ttft_measurer(prompt, args)
    elif args.engine == "tgi":
        measurer = tgi_perf.ttft_measurer(prompt, args)
    elif args.engine == "triton":
        measurer = triton_perf.ttft_measurer(prompt, args)
    else:
        print(f"TTFT test not implemented for {args.engine}")
        return
    run_func_n_times(measurer, args.iterations)

def run_tpot(args):
    prompt = read_prompt_from_file(args.prompt_file)
    measurer = None
    if args.engine == "vllm":
        measurer = vllm_perf.tpot_measurer(prompt, args)
    elif args.engine == "openai":
        measurer = openai_perf.tpot_measurer(prompt, args)
    elif args.engine == "tgi":
        measurer = tgi_perf.tpot_measurer(prompt, args)
    elif args.engine == "triton":
        measurer = triton_perf.tpot_measurer(prompt, args)
    else:
        print(f"TPOT test not implemented for {args.engine}")
        return
    asyncio.run(async_run_func_n_times(measurer, args.iterations))

def run_static_batch(args):
    prompt = read_prompt_from_file(args.prompt_file)
    measurer = None
    if args.engine == "vllm":
        measurer = vllm_perf.static_batch_measurer(prompt, args)
    else:
        print(f"Static batch test not implemented for {args.engine}")
        return
    run_func_n_times(measurer, args.iterations)

def run_rate_throughput(args):
    prompt = read_prompt_from_file(args.prompt_file)
    measurer = None
    if args.engine == "vllm":
        measurer = vllm_perf.rate_throughput_measurer(prompt, args)
    else:
        print(f"Rate throughput test not implemented for {args.engine}")
        return
    
    async def wrapper():
        return await run_func_periodically(measurer, args.qps, args.total_requests)
    asyncio.run(async_run_func_n_times(wrapper, args.iterations))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMPerf tools to measure LLM performance")
    parser.add_argument("--engine", type=str, default="vllm", help="The engine (vllm, openai, tgi)")

    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    ttft_parser = subparsers.add_parser("ttft", help="Measure Time To First Token (TTFT)")
    ttft_parser.add_argument("--model", type=str, default="", help="The model.")
    ttft_parser.add_argument("--dtype", type=str, default="float16", help="The dtype.")
    ttft_parser.add_argument("--prompt_file", type=str, help="Path to a file containing the prompt.")
    ttft_parser.add_argument("--iterations", type=int, default=10, help="The iterations parameter.")
    ttft_parser.add_argument("--api_key", type=str, default="API_KEY", help="The OpenAI API Key")
    ttft_parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1", help="The OpenAI Server URL")


    tpot_parser = subparsers.add_parser("tpot", help="Measure Time Per Output Token (TPOT)")
    tpot_parser.add_argument("--prompt_file", type=str, help="Path to a file containing the prompt.")
    tpot_parser.add_argument("--output_tokens", type=int, default=128, help="Number of tokens to retrieve")
    tpot_parser.add_argument("--iterations", type=int, default=10, help="The iterations parameter.")
    tpot_parser.add_argument("--api_key", type=str, default="API_KEY", help="The OpenAI API Key")
    tpot_parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1", help="The OpenAI Server URL")
    tpot_parser = AsyncEngineArgs.add_cli_args(tpot_parser)

    stb_parser = subparsers.add_parser("static_batch_throughput", help="Measure throughput in static batch")
    stb_parser.add_argument("--model", type=str, default="", help="The model.")
    stb_parser.add_argument("--dtype", type=str, default="float16", help="The dtype.")
    stb_parser.add_argument("--prompt_file", type=str, help="Path to a file containing the prompt.")
    stb_parser.add_argument("--iterations", type=int, default=10, help="The iterations parameter.")
    stb_parser.add_argument("--output_tokens", type=int, default=128, help="Number of tokens to retrieve")
    stb_parser.add_argument("--batch_size", type=int, default=128, help="Number of sequences to batch")

    rth_parser = subparsers.add_parser("rate_throughput", help="Measure throughput with sending requests at constant rate")
    rth_parser.add_argument("--prompt_file", type=str, help="Path to a file containing the prompt.")
    rth_parser.add_argument("--iterations", type=int, default=10, help="The iterations parameter.")
    rth_parser.add_argument("--output_tokens", type=int, default=128, help="Number of tokens to retrieve")
    rth_parser.add_argument("--qps", type=int, default=4, help="Number of queries to send per second")
    rth_parser.add_argument("--total_requests", type=int, default=128, help="Number of requests to send in total")
    rth_parser = AsyncEngineArgs.add_cli_args(rth_parser)
    args = parser.parse_args()


    if args.command == "ttft":
        run_ttft(args)
    elif args.command == "tpot":
        run_tpot(args)
    elif args.command == "static_batch_throughput":
        run_static_batch(args)
    elif args.command == "rate_throughput":
        run_rate_throughput(args)