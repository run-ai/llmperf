import asyncio
import argparse
import json
from timeit import default_timer as timer

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

engine = None

start_time = 0
tokens_arrived = 0
active_requests = 0 

remaining_requests = 0

async def send_requests_periodically(inputs, interval_seconds):
    global start_time, remaining_requests
    tasks = []
    total_tokens = 0
    for (prompt, max_tokens) in inputs:
        total_tokens += max_tokens
        task = asyncio.ensure_future(send_request(prompt, max_tokens))
        tasks.append(task)
        await asyncio.sleep(interval_seconds)  # Introduce a delay of 1/5 second between task creations
    await asyncio.gather(*tasks)
    t = timer() - start_time
    print(f"Throughput: {total_tokens / t}")
    exit()

async def send_request(prompt, max_tokens):
    global remaining_requests, engine, active_requests
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )
    id = random_uuid()
    resps = engine.generate(prompt, sampling_params, id)
    remaining_requests -= 1
    active_requests += 1
    async for rep in resps:
        pass
    active_requests -= 1

async def print_metrics_periodically(interval_seconds):
    global start_time, remaining_requests, active_requests

    while True:
        print(f"Total tokens: {tokens_arrived}, Elapsed time: {timer() - start_time}, Remaining requests: {remaining_requests}, Active requests: {active_requests}")
        await asyncio.sleep(interval_seconds)

def read_inputs_from_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    return [(item["prompt"], item["output_len"]) for item in data]

async def main():
    global start_time, remaining_requests, engine

    parser = argparse.ArgumentParser(description="vLLM Throughput with QPS")
    parser.add_argument("--print_interval", type=float, default=1, help="Interval in seconds for printing responses")
    parser.add_argument("--qps", type=float, default=4, help="Interval in seconds for sending requests")
    parser.add_argument("--file", type=str, help="Sampled prompt file")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)

    args = parser.parse_args()
    inputs = read_inputs_from_file(args.file)
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    remaining_requests = len(inputs)
    requests_interval = 1 / args.qps

    start_time = timer()
    send_task = asyncio.create_task(send_requests_periodically(inputs, requests_interval))
    print_task = asyncio.create_task(print_metrics_periodically(args.print_interval))

    await asyncio.gather(send_task, print_task)

if __name__ == "__main__":
    asyncio.run(main())