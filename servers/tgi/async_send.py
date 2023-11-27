import asyncio
import argparse
from timeit import default_timer as timer
import math
import json
from text_generation import AsyncClient

endpoint_url = "http://127.0.0.1:80/"
client = AsyncClient(endpoint_url)

start_time = 0
tokens_arrived = 0
active_requests = 0 
remaining_requests = 0

async def send_requests_periodically(prompts, qps, remaining_requests):
    global start_time
    tasks = []
    i = 0
    total_tokens = 0
    for _ in range(math.floor(remaining_requests/qps)):
        for _ in range(qps):
            prompt = prompts[i]
            total_tokens += prompt["output_len"]
            task = asyncio.create_task(send_request(prompt["prompt"], prompt["output_len"]))
            tasks.append(task)
            i += 1
        await asyncio.sleep(1)
    await asyncio.gather(*tasks)
    t = timer() - start_time
    print(f"Throughput: {total_tokens / t}")
    exit()

async def send_request(prompt, max_tokens):
    global remaining_requests, active_requests
    remaining_requests -= 1
    active_requests += 1
    _ = await client.generate(prompt, max_new_tokens=max_tokens)
    active_requests -= 1

async def print_metrics_periodically(interval_seconds):
    global start_time, remaining_requests, active_requests

    while True:
        print(f"Total tokens: {tokens_arrived}, Elapsed time: {timer() - start_time}, Remaining requests: {remaining_requests}, Active requests: {active_requests}")
        await asyncio.sleep(interval_seconds)

async def main():
    global start_time, remaining_requests, engine

    parser = argparse.ArgumentParser(description="vLLM Throughput with QPS")
    parser.add_argument("--print_interval", type=float, default=1, help="Interval in seconds for printing responses")
    parser.add_argument("--qps", type=int, default=4, help="Interval in seconds for sending requests")
    parser.add_argument("--file", type=str, help="Sampled prompt file")
    parser.add_argument("--num_requests", type=int, default=500, help="Number of requests")

    args = parser.parse_args()
    with open(args.file, 'r') as file:
        prompts = json.load(file)

    remaining_requests = args.num_requests

    start_time = timer()
    send_task = asyncio.create_task(send_requests_periodically(prompts, args.qps, remaining_requests))
    print_task = asyncio.create_task(print_metrics_periodically(args.print_interval))

    await asyncio.gather(send_task, print_task)

if __name__ == "__main__":
    asyncio.run(main())