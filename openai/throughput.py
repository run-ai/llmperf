import aiohttp
import asyncio
import openai
import json
import random
import argparse
import math
from timeit import default_timer as timer
import plotille
from transformers import AutoTokenizer
import threading

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.outputs import RequestOutput
from vllm.utils import random_uuid
from vllm.sampling_params import SamplingParams

openai.api_key = "YOUR_API_KEY"
openai.api_base = "http://localhost:8000/v1"

tokenizer = None
engine = None

start_time = 0
tokens_arrived = 0  # Global variable to store the count of tokens arrived
active_requests = 0  # Global variable to store the count of active requests
remaining_requests = 0

def sample_requests(dataset_path, num_samples):
    with open(dataset_path) as f:
        dataset = json.load(f)
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    dataset = [data["conversations"][0]["value"] for data in dataset]
    filtered_data = []
    for data in dataset:
        input_ids_len = len(tokenizer(data)['input_ids'])
        if input_ids_len >= 2048:
            continue
        filtered_data.append((data, input_ids_len))
        if len(filtered_data) >= num_samples * 10:
            break
    sampled_requests = random.sample(filtered_data, num_samples)
    
    print(plotille.hist([len for _, len in sampled_requests], lc="green", bins=25))
    return [data for data, _ in sampled_requests]

async def send_requests_periodically(reqs, interval_seconds=1):
    global tokens_arrived, active_requests, remaining_requests

    for req in reqs:
        try:
            active_requests += 1
            remaining_requests -= 1
            my_thread = threading.Thread(target=process_request, args=(req,))
            my_thread.start()
            await asyncio.sleep(interval_seconds)
        except Exception as e:
            print(f"Error: {e}")

def process_request(req):
    global tokens_arrived, active_requests  # Declare the global variables
    request_id = f"cmpl-{random_uuid()}"
    try:
        sampling_params = SamplingParams(
            temperature=0,
        )
        result_generator = engine.generate(req, sampling_params, request_id)
        final_res: RequestOutput = None
        for res in result_generator:
            final_res = res
        num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
        tokens_arrived += num_generated_tokens  # Increment the count of tokens arrived
        active_requests -= 1  # Decrement the count of active requests
    except aiohttp.ClientError as e:
        print(f"Request failed with error: {e}")

async def print_metrics_periodically(interval_seconds):
    global start_time, remaining_requests, active_requests
    buffer_for_print = 5

    x = []
    y = []
    while True:
        elapsed = timer() - start_time
        x.append(math.floor(elapsed))
        y.append(tokens_arrived)
        print(f"Total tokens: {tokens_arrived}, Elapsed time: {elapsed}, Remaining requests: {remaining_requests}, Active requests: {active_requests}")

        if buffer_for_print == 0:
            print(plotille.plot(x, y, height=30, width=100, interp="linear", lc="green", X_label="Time", Y_label="Tokens"))
            exit()
        elif remaining_requests == 0 and active_requests == 0:
            buffer_for_print -= 1
        await asyncio.sleep(interval_seconds)

async def main():
    global start_time, remaining_requests, engine, tokenizer

    parser = argparse.ArgumentParser(description="Async OpenAI Requests")
    parser.add_argument("--print_interval", type=float, default=2, help="Interval in seconds for printing responses")
    parser.add_argument("--qps", type=float, default=4, help="Interval in seconds for printing responses")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--dataset", type=str, help="Dataset file")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    requests = sample_requests(args.dataset, args.samples)
    remaining_requests = len(requests)

    requests_interval = 1 / args.qps

    start_time = timer()
    send_task = asyncio.create_task(send_requests_periodically(requests, requests_interval))
    print_task = asyncio.create_task(print_metrics_periodically(args.print_interval))

    await asyncio.gather(send_task, print_task)

if __name__ == "__main__":
    asyncio.run(main())