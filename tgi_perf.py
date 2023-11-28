from text_generation import Client, AsyncClient
from timeit import default_timer as timer

def ttft_measurer(prompt, args):
    client = Client(args.server)
    def single_request():
        start = timer()
        _ = client.generate(prompt, max_new_tokens=1)
        return timer() - start
    return single_request

def tpot_measurer(prompt, args):
    client = Client(args.server)
    async def single_request():
        i = 0
        for _ in client.generate_stream(prompt, max_new_tokens=args.output_tokens):
            if i == 0:
                start = timer()
            i += 1
        return (timer() - start) / (i - 1)
    return single_request

def rate_throughput_measurer(prompt, args):
    client = AsyncClient(args.server)
    async def single_request():
        _ = await client.generate(prompt, max_new_tokens=args.output_tokens)
        return args.output_tokens
    return single_request

def sample_rate_throughput_measurer(args):
    client = AsyncClient(args.server, timeout=1440)
    async def single_request(sample):
        _ = await client.generate(sample["prompt"], max_new_tokens=sample["output_len"])
        return sample["output_len"]
    return single_request

def sample_output_rate_throughput_measurer(args):
    client = AsyncClient(args.server, timeout=1440)
    async def single_request(sample):
        response = await client.generate(sample["prompt"])
        print(response.details.generated_tokens)
        return response.details.generated_tokens
    return single_request
