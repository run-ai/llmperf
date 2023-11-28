import openai
from timeit import default_timer as timer

def ttft_measurer(prompt, args):
    openai.api_key = args.api_key
    openai.api_base = args.api_base
    models = openai.Model.list()
    model = models["data"][0]["id"]

    def single_request():
        start = timer()
        completion = openai.Completion.create(
            model=model,
                        echo=False,
                        prompt=prompt,
                        max_tokens=1,
                        temperature=0,
                        n=1,
                        stream=True,
            )
        for _ in completion:
            pass
        return timer() - start
    return single_request

def tpot_measurer(prompt, args):
    openai.api_key = args.api_key
    openai.api_base = args.api_base
    models = openai.Model.list()
    model = models["data"][0]["id"]

    async def single_request():
        start = timer()
        completion = openai.Completion.create(
            model=model,
                        echo=False,
                        prompt=prompt,
                        max_tokens=args.output_tokens,
                        temperature=0,
                        n=1,
                        stream=True,
            )
        i = 0
        for _ in completion:
            if i == 0:
                start = timer()
            i += 1
        return (timer() - start) / (i - 1)
    return single_request

def rate_throughput_measurer(prompt, args):
    openai.api_key = args.api_key
    openai.api_base = args.api_base
    models = openai.Model.list()
    model = models["data"][0]["id"]

    async def single_request():
        completion = await openai.Completion.acreate(
            model=model,
                        echo=False,
                        prompt=prompt,
                        max_tokens=args.output_tokens,
                        temperature=0,
                        n=1,
                        stream=True,
            )
        for _ in completion:
            pass
        return args.output_tokens
    return single_request

def sample_rate_throughput_measurer(args):
    openai.api_key = args.api_key
    openai.api_base = args.api_base
    models = openai.Model.list()
    model = models["data"][0]["id"]

    async def single_request(sample):
        completion = await openai.Completion.acreate(
            model=model,
                        echo=False,
                        prompt=sample["prompt"],
                        max_tokens=sample["output_len"],
                        temperature=0,
                        n=1,
                        stream=True,
            )
        for _ in completion:
            pass
        return sample["output_len"]
    return single_request
