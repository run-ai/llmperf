import openai
from timeit import default_timer as timer

def ttft_measurer(prompt, args):
    openai.api_key = args.api_key
    openai.api_base = args.api_base
    models = openai.Model.list()
    model = models["data"][0]["id"]

    def openai_wrapper():
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
    return openai_wrapper

def tpot_measurer(prompt, args):
    openai.api_key = args.api_key
    openai.api_base = args.api_base
    models = openai.Model.list()
    model = models["data"][0]["id"]

    async def openai_wrapper():
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
        async for _ in completion:
            if i == 0:
                start = timer()
            i += 1
        return (timer() - start) / (i - 1)
    return openai_wrapper