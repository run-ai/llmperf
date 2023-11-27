from text_generation import Client
from timeit import default_timer as timer

def ttft_measurer(prompt, args):
    client = Client(args.server)
    def openai_wrapper():
        start = timer()
        _ = client.generate(prompt, max_new_tokens=1)
        return timer() - start
    return openai_wrapper

def tpot_measurer(prompt, args):
    client = Client(args.server)
    async def openai_wrapper():
        i = 0
        for _ in client.generate_stream(prompt, max_new_tokens=args.output_tokens):
            if i == 0:
                start = timer()
            i += 1
        return (timer() - start) / (i - 1)
    return openai_wrapper