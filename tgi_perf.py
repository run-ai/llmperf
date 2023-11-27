from text_generation import Client
from timeit import default_timer as timer

def ttft_measurer(prompt, args):
    client = Client(args.server)
    def openai_wrapper():
        start = timer()
        _ = client.generate(prompt, max_new_tokens=1)
        return timer() - start
    return openai_wrapper