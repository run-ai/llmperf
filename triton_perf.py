import requests
from timeit import default_timer as timer

def ttft_measurer(prompt, args):
    server = args.server
    model = args.model
    def openai_wrapper():
        req = {
            "text_input": prompt,
            "max_tokens": 1,
            "bad_words": "",
            "stop_words": ""
        }
        start = timer()
        _ = requests.post(f"{server}/v2/models/{model}/generate", json=req)
        return timer() - start
    return openai_wrapper