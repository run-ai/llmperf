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