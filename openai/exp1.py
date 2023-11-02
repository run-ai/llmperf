import openai
from timeit import default_timer as timer

openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

models = openai.Model.list()
model = models["data"][0]["id"]

start = timer()
completion = openai.Completion.create(
    model=model,
    echo=True,
    prompt="Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. Mr. Dursley made drills. He was a big, beefy man with hardly any neck, although he did have a very large moustache. Mrs. Dursley was thin and blonde and had twice the usual amount of neck, which came in very useful as she spent so much of her time spying on the neighbours. The Dursleys had a small son called Dudley and in their opinion there was no finer",
    max_tokens=100,
    temperature=0,
    n=1,
    stream=True)

for c in completion:
    print(timer() - start)
    exit()

