# Scripts for running vLLM Benchmarks

## Common
```
DTYPE: [float16, float32]
PROMPT_FILE: [example: ../../input_examples/llama2/128_tokens]
```

## TTFT

```
python ttft.py --model NousResearch/Llama-2-7b-hf --dtype DTYPE --prompt_file PROMPT_FILE --iterations 10
```

## TPOT
```
python tpot.py --model NousResearch/Llama-2-7b-hf --dtype DTYPE --prompt_file PROMPT_FILE --iterations 10 --trust-remote-code --output_tokens X
```

## Static batch throughput
```
python exp3.py --model NousResearch/Llama-2-7b-hf --dtype DTYPE --prompt_file PROMPT_FILE --output_tokens X --batch_size BATCH_SIZE
```