# Scripts for running TensorRT-LLM Benchmarks

Build the original benchmarks supplied with the TensorRT-LLM repo.
Replace the `gptManagerBenchmark.cpp` / `gptSessionBenchmark.cpp` file with the content of the llmperf file and rebuild it

## Common
```
DTYPE: [float16, float32]
PROMPT_FILE: [example: ../../input_examples/llama2/128_tokens]
```

ttft.cpp -> gptSessionBenchmark.cpp
## TTFT

```
./benchmarks/gptSessionBenchmark --model llama --engine_dir /share/engine/ --batch_size 1 --input_output_len "128,128" --num_runs 100
```

tpot.cpp -> gptSessionBenchmark.cpp
## TPOT
```
./benchmarks/gptSessionBenchmark --model llama --engine_dir /share/engine/ --batch_size 1 --input_output_len "128,128" --num_runs 100
```

## Static batch throughput
```
python exp3.py --model NousResearch/Llama-2-7b-hf --dtype DTYPE --prompt_file PROMPT_FILE --output_tokens X --batch_size BATCH_SIZE
```