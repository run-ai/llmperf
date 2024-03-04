# LLMPerf - Language Model Performance Measurement

LLMPerf is a tool for measuring the throughput and latency performance of large language models (LLMs) such as Llama 2, GPT-3, or similar models. LLMPerf provides a framework for conducting performance measurements and benchmarking on these models using following frameworks:

|Type of the framework | Framework |
| --- | --- |
| Engine | [vLLM,](https://github.com/vllm-project/vllm) <be>[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) |
| Engine & Server | [TGI,](https://github.com/huggingface/text-generation-inference) <be>[Triton with TensorRT-LLM,](https://github.com/triton-inference-server) <be>[RayLLM with vLLM](https://github.com/ray-project/ray-llm)|

## Our Findings

For our benchmarking results with these frameworks, please refer to [our whitepaper](https://pages.run.ai/hubfs/PDFs/Serving-Large-Language-Models-Run-ai-Benchmarking-Study.pdf).

## Features

- **Scalable Testing**: LLMPerf can be used to evaluate language models across various scales and configurations, allowing for comprehensive performance assessments.

- **Performance Metrics**: It provides a range of performance metrics, including response time, throughput, and quality, to help you gain insights into how well your LLM performs under different conditions.

- **Custom Test Scenarios**: LLMPerf allows you to create custom test scenarios that simulate real-world usage, helping you assess model behavior in specific contexts.

- **Extensibility**: You can easily extend LLMPerf to add custom performance metrics or integrate it with your existing infrastructure for automated testing.

- **Reproduction scripts**: You can easily reproduce the results with your own machine

## Getting Started

### Prerequisites

- Machine with NVIDIA GPU (Tested on A100-40GB)
- Python 3.6 or higher
- Pip (Python package manager)

### Installation

1. Clone the LLMPerf repository:

```git clone https://github.com/run-ai/llmperf.git```

2. Navigate to the LLMPerf directory:

```cd llmperf```

3. Install the required Python packages:

```pip install -r requirements.txt```

### Usage

#### Run benchmarking experiments

##### vLLM
```
# Benchmarking time per output token using vLLM
python llmperf.py tpot --prompt_file input_examples/llama2/128_tokens --iterations 10 --output_tokens 128 vllm --model NousResearch/Llama-2-7b-chat-hf --dtype float16
```

##### RayLLM

```
docker run -it --gpus all --shm-size=15gb -p 8000:8000 anyscale/aviary:latest bash
ray start --head

# Copy the content of rayllm/model.yaml to file model.yaml

aviary run --model model.yaml
```

##### TensorRT-LLM

To be able to run experiments, TensorRT-LLM needs to be installed on your machine and the engine needs to be created. Please refer to [this GitHub repository](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/installation.md) for these steps. 

```
# Benchmarking time per output token using TensorRT-LLM
./benchmarks/gptSessionBenchmark \
    --model llama \
    --engine_dir "/app/tensorrt_llm/engine/" \
    --batch_size "32" \
    --input_output_len "128,128"


# Benchmarking throughput using TensorRT-LLM
./benchmarks/gptManagerBenchmark \
    --model llama \
    --engine_dir "/app/tensorrt_llm/engine/" \
    --batch_size "32" \
    --input_output_len "128,128"
```

For more information about the settings and possible flags, please refer to [the documentation](https://github.com/NVIDIA/TensorRT-LLM/blob/rel/benchmarks/cpp/README.md).

#### Triton - TRT-LLM
```
~/triton-trtllm$ docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v /home/omer/triton-trtllm/model-repo/:/model-repo -v /home/omer/share/:/share nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3 tritonserver --model-repo=/model-repo/
```

#### Measurement

##### OpenAI compatible API (vLLM / RayLLM)

```
cd openai
python openai/llmperf.py --file [TEXT INPUT FILE] --max_tokens [OUTPUT TO GENERATE]
```

## Contributing

We welcome contributions to LLMPerf. If you have ideas for improving the tool or would like to report issues, please check the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to get involved.

## License

LLMPerf is released under the [MIT License](LICENSE). You are free to use, modify, and distribute this software in accordance with the terms of the license.

## Support

If you encounter any issues or have questions about using LLMPerf, please feel free to [open an issue](https://github.com/run-ai/llmperf/issues) on the repository.
