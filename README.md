# LLMPerf - Language Model Performance Measurement

## Introduction

LLMPerf is a tool for measuring the performance of large language models (LLMs) such as Llama 2, GPT-3, or similar models. Language models have become increasingly important in various natural language processing tasks, and it's essential to evaluate their performance accurately. LLMPerf provides a framework for conducting performance measurements and benchmarking on these models.

## Features

- **Scalable Testing**: LLMPerf can be used to evaluate language models across various scales and configurations, allowing for comprehensive performance assessments.

- **Performance Metrics**: It provides a range of performance metrics, including response time, throughput, and quality, to help you gain insights into how well your LLM performs under different conditions.

- **Custom Test Scenarios**: LLMPerf allows you to create custom test scenarios that simulate real-world usage, helping you assess model behavior in specific contexts.

- **Extensibility**: You can easily extend LLMPerf to add custom performance metrics or integrate it with your existing infrastructure for automated testing.

- **Reproduction scripts**: You can easily reproduce the results with your own machine

## Getting Started

### Prerequisites

- Machine with NVIDIA GPU
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

#### Run models

##### vLLM
```
python3 -m vllm.entrypoints.openai.api_server --model [MODEL] --trust-remote-code --dtype [PERCISION]
```

##### RayLLM
```
docker run -it --gpus all --shm-size=15gb -p 8000:8000 anyscale/aviary:latest bash
ray start --head

# Copy the content of rayllm/model.yaml to file model.yaml

aviary run --model model.yaml
```

##### TensorRT-LLM
```
# Prepare for compilation
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs install
git lfs pull

# Compile and build TensorRT-LLM docker
make -C docker release_build

# Compile engine
docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 tensorrt_llm/release bash
git lfs install
git clone MODEL
mkdir engine
cd examples/llama/
python build.py --model_dir /app/tensorrt_llm/Llama-2-7b-hf/ \
                --dtype PERCISION \
                --remove_input_padding \
                --use_gpt_attention_plugin PERCISION \
                --enable_context_fmha \
                --use_gemm_plugin PERCISION \
                --max_output_len 2048 \
                --output_dir /app/tensorrt_llm/engine/

```

#### Measurement

##### OpenAI compatible API

```
cd openai
python measure.py --file [TEXT INPUT FILE] --max_tokens [OUTPUT TO GENERATE]
```

## Contributing

We welcome contributions to LLMPerf. If you have ideas for improving the tool or would like to report issues, please check the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to get involved.

## License

LLMPerf is released under the [MIT License](LICENSE). You are free to use, modify, and distribute this software in accordance with the terms of the license.

## Support

If you encounter any issues or have questions about using LLMPerf, please feel free to [open an issue](https://github.com/run-ai/llmperf/issues) on the repository.
