import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaTokenizer

import tensorrt_llm
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from timeit import default_timer as timer

from build import get_engine_name  # isort:skip

EOS_TOKEN = 2
PAD_TOKEN = 2


def read_prompt_from_file(file_path):
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt

def read_config(config_path: Path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    remove_input_padding = config['plugin_config']['remove_input_padding']
    dtype = config['builder_config']['precision']
    tp_size = config['builder_config']['tensor_parallel']
    pp_size = config['builder_config']['pipeline_parallel']
    world_size = tp_size * pp_size
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = config['builder_config']['num_heads'] // tp_size
    hidden_size = config['builder_config']['hidden_size'] // tp_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    num_kv_heads = config['builder_config'].get('num_kv_heads', num_heads)
    paged_kv_cache = config['plugin_config']['paged_kv_cache']
    tokens_per_block = config['plugin_config']['tokens_per_block']
    quant_mode = QuantMode(config['builder_config']['quant_mode'])
    if config['builder_config'].get('multi_query_mode', False):
        tensorrt_llm.logger.warning(
            "`multi_query_mode` config is deprecated. Please rebuild the engine."
        )
        num_kv_heads = 1
    num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size
    use_custom_all_reduce = config['plugin_config'].get('use_custom_all_reduce',
                                                        False)

    model_config = ModelConfig(num_heads=num_heads,
                               num_kv_heads=num_kv_heads,
                               hidden_size=hidden_size,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               gpt_attention_plugin=use_gpt_attention_plugin,
                               paged_kv_cache=paged_kv_cache,
                               tokens_per_block=tokens_per_block,
                               remove_input_padding=remove_input_padding,
                               dtype=dtype,
                               quant_mode=quant_mode,
                               use_custom_all_reduce=use_custom_all_reduce)

    return model_config, tp_size, pp_size, dtype

def parse_input(input_text: str, tokenizer, end_id: int,
                remove_input_padding: bool):
    input_tokens = []
    input_tokens.append(
        tokenizer.encode(input_text, add_special_tokens=False))

    input_ids = None
    input_lengths = torch.tensor([len(x) for x in input_tokens],
                                 dtype=torch.int32,
                                 device='cuda')
    if remove_input_padding:
        input_ids = np.concatenate(input_tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.int32,
                                 device='cuda').unsqueeze(0)
    else:
        input_ids = torch.nested.to_padded_tensor(
            torch.nested.nested_tensor(input_tokens, dtype=torch.int32),
            end_id).cuda()

    return input_ids, input_lengths

def single_measure(decoder: any,
                   input_ids: torch.Tensor,
                   input_lengths: torch.Tensor,
                   sampling_config: any
                   ): 
    start = timer()
    output_gen_ids = decoder.decode(input_ids,
                                    input_lengths,
                                    sampling_config,
                                    streaming=True)
    torch.cuda.synchronize()

    ttft = 0
    i = 0
    tpot_start = 0
    for output_ids in enumerate(output_gen_ids):
        i += 1
        if i == 1:
            ttft = timer() - start
            tpot_start = timer()
    return ttft, i - 1, timer() - tpot_start


def generate(
    max_output_len: int,
    log_level: str = 'error',
    engine_dir: str = 'llama_outputs',
    file: str = None,
    tokenizer_dir: str = None,
    iterations: int = None
):
    tensorrt_llm.logger.set_level(log_level)

    if args.file:
        input_text = read_prompt_from_file(file)
    else:
        print("Please specify a file containing the prompt using the --file argument.")
        exit()

    engine_dir = Path(engine_dir)
    config_path = engine_dir / 'config.json'
    model_config, tp_size, pp_size, dtype = read_config(config_path)
    world_size = tp_size * pp_size

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=tp_size,
                                           pp_size=pp_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir, legacy=False)

    sampling_config = SamplingConfig(end_id=EOS_TOKEN,
                                     pad_id=PAD_TOKEN,
                                     num_beams=1,
                                     temperature=0.0)

    engine_name = get_engine_name('llama', dtype, tp_size, pp_size,
                                  runtime_rank)
    serialize_path = engine_dir / engine_name
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping,
                                                     debug_mode=False,
                                                     debug_tensors_to_save=None)
    if runtime_rank == 0:
        print(f"Running the {dtype} engine ...")

    input_ids, input_lengths = parse_input(input_text, tokenizer,
                                           EOS_TOKEN,
                                           model_config.remove_input_padding)

    max_input_length = torch.max(input_lengths).item()
    decoder.setup(input_lengths.size(0), max_input_length, max_output_len, 1)


    output_gen_ids_warmup = decoder.decode(input_ids,
                                    input_lengths,
                                    sampling_config,
                                    streaming=False)
    torch.cuda.synchronize()
    ttft_time = 0
    total_tpot_tokens = 0
    total_tpot_time = 0

    for i in range(iterations):
        ttft, tpot_tokens, tpot_time = single_measure(decoder, input_ids, input_lengths, sampling_config)
        ttft_time += ttft
        total_tpot_tokens += tpot_tokens
        total_tpot_time += tpot_time
        print(f"Iteration {i + 1}: TTFT: {ttft} seconds, {tpot_tokens} TPOT tokens: {tpot_time} seconds")

    average_ttft_time = ttft_time / iterations
    average_tpot_throughput = total_tpot_time / total_tpot_tokens
    print(f"Average for {iterations} runs: TTFT: {average_ttft_time} seconds, TPOT: {average_tpot_throughput} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, default=128, required=True)
    parser.add_argument('--engine_dir', type=str, default='llama_outputs')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default=".",
                        help="Directory containing the tokenizer.model.")
    parser.add_argument("--file", type=str, help="Path to a file containing the prompt.")
    parser.add_argument("--iterations", type=str, default=10, help="The iterations parameter.")
    args = parser.parse_args()
    generate(**vars(args))
