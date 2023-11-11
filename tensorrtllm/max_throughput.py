import argparse
from timeit import default_timer as timer
from pathlib import Path
import json
import torch
import tensorrt_llm
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from transformers import LlamaTokenizer
from build import get_engine_name  # isort:skip
import numpy as np

EOS_TOKEN = 2
PAD_TOKEN = 2

llm = 0
tokenizer = 0
remove_input_paddings = False

def single_measure(prompt, max_tokens, batch_size):
    global llm, tokenizer, remove_input_paddings
    sampling_config = SamplingConfig(end_id=EOS_TOKEN,
                                     pad_id=PAD_TOKEN,
                                     num_beams=1,
                                     temperature=0.0)
    input_ids_g, input_len_g = parse_input(prompt, tokenizer, EOS_TOKEN, remove_input_paddings, batch_size)
    start = timer()
    output_gen_ids = llm.decode(input_ids_g,
                                input_len_g,
                                sampling_config,
                                streaming=False)
    torch.cuda.synchronize()
    tokens_count = batch_size * max_tokens
    duration = timer() - start
    return tokens_count / duration

def measure(prompt, max_tokens, batch_size, num_iterations):
    sum_throughput = 0

    for i in range(num_iterations):
        throghput = single_measure(prompt, max_tokens, batch_size)
        sum_throughput += throghput
        print(f"Iteration {i + 1} throughput: {throghput}")

    average_throughput = sum_throughput / num_iterations
    print(f"Average throughput: {average_throughput}")

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
                remove_input_padding: bool, batch_size = 1):
    input_tokens = []
    for _ in range(batch_size):
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

def init_model(
    max_output_len: int,
    log_level: str = 'error',
    engine_dir: str = 'llama_outputs',
    file: str = None,
    tokenizer_dir: str = None,
    iterations = 10,
    batch_size = 128):
    global llm, tokenizer, remove_input_paddings
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

    remove_input_paddings = model_config.remove_input_padding
    input_ids_g, input_lengths_g = parse_input(input_text, tokenizer,
                                           EOS_TOKEN,
                                           model_config.remove_input_padding, batch_size)

    max_input_length = torch.max(input_lengths_g).item()
    decoder.setup(input_lengths_g.size(0), max_input_length, max_output_len, 1)
    llm = decoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure maximal throughput for TRT-LLM engine")
    parser.add_argument('--max_output_len', type=int, default=128, required=True)
    parser.add_argument('--engine_dir', type=str, default='llama_outputs')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default=".",
                        help="Directory containing the tokenizer.model.")
    parser.add_argument("--file", type=str, help="Path to a file containing the prompt.")
    parser.add_argument("--iterations", type=str, default=10, help="The iterations parameter.")
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size parameter.")
    args = parser.parse_args()

    init_model(**vars(args))

    if args.file:
        prompt = read_prompt_from_file(args.file)
        measure(prompt, args.max_output_len, args.batch_size, args.iterations)
    else:
        print("Please specify a file containing the prompt using the --file argument.")