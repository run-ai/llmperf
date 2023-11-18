import argparse
import json
import random
from typing import List, Tuple
from transformers import (AutoTokenizer, PreTrainedTokenizerBase)

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 2048 or prompt_len + output_len > 4096:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_token_ids, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests

def save_prompts_vllm(samples):
    new_array = [(item[0], item[3]) for item in samples]
    json_data = [{"prompt": item[0], "output_len": item[1]} for item in new_array]
    with open('vllm_output.json', 'w') as json_file:
        json.dump(json_data, json_file)

def save_prompts_trtllm(samples):
    new_array = [(item[1], item[3]) for item in samples]
    json_data = [{"input_ids": item[0], "output_len": item[1]} for item in new_array]
    with open('trtllm_output.json', 'w') as json_file:
        json.dump(json_data, json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Path to the dataset.")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
             args.tokenizer,
            trust_remote_code=True)
    sampled_requests = sample_requests(args.dataset, args.num_prompts, tokenizer)
    save_prompts_vllm(sampled_requests)
    save_prompts_trtllm(sampled_requests)
