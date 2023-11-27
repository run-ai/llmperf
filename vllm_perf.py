from vllm import LLM, SamplingParams
from timeit import default_timer as timer

def ttft_measurer(prompt, args):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype=args.dtype,
    )
    tokenizer = llm.get_tokenizer()
    def vllm_wrapper():
        sampling_params = SamplingParams(
                temperature=0.0,
                ignore_eos=True,
                max_tokens=1,
            )
        prompt_token_ids = tokenizer.encode(prompt)
        llm._add_request(
                prompt=None,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                )
        start = timer()
        llm._run_engine(use_tqdm=True)
        return timer() - start
    return vllm_wrapper