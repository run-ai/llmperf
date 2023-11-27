from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid
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

def tpot_measurer(prompt, args):
    engine_args = AsyncEngineArgs.from_cli_args(args)
    llm = AsyncLLMEngine.from_engine_args(engine_args)
    async def vllm_wrapper():
        sampling_params = SamplingParams(
                temperature=0.0,
                ignore_eos=True,
                max_tokens=args.output_tokens,
            )
        request_id = random_uuid()
        results_generator = llm.generate(prompt, sampling_params, request_id)
        i = 0
        async for _ in results_generator:
            if i == 0:
                start = timer()
            i += 1
        return (timer() - start) / (i - 1)
    return vllm_wrapper