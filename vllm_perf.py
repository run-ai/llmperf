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
    def single_request():
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
        llm._run_engine(use_tqdm=False)
        return timer() - start
    return single_request

def tpot_measurer(prompt, args):
    engineArgs = AsyncEngineArgs(args.model)
    engineArgs.trust_remote_code = True
    engineArgs.dtype = args.dtype
    engineArgs.disable_log_stats = True
    engineArgs.disable_log_requests = True
    llm = AsyncLLMEngine.from_engine_args(engineArgs)

    async def single_request():
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
    return single_request

def static_batch_measurer(prompt, args):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype=args.dtype,
    )
    tokenizer = llm.get_tokenizer()
    def single_request():
        sampling_params = SamplingParams(
                temperature=0.0,
                ignore_eos=True,
                max_tokens=args.output_tokens,
            )
        prompt_token_ids = tokenizer.encode(prompt)
        for _ in range(args.batch_size):
            llm._add_request(
                prompt=None,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                )
        start = timer()
        llm._run_engine(use_tqdm=True)
        total_time = timer() - start
        tokens_count = args.batch_size * args.output_tokens
        return tokens_count / total_time
    return single_request

def rate_throughput_measurer(prompt, args):
    llm = init_async_llm(args)

    async def single_request():
        sampling_params = SamplingParams(
                temperature=0.0,
                ignore_eos=True,
                max_tokens=args.output_tokens,
            )
        request_id = random_uuid()
        results_generator = llm.generate(prompt, sampling_params, request_id)
        async for _ in results_generator:
            pass
        return args.output_tokens
    return single_request

def sample_rate_throughput_measurer(args):
    llm = init_async_llm(args)
    async def single_request(sample):
        sampling_params = SamplingParams(
                temperature=0.0,
                ignore_eos=True,
                max_tokens=sample["output_len"],
            )
        request_id = random_uuid()
        results_generator = llm.generate(sample["prompt"], sampling_params, request_id)
        async for _ in results_generator:
            pass
        return sample["output_len"]
    return single_request

def sample_output_rate_throughput_measurer(args):
    llm = init_async_llm(args)
    async def single_request(sample):
        sampling_params = SamplingParams(
                top_k=args.top_k,
                temperature=args.temperature,
                max_tokens=4096,
            )
        request_id = random_uuid()
        results_generator = llm.generate(sample["prompt"], sampling_params, request_id)
        i = 0
        async for _ in results_generator:
            i += 1
        return i
    return single_request

def init_async_llm(args):
    engineArgs = AsyncEngineArgs(args.model)
    engineArgs.trust_remote_code = True
    engineArgs.dtype = args.dtype
    engineArgs.max_num_seqs = args.batch_size
    engineArgs.gpu_memory_utilization = args.gpu_memory_utilization
    engineArgs.disable_log_stats = True
    engineArgs.disable_log_requests = True
    return AsyncLLMEngine.from_engine_args(engineArgs)
