import requests
import aiohttp
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype
from timeit import default_timer as timer
import numpy as np
from functools import partial
import queue

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

def prepare_tensor(name, input):
    t = grpcclient.InferInput(name, input.shape,
                               np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t

def ttft_measurer(prompt, args):
    server = args.server
    model = args.model
    def single_request():
        req = {
            "text_input": prompt,
            "max_tokens": 1,
            "bad_words": "",
            "stop_words": ""
        }
        start = timer()
        print(f"{server}/v2/models/{model}/generate")
        res = requests.post(f"{server}/v2/models/{model}/generate", json=req)
        print(res)
        return timer() - start
    return single_request

def tpot_measurer(prompt, args):
    client = grpcclient.InferenceServerClient(url=args.server)
    input0 = [[prompt]]
    input0_data = np.array(input0).astype(object)
    output0_len = np.ones_like(input0).astype(np.uint32) * args.output_tokens
    bad_words_list = np.array([[""]], dtype=object)
    stop_words_list = np.array([[""]], dtype=object)
    streaming = [[True]]
    streaming_data = np.array(streaming, dtype=bool)
    beam_width = [[1]]
    beam_width_data = np.array(beam_width, dtype=np.uint32)
    inputs = [
        prepare_tensor("text_input", input0_data),
        prepare_tensor("max_tokens", output0_len),
        prepare_tensor("bad_words", bad_words_list),
        prepare_tensor("stop_words", stop_words_list),
        prepare_tensor("stream", streaming_data),
        prepare_tensor("beam_width", beam_width_data),
    ]

    async def single_request():
        user_data = UserData()
        i = 0
        start = timer()
        def callback(user_data, result, error):
            nonlocal start
            if error:
                user_data._completed_requests.put(error)
            else:
                i += 1
                if i == 1:
                    start = timer()
                user_data._completed_requests.put(result)
        client.start_stream(callback=partial(callback, user_data))
        client.async_stream_infer(args.model, inputs, request_id=1)
        client.stop_stream()
        while True:
            try:
                result = user_data._completed_requests.get(block=False)
            except Exception:
                break

            if type(result) == InferenceServerException:
                print("Received an error from server:")
                print(result)
            else:
                result.as_numpy('text_output')
            return (timer() - start) / (i - 1)
    return single_request

def rate_throughput_measurer(prompt, args):
    server = args.server
    model = args.model
    conn = aiohttp.TCPConnector(limit=None, ttl_dns_cache=300)
    session = aiohttp.ClientSession(connector=conn)
    async def single_request():
        req = {
            "text_input": prompt,
            "max_tokens": args.output_tokens,
            "bad_words": "",
            "stop_words": ""
        }
        async with session.post(f"{server}/v2/models/{model}/generate", json=req) as response:
            _ = await response.text()
        return args.output_tokens
    return single_request

def sample_rate_throughput_measurer(args):
    server = args.server
    model = args.model
    conn = aiohttp.TCPConnector(limit=None, ttl_dns_cache=300)
    session = aiohttp.ClientSession(connector=conn)
    async def single_request(sample):
        req = {
            "text_input": sample["prompt"],
            "max_tokens": sample["output_len"],
            "bad_words": "",
            "stop_words": ""
        }
        async with session.post(f"{server}/v2/models/{model}/generate", json=req) as response:
            _ = await response.text()
        return sample["output_len"]
    return single_request
