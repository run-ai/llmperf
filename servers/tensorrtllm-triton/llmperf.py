#!/usr/bin/python

import os
import sys
from functools import partial
from timeit import default_timer as timer

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import queue
import sys

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
from utils import utils

MODEL_NAME = "ensemble"

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def single_measure(triton_client, inputs, request_id):
    user_data = UserData()
    start = 0
    ttft = 0
    i = 0
    tpot_start = 0

    def callback(user_data, result, error):
        nonlocal i
        nonlocal ttft
        nonlocal tpot_start
        if error:
            user_data._completed_requests.put(error)
        else:
            i += 1
            if i == 1:
                ttft = timer() - start
                tpot_start = timer()
            user_data._completed_requests.put(result)
            output = result.as_numpy('text_output')
    # Establish stream
    triton_client.start_stream(callback=partial(callback, user_data))


    start = timer()
    # Send request
    triton_client.async_stream_infer(MODEL_NAME, inputs, request_id=request_id)

    #Wait for server to close the stream
    triton_client.stop_stream()

    # Parse the responses
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
    return ttft, i - 1, timer() - tpot_start
    

def measure(triton_client, inputs, request_id, num_iterations):
    ttft_time = 0
    total_tpot_tokens = 0
    total_tpot_time = 0

    for i in range(num_iterations):
        ttft, tpot_tokens, tpot_time = single_measure(triton_client, inputs, request_id)
        ttft_time += ttft
        total_tpot_tokens += tpot_tokens
        total_tpot_time += tpot_time
        print(f"Iteration {i + 1}: TTFT: {ttft} seconds, {tpot_tokens} TPOT tokens: {tpot_time} seconds")

    average_ttft_time = ttft_time / num_iterations
    average_tpot_throughput = total_tpot_time / total_tpot_tokens
    print(f"Average for {num_iterations} runs: TTFT: {average_ttft_time} seconds, TPOT: {average_tpot_throughput} seconds")

def test(triton_client, prompt, request_id):
    input0 = [[prompt]]
    input0_data = np.array(input0).astype(object)
    output0_len = np.ones_like(input0).astype(np.uint32) * FLAGS.output_len
    bad_words_list = np.array([[""]], dtype=object)
    stop_words_list = np.array([[""]], dtype=object)
    streaming = [[FLAGS.streaming]]
    streaming_data = np.array(streaming, dtype=bool)
    beam_width = [[FLAGS.beam_width]]
    beam_width_data = np.array(beam_width, dtype=np.uint32)

    inputs = [
        utils.prepare_tensor("text_input", input0_data, FLAGS.protocol),
        utils.prepare_tensor("max_tokens", output0_len, FLAGS.protocol),
        utils.prepare_tensor("bad_words", bad_words_list, FLAGS.protocol),
        utils.prepare_tensor("stop_words", stop_words_list, FLAGS.protocol),
        utils.prepare_tensor("stream", streaming_data, FLAGS.protocol),
        utils.prepare_tensor("beam_width", beam_width_data, FLAGS.protocol),
    ]

    measure(triton_client, inputs, request_id, FLAGS.iterations)


def read_prompt_from_file(file_path):
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        help='Inference server URL.')

    parser.add_argument('-f',
                        '--file',
                        type=str,
                        required=True,
                        help='Input prompt file.')
    parser.add_argument(
        "-S",
        "--streaming",
        action="store_true",
        required=False,
        default=False,
        help="Enable streaming mode. Default is False.",
    )

    parser.add_argument(
        "-b",
        "--beam-width",
        required=False,
        type=int,
        default=1,
        help="Beam width value",
    )

    parser.add_argument(
        '-i',
        '--protocol',
        type=str,
        required=False,
        default='grpc',
        choices=['grpc'],
        help='Protocol ("http"/"grpc") used to ' +
        'communicate with inference service. Default is "http".')

    parser.add_argument('-o',
                        '--output_len',
                        type=int,
                        default=100,
                        required=False,
                        help='Specify output length')

    parser.add_argument('--request_id',
                        type=str,
                        default='1',
                        required=False,
                        help='The request_id for the stop request')

    parser.add_argument('--iterations',
                        type=int,
                        default=10,
                        required=False,
                        help='Num of iterations')

    FLAGS = parser.parse_args()
    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"

    try:
        client = grpcclient.InferenceServerClient(url=FLAGS.url)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    prompt = read_prompt_from_file(FLAGS.file)
    test(client, prompt, FLAGS.request_id)