from argparse import ArgumentParser
from e2e.utils import load_inputs, convert_inputs
import onnxruntime as ort
import numpy as np
from time import time


def main(folder: str, parallelism: int):
  inputs = load_inputs(folder)
  inputs_np = convert_inputs(inputs, 'numpy')
  # EP_list = ['OpenVINOExecutionProvider']
  options = ort.SessionOptions()
  options.inter_op_num_threads = parallelism
  options.intra_op_num_threads = parallelism
  options.graph_optimization_level = ort.GraphOptimizationLevel(
      ort.capi.onnxruntime_pybind11_state.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
  # options.profile_file_prefix='12'
  # providers=EP_list, provider_options=[{'num_of_threads': parallelism}]
  sess = ort.InferenceSession(f'out/{folder}/model.onnx', options)
  # warm up
  sess.run(None, inputs_np)
  times = 10
  total = 0
  for i in range(times):
    start_time = time()
    sess.run(None, inputs_np)
    total += time() - start_time
  print(f'ort infer {folder} took {total/times:.6f}s')


if __name__ == '__main__':
  parser = ArgumentParser(description='Process model parameters.')
  parser.add_argument('--folder-name', type=str, help='Name of the folder.', default='qwen2-7B-1')
  parser.add_argument('--parallelism', type=int, help='the max parallelism.', default=1)
  args = parser.parse_args()
  main(args.folder_name, args.parallelism)
