import torch
from argparse import ArgumentParser
from e2e.utils import load_inputs, convert_inputs
import numpy as np
from time import time
import intel_extension_for_pytorch as ipex


def main(folder: str, parallelism: int):
  inputs = load_inputs(folder)
  torch.set_num_interop_threads(parallelism)
  torch.set_num_threads(parallelism)
  model = torch.load(f'out/{folder}/model.pt', weights_only=False)
  tik = time()
  opt_model = torch.compile(model, backend='inductor')
  print(f'inductor compile {folder} took {time() - tik:.6f}s')
  # warm up
  opt_model(**inputs)

  times = 10
  with torch.no_grad():
    total = 0
    for i in range(times):
      start_time = time()
      opt_model(**inputs)
      total += time() - start_time
    print(f'inductor infer {folder} by parallel {parallelism} took {total/times:.6f}s')


if __name__ == '__main__':
  parser = ArgumentParser(description='Process model parameters.')
  parser.add_argument('--folder-name', type=str, help='Name of the folder.', default='qwen2-7B-1')
  parser.add_argument('--parallelism', type=int, help='the max parallelism.', default=1)
  args = parser.parse_args()
  main(args.folder_name, args.parallelism)
