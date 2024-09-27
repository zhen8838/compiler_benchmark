import torch
from argparse import ArgumentParser
from e2e.utils import load_inputs, convert_inputs
import numpy as np


def main(folder: str, parallelism: int):
  inputs = load_inputs(folder)
  torch.set_num_interop_threads(parallelism)
  torch.set_num_threads(parallelism)
  model = torch.jit.load(f'out/onednn/{folder}/model.pt')
  times = 1
  total = np.testing.measure("model(**inputs)", times)
  print(f'onednn infer {folder} took {total/times:.6f}s')


if __name__ == '__main__':
  parser = ArgumentParser(description='Process model parameters.')
  parser.add_argument('--folder-name', type=str, help='Name of the folder.', default='qwen2-7B-1')
  parser.add_argument('--parallelism', type=int, help='the max parallelism.', default=1)
  args = parser.parse_args()
  main(args.folder_name, args.parallelism)
