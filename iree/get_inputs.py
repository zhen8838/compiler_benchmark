import os
from e2e.utils import convert_inputs, load_inputs
from argparse import ArgumentParser
import numpy as np

typeMap = {
    'float32': 'f32',
    'int64': 'i64',
}


def main(folder: str):
  inputs = load_inputs(folder)
  outfolder = f'out/iree/{folder}'
  if not os.path.exists(outfolder):
    os.makedirs(outfolder)
  inputs_np = convert_inputs(inputs, 'numpy')
  count = 0
  for (_, v) in inputs_np.items():
    np.save(f"{outfolder}/input{count}.npy", v)
    shape = 'x'.join([str(s) for s in v.shape])
    dtype = typeMap[str(v.dtype)]
    print(f"--input={shape}x{dtype}=@{outfolder}/input{count}.npy")
    count += 1


if __name__ == '__main__':
  parser = ArgumentParser(description='Process model parameters.')
  parser.add_argument('--folder-name', type=str, help='Name of the folder.', default='qwen2-7B-1')
  args = parser.parse_args()
  main(args.folder_name)
