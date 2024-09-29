import os
from pathlib import Path
import subprocess
import numpy as np
from e2e.utils import convert_inputs, load_inputs
from argparse import ArgumentParser
import iree.runtime as ireert


def main(folder: str):
  inputs = load_inputs(folder)
  model_path = f'out/iree/{folder}/model.vmfb'
  outfolder = Path(f'out/iree/{folder}')
  inputs_np = convert_inputs(inputs, 'numpy')

  # mod = ireert.load_vm_flatbuffer_file(str(model_path), backend='llvm-cpu')
  args = [ireert.benchmark_exe(),
          f"--module={model_path}",
          "--function=main_graph",
          "--device=local-sync",
          "--benchmark_time_unit=s",
          "--print_statistics=true",
          "--benchmark_repetitions=5"]

  for (i, inp) in enumerate(inputs_np.values()):
    shape = "x".join([str(d) for d in inp.shape])
    abitype = ireert.benchmark.DTYPE_TO_ABI_TYPE[inp.dtype]
    input_path = outfolder / f'{i}.npy'
    if not input_path.exists():
      np.save(input_path, inp)

    args.append(f"--input={shape}x{abitype}=@{input_path}")

  benchmark_process = subprocess.run(args=args)


if __name__ == '__main__':
  parser = ArgumentParser(description='Process model parameters.')
  parser.add_argument('--folder-name', type=str, help='Name of the folder.', default='qwen2-7B-1')
  args = parser.parse_args()
  main(args.folder_name)
