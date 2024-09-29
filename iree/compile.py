from subprocess import Popen
import subprocess
import sys
from pathlib import Path
import time
from iree.compiler.tools import compile_file
from iree.compiler.tools.binaries import find_tool
from e2e.utils import convert_inputs, load_inputs
from argparse import ArgumentParser


def main(folder: str, parallelism: int):
  model_path = Path(f'out/{folder}/model.onnx')
  outfolder = Path(f'out/iree/{folder}')
  mlir_path = outfolder / 'model.mlir'
  if not outfolder.exists():
    outfolder.mkdir(parents=True)
  if not mlir_path.exists():
    Popen(f"iree-import-onnx {model_path} -o {mlir_path}", shell=True).wait()
  vmfb_path = outfolder / 'model.vmfb'
  tik = time.time()
  iree_compile_exe = find_tool("iree-compile")
  args = [iree_compile_exe,
          f"{mlir_path}",
          "--iree-hal-target-backends=llvm-cpu",
          "--iree-llvmcpu-target-cpu=host",
          "--iree-llvmcpu-target-cpu-features=host",
          "--iree-llvmcpu-loop-interleaving",
          "--iree-llvmcpu-slp-vectorization",
          "--iree-llvmcpu-loop-unrolling",
          "--iree-llvmcpu-loop-vectorization",
          "--iree-llvmcpu-enable-ukernels=all",
          "--iree-opt-aggressively-propagate-transposes",
          "--iree-opt-outer-dim-concat",
          "-o",
          f"{vmfb_path}"]
  subprocess.run(args=args)
  tok = time.time()
  print(f'iree compile {folder} took {tok-tik}s')


if __name__ == '__main__':
  parser = ArgumentParser(description='Process model parameters.')
  parser.add_argument('--folder-name', type=str, help='Name of the folder.', default='qwen2-7B-1')
  parser.add_argument('--parallelism', type=int, help='the max parallelism.', default=1)
  args = parser.parse_args()
  main(args.folder_name, args.parallelism)
