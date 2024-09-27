from pathlib import Path
import time
import torch
from argparse import ArgumentParser
from e2e.utils import load_inputs, convert_inputs
import onnxruntime as ort
import numpy as np
import intel_extension_for_pytorch as ipex


def main(folder: str):
  model = torch.load(f'out/{folder}/model.pt', weights_only=False)
  model.eval()
  tik = time.time()
  opt_model: torch.jit.ScriptModule = ipex.llm.optimize(
      model, dtype=torch.float32, device="cpu", inplace=True, deployment_mode=True)
  tok = time.time()
  print(f'onednn compile {folder} took {tok-tik}s')
  outfolder = Path(f'out/onednn/{folder}')
  if not outfolder.exists():
    outfolder.mkdir(parents=True)
  torch.jit.save(opt_model, f'{outfolder}/model.pt')


if __name__ == '__main__':
  parser = ArgumentParser(description='Process model parameters.')
  parser.add_argument('--folder-name', type=str, help='Name of the folder.', default='qwen2-7B-1')
  args = parser.parse_args()
  main(args.folder_name)
