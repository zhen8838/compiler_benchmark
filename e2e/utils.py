from typing import Dict, Literal
import torch


def convert_inputs(inputs: Dict[str, torch.Tensor], format: Literal['numpy', 'shape', 'dtype']):
  match format:
    case 'numpy':
      return {k: v.numpy() for (k, v) in inputs.items()}
    case 'shape':
      return {k: list(v.shape) for (k, v) in inputs.items()}
    case 'dtype':
      return {k: str(v.dtype)[6:] for (k, v) in inputs.items()}


def load_inputs(folder: str) -> Dict[str, torch.Tensor]:
  return torch.load(f'out/{folder}/inputs.pt', weights_only=True)


if __name__ == '__main__':
  folder = "qwen2-7B-1"
  inputs: Dict[str, torch.Tensor] = torch.load(f'out/{folder}/inputs.pt')
  print(convert_inputs(inputs, 'shape'))
  print(convert_inputs(inputs, 'dtype'))
  print(convert_inputs(inputs, 'numpy'))
