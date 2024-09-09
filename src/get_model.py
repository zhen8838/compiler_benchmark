from argparse import ArgumentParser
from typing import Literal, Union, Tuple
from transformers import LlamaModel, LlamaConfig
import torch


def get_llama(size: Literal["65B", "7B"]) -> Tuple[torch.nn.Module, dict]:
  cfg65b = {
      "bos_token_id": 1,
      "eos_token_id": 2,
      "hidden_act": "silu",
      "hidden_size": 8192,
      "initializer_range": 0.02,
      "intermediate_size": 22016,
      "max_sequence_length": 2048,
      "model_type": "llama",
      "num_attention_heads": 64,
      "num_hidden_layers": 1,  # 80
      "pad_token_id": 0,
      "rms_norm_eps": 1e-05,
      "tie_word_embeddings": False,
      "torch_dtype": "float32",
      "transformers_version": "4.28.0.dev0",
      "use_cache": False,  # true
      "vocab_size": 32000
  }

  cfg = None
  inputs = None
  N = 384
  if size == "65B":
    cfg = cfg65b
    inputs = dict(
        # input_ids=None,
        attention_mask=torch.rand([1, 1, N, N], dtype=torch.float32),
        position_ids=torch.randint(0, N, [1, N], dtype=torch.int64),
        # past_key_values=None,
        inputs_embeds=torch.rand([1, N, 8192], dtype=torch.float32),
        # use_cache=None,
        # output_attentions=None,
        # output_hidden_states=None,
        # return_dict=None,
        # cache_position=None
    )
  configuration = LlamaConfig(**cfg)
  # Initializing a model from the llama-7b style configuration
  return (LlamaModel(configuration), inputs)


def get_model(name: Literal["llama"], format: Literal["onnx"], kwargs: dict):
  pass


def main():
  parser = ArgumentParser(description='Process model parameters.')
  parser.add_argument('--model-name', type=str, help='Name of the model.')
  parser.add_argument('--model-cfg', type=str, help='Configuration of the model.')
  parser.add_argument('--output-format', type=str, help='Output format.')
  args = parser.parse_args()

  match args.model_name:
    case "llama":
      (model, inputs) = get_llama("65B")

  match args.output_format:
    case "onnx":
      filepath = 'out/decoder-65B.onnx'
      torch.onnx.export(model=model, args=inputs, f=filepath, verbose=False)
    case 'pt':
      filepath = 'out/decoder-65B.pt'
      model.eval()
      mod = torch.jit.trace(model, example_kwarg_inputs=inputs, strict=False)
      mod.save(filepath)


if __name__ == '__main__':
  main()
