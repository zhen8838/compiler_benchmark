import os
from argparse import ArgumentParser
from typing import Literal, Union, Tuple
import numpy as np
from transformers import LlamaModel, LlamaConfig
import torch


def get_llama(size: Literal["65B", "7B"], num_hidden_layers: int = 1) -> Tuple[torch.nn.Module, dict]:
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
      "num_hidden_layers": num_hidden_layers,  # 80
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


def get_qwen2(size=Literal["7B"], num_hidden_layers: int = 1):
  from transformers import Qwen2Model, Qwen2Config
  cfg7b = {
      "architectures": [
          "Qwen2ForCausalLM"
      ],
      "attention_dropout": 0.0,
      "bos_token_id": 151643,
      "eos_token_id": 151643,
      "hidden_act": "silu",
      "hidden_size": 3584,
      "initializer_range": 0.02,
      "intermediate_size": 18944,
      "max_position_embeddings": 131072,
      "max_window_layers": 28,
      "model_type": "qwen2",
      "num_attention_heads": 28,
      "num_hidden_layers": num_hidden_layers,  # 28,
      "num_key_value_heads": 4,
      "rms_norm_eps": 1e-06,
      "rope_theta": 1000000.0,
      "sliding_window": 131072,
      "tie_word_embeddings": False,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.40.1",
      "use_cache": False,
      "use_mrope": False,
      "use_sliding_window": False,
      "vocab_size": 152064
  }
  N=384
  match size:
    case '7B':
      cfg = cfg7b
      inputs = dict(
        # input_ids=None,
        attention_mask=torch.rand([1, 1, N, N], dtype=torch.float32),
        position_ids=torch.randint(0, N, [1, N], dtype=torch.int64),
        # past_key_values=None,
        inputs_embeds=torch.rand([1, N, 3584], dtype=torch.float32),
        # use_cache=None,
        # output_attentions=None,
        # output_hidden_states=None,
        # return_dict=None,
        # cache_position=None
      )

  configuration = Qwen2Config(**cfg)
  # Initializing a model from the llama-7b style configuration
  return (Qwen2Model(configuration), inputs)


def main():
  parser = ArgumentParser(description='Process model parameters.')
  parser.add_argument('--model-name', type=str, help='Name of the model.')
  parser.add_argument('--model-size', type=str, help='size of the model.')
  parser.add_argument('--num-hidden-layers', type=int, help='num decoder layers.', default=1)
  args = parser.parse_args()

  match args.model_name:
    case "llama":
      (model, inputs) = get_llama(args.model_size, args.num_hidden_layers)
    case "qwen2":
      (model, inputs) = get_qwen2(args.model_size, args.num_hidden_layers)

  folder = f'out/{args.model_name}-{args.model_size}-{args.num_hidden_layers}'
  if not os.path.exists(folder):
    os.mkdir(folder)
  torch.save(inputs, os.path.join(folder, 'inputs.pt'))

  # onnx
  torch.onnx.export(model=model, args=inputs, f=os.path.join(folder, 'model.onnx'), verbose=False)
  # pt
  model.eval()
  mod = torch.jit.trace(model, example_kwarg_inputs=inputs, strict=False)
  mod.save(os.path.join(folder, 'model.pt'))


if __name__ == '__main__':
  main()
