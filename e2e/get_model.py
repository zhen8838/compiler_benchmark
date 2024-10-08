import json
import os
from argparse import ArgumentParser
from typing import Literal, Union, Tuple
import numpy as np
from transformers import LlamaModel, LlamaConfig
from transformers import AutoModel, PretrainedConfig, AutoModelForCausalLM
import torch


def get_llama(size: Literal["65B", "7B"], num_hidden_layers: int = -1) -> Tuple[torch.nn.Module, dict]:
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
  match size:
    case "65B":
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
    case _:
      raise NotImplementedError(size)
  if num_hidden_layers > 0:
    cfg['num_hidden_layers'] = num_hidden_layers
  configuration = LlamaConfig(**cfg)
  # Initializing a model from the llama-7b style configuration
  return (LlamaModel(configuration), inputs, cfg['num_hidden_layers'])


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
      "num_hidden_layers": 28,
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

  N = 384
  match size:
    case '7B':
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
    case _:
      raise NotImplementedError(size)

  cfg = cfg7b
  if num_hidden_layers > 0:
    cfg['num_hidden_layers'] = num_hidden_layers
  configuration = Qwen2Config(**cfg)
  # Initializing a model from the llama-7b style configuration
  return (Qwen2Model(configuration), inputs, cfg['num_hidden_layers'])


def get_deepseekv2(num_hidden_layers: int = 1):
  cfg = PretrainedConfig.from_json_file('e2e/deepseekv2Cfg.json')
  cfg.name_or_path = 'e2e'
  cfg.use_cache = False
  if num_hidden_layers > 0:
    cfg.num_hidden_layers = num_hidden_layers
  N = 384
  inputs = dict(attention_mask=torch.rand([1, N], dtype=torch.float32),
                position_ids=torch.randint(0, N, [1, N], dtype=torch.int64),
                inputs_embeds=torch.rand([1, N, cfg.hidden_size], dtype=torch.float32))
  return (AutoModel.from_config(cfg, trust_remote_code=True), inputs, cfg.num_hidden_layers)


def get_rwkv5(size=Literal["7B"], num_hidden_layers: int = -1):
  match size:
    case '7B':
      cfg = PretrainedConfig.from_json_file('e2e/rwkv5Cfg7B.json')
      cfg.name_or_path = 'e2e'
      cfg.use_cache = False
      if num_hidden_layers > 0:
        cfg.num_hidden_layers = num_hidden_layers
      N = 384
      inputs = dict(attention_mask=torch.rand([1, N], dtype=torch.float32),
                    inputs_embeds=torch.rand([1, N, cfg.hidden_size], dtype=torch.float32))
      return (AutoModelForCausalLM.from_config(cfg, trust_remote_code=True), inputs, cfg.num_hidden_layers)
    case _:
      raise NotImplementedError(size)


def main(model_name: str, model_size: str, num_hidden_layers: int):
  match model_name:
    case "llama":
      (model, inputs, num_layers) = get_llama(model_size, num_hidden_layers)
    case "qwen2":
      (model, inputs, num_layers) = get_qwen2(model_size, num_hidden_layers)
    case 'deepseekv2':
      (model, inputs, num_layers) = get_deepseekv2(num_hidden_layers)
    case 'rwkv5':
      (model, inputs, num_layers) = get_rwkv5(model_size, num_hidden_layers)
    case _:
      raise NotImplementedError(model_name)

  folder = f'out/{model_name}-{model_size}-{num_layers}'
  if not os.path.exists(folder):
    os.mkdir(folder)
  torch.save(inputs, os.path.join(folder, 'inputs.pt'))

  # onnx
  torch.onnx.export(model=model, args=inputs, f=os.path.join(folder, 'model.onnx'),
                    verbose=False, input_names=[k for (k, v) in inputs.items()])
  # pt
  model.eval()
  mod = torch.jit.trace(model, example_kwarg_inputs=inputs, strict=False)
  mod.save(os.path.join(folder, 'model.pt'))

  # fx pt
  try:
    modfx = torch.export.export(model, (), kwargs=inputs)
    torch.export.save(modfx, os.path.join(folder, 'model.pt2'))
  except:
    print("fx export failed!")

  # aot
  try:
    torch._export.aot_compile(model, (), kwargs=inputs, options={
                              "aot_inductor.output_path": os.path.join(folder, "model.so")})
  except:
    print("aot compile failed!")


if __name__ == '__main__':
  parser = ArgumentParser(description='Process model parameters.')
  parser.add_argument('--model-name', type=str, help='Name of the model.',
                      choices=['llama', 'qwen2', 'deepseekv2', 'rwkv5'])
  parser.add_argument('--model-size', type=str, help='size of the model.')
  parser.add_argument('--num-hidden-layers', type=int, help='num decoder layers.', default=-1)
  args = parser.parse_args()
  main(args.model_name, args.model_size, args.num_hidden_layers)
