import tvm
import numpy as np
from tvm import relax
import numpy as np
import osarch
from e2e.utils import convert_inputs, load_inputs
from argparse import ArgumentParser


def main(folder: str, profile: bool):
  ext = 'so' if osarch.detect_system_os() == 'linux' else 'dylib'

  mod = tvm.runtime.load_module(f"out/tvm/{folder}/model.{ext}")
  dev = tvm.cpu(0)
  vm = tvm.relax.VirtualMachine(mod, dev, profile=profile)
  inputs = load_inputs(folder)
  inputs_np = convert_inputs(inputs, 'numpy')
  inputs_tvm = list([tvm.nd.array(v, dev) for (k,v) in inputs_np.items()])
  if profile:
    report = vm.profile('main', *inputs_tvm)
    print(report)
  else:
    evaluator = vm.time_evaluator("main", dev, number=10)
    eval_res = evaluator(*inputs_tvm)
    print(eval_res)

if __name__ == '__main__':
  parser = ArgumentParser(description='Process model parameters.')
  parser.add_argument('--folder-name', type=str, help='Name of the folder.', default='qwen2-7B-1')
  parser.add_argument('--profile', type=bool, help='enable profile.', default=False)
  args = parser.parse_args()
  main(args.folder_name, args.profile)
