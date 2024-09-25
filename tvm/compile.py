import os
import torch
import tvm
from tvm import relax
import tvm.meta_schedule as ms
from tvm.relax.frontend.onnx import from_onnx
import onnx
import osarch
from e2e.utils import convert_inputs, load_inputs
from argparse import ArgumentParser


@tvm.transform.module_pass(opt_level=0)
class TuneIRMod:
  def __init__(self, work_dir: str, max_trials_global: int, max_jobs_per_core: int = 1):
    self.max_jobs_per_core = max_jobs_per_core
    self.work_dir = work_dir
    self.max_trials_global = max_trials_global

  def transform_module(self, mod: tvm.IRModule, ctx: tvm.ir.transform.PassContext):
    target: tvm.target.Target = tvm.target.Target.current(False)
    rules = ms.ScheduleRule.create(target.kind.name)
    newrules = []
    for rule in rules:
      if isinstance(rule, ms.schedule_rule.ParallelizeVectorizeUnroll):
        rule: ms.schedule_rule.ParallelizeVectorizeUnroll
        newrules.append(ms.schedule_rule.ParallelizeVectorizeUnroll(
            self.max_jobs_per_core, rule.max_vectorize_extent, rule.unroll_max_steps, rule.unroll_explicit))
      else:
        newrules.append(rule.clone())
    mutators = ms.Mutator.create(target.kind.name)
    newmutators = []
    for m in mutators:
      if isinstance(m, ms.mutator.MutateParallel):
        if self.max_jobs_per_core > 1:
          newmutators.append(ms.mutator.MutateParallel(self.max_jobs_per_core))
      else:
        newmutators.append(m.clone())
    sg = ms.space_generator.PostOrderApply(sch_rules=newrules)

    ms.tune_tir(
        mod=mod,
        target=target,
        work_dir=self.work_dir,
        max_trials_global=self.max_trials_global,
        space=sg,
    )

    return mod


def main(folder: str, parallelism: int, total_trials: int):
  # config target
  model_path = f"out/{folder}/model.onnx"
  target = tvm.target.Target(
      f"llvm -mtriple={tvm.target.codegen.llvm_get_system_triple()} -mcpu={tvm.target.codegen.llvm_get_system_cpu()} -num-cores=1")
  ext = 'so' if osarch.detect_system_os() == 'linux' else 'dylib'

  # load model
  onnx_model = onnx.load_model(model_path, load_external_data=True)
  inputs = load_inputs(folder)
  input_shapes = convert_inputs(inputs, 'shape')
  input_dtypes = convert_inputs(inputs, 'dtype')
  mod: tvm.IRModule = from_onnx(onnx_model, input_shapes, input_dtypes)
  outfolder = f"out/tvm/{folder}"
  database_dir = f"{outfolder}/tuning_logs"

  with target, tvm.ir.transform.PassContext(opt_level=0):
    mod = tvm.ir.transform.Sequential([
        # Convert BatchNorm into a sequence of simpler ops for fusion
        relax.transform.DecomposeOpsForInference(),
        # Canonicalize the bindings
        relax.transform.CanonicalizeBindings(),
        # Run default optimization pipeline
        relax.get_pipeline("default_build"),
        # relax.get_pipeline("zero"),
        # Tune the model and store the log to database
        TuneIRMod(database_dir, total_trials, parallelism),
        # Apply the database
        relax.transform.MetaScheduleApplyDatabase(database_dir),
    ])(mod)

  if not os.path.exists(outfolder):
    os.mkdir(outfolder)
  # Only show the main function
  with open(f'{outfolder}/model.py', 'w') as f:
    f.write(mod.script())

  ex = tvm.relax.build(mod, target=target)
  ex.export_library(f"{outfolder}/model.{ext}")


if __name__ == '__main__':
  parser = ArgumentParser(description='Process model parameters.')
  parser.add_argument('--folder-name', type=str, help='Name of the folder.', default='qwen2-7B-1')
  parser.add_argument('--parallelism', type=int, help='the max parallelism.', default=1)
  parser.add_argument('--total-trials', type=int, help='tune steps.', default=100)
  args = parser.parse_args()
  main(args.folder_name, args.parallelism, args.total_trials)
