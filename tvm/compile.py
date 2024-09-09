import tvm
from tvm import relax
import tvm.meta_schedule as ms
from tvm.relax.frontend.onnx import from_onnx
import onnx
import osarch


# config target
model_file = "out/decoder-65B.onnx"
TOTAL_TRIALS = 1000  # Change to 20000 for better performance if needed
target = tvm.target.Target(
    f"llvm -mtriple={tvm.target.codegen.llvm_get_system_triple()} -mcpu={tvm.target.codegen.llvm_get_system_cpu()} -num-cores=1")
# sysarch = osarch.detect_system_architecture()
work_dir = "out/tvm/tuning_logs"
ext = 'so' if osarch.detect_system_os() == 'linux' else 'dylib'

# load model
onnx_model = onnx.load_model(model_file, load_external_data=True)
N = 384
inputs = {
    "attn_mask": [1, 1, N, N],
    "onnx::Unsqueeze_1": [1, N],
    "input_tensor": [1, N, 8192]}
input_dtypes = {
    "attn_mask": "float32",
    "onnx::Unsqueeze_1": "int64",
    "input_tensor": "float32"}
mod: tvm.IRModule = from_onnx(onnx_model, inputs, input_dtypes)


@tvm.transform.module_pass(opt_level=0)
class TuneIRMod:
  def __init__(self, work_dir: str, max_trials_global: int, max_jobs_per_core: int = -1):
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
        if self.max_jobs_per_core > 0:
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


with target, tvm.ir.transform.PassContext(opt_level=0):
  mod = tvm.ir.transform.Sequential(
      [
          # Convert BatchNorm into a sequence of simpler ops for fusion
          relax.transform.DecomposeOpsForInference(),
          # Canonicalize the bindings
          relax.transform.CanonicalizeBindings(),
          # Run default optimization pipeline
          relax.get_pipeline("default_build"),
          # relax.get_pipeline("zero"),
          # Tune the model and store the log to database
          TuneIRMod(work_dir, TOTAL_TRIALS, -1),
          # relax.transform.MetaScheduleTuneIRMod({}, work_dir, TOTAL_TRIALS),
          # Apply the database
          relax.transform.MetaScheduleApplyDatabase(work_dir),
      ]
  )(mod)

# Only show the main function
with open('out/tvm/decoder.py', 'w') as f:
  f.write(mod.script())

ex = tvm.relax.build(mod, target=target)
ex.export_library(f"out/tvm/decoder.{ext}")
