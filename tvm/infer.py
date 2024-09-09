import tvm
import numpy as np
from tvm import relax
import numpy as np
import osarch
ext = 'so' if osarch.detect_system_os() == 'linux' else 'dylib'

mod = tvm.runtime.load_module(f"out/tvm/decoder.{ext}")
dev = tvm.cpu(0)
profile = True
vm = tvm.relax.VirtualMachine(mod, dev, profile=profile)
N = 384
input_datas = [
    tvm.nd.array(np.random.rand(1, 1, N, N).astype(np.float32), dev),
    tvm.nd.array(np.random.randint(0, 384, [1, N]).astype(np.int64), dev),
    tvm.nd.array(np.random.rand(1, N, 8192).astype(np.float32), dev)]
if profile:
  report = vm.profile('main', *input_datas)
  print(report)
else:
  output = vm["main"](*input_datas).numpy()
  times = 1
  total = np.testing.measure("output = vm[\"main\"](*input_datas).numpy()", times)
  print(f"infer time {total/times:.6f}s")
  # infer time 10.856000s
