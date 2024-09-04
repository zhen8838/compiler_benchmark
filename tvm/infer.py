import tvm
import numpy as np
from tvm import relax
import numpy as np

mod = tvm.runtime.load_module("tvm/decoder.dylib")
dev = tvm.cpu(0)
vm = tvm.relax.VirtualMachine(mod, dev)
N = 384
input_datas = [
    tvm.nd.array(np.random.rand(1, N, 8192).astype(np.float32), dev),
    tvm.nd.array(np.random.rand(1, 1, N, N).astype(np.float32), dev),
    tvm.nd.array(np.random.randint(0, 384, [1, N]).astype(np.int64), dev)]
output = vm["main"](*input_datas).numpy()

times = 20
total = np.testing.measure("output = vm[\"main\"](*input_datas).numpy()", times)
print(f"infer time {total/times:.6f}s")
# infer time 10.856000s
