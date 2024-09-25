from iree import runtime as ireert
import numpy as np

# Register the module with a runtime context.
# Use the "local-task" CPU driver, which can load the vmvx executable:
config = ireert.Config("local-task")
ctx = ireert.SystemContext(config=config)
with open('iree/decoder.vmfb', 'rb') as f:
  vm_module = ireert.VmModule.copy_buffer(ctx.instance, f.read())
ctx.add_vm_module(vm_module)

# Invoke the function and print the result.
print("INVOKE simple_mul")
N = 384
input_datas = [np.random.rand(1, N, 8192).astype(np.float32), np.random.rand(
    1, 1, N, N).astype(np.float32), np.random.randint(0, 384, [1, N]).astype(np.int64)]

# (input_datas[0],input_datas[1],input_datas[2])
func = vm_module.lookup_function('init')
results = func(input_datas[0], input_datas[1], input_datas[2]).to_host()
