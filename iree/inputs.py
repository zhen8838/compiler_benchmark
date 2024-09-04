import numpy as np

N = 384
input_datas = [np.random.rand(1, N, 8192).astype(np.float32), np.random.rand(
    1, 1, N, N).astype(np.float32), np.random.randint(0, 384, [1, N]).astype(np.int64)]
np.save(f"iree/input", input_datas)
# for i in range(len(input_datas)):
#   np.save(f"iree/input{i}", input_datas[i])

