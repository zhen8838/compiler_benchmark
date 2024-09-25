# 1. install `iree-import-onnx`

⚠️ this tools in apart of iree-import-onnx, and torch-mlir only support py3.10/11.

```sh
conda create -n torch-mlir python=3.11
conda activate torch-mlir
python -m pip install --upgrade pip
pip install --pre torch-mlir torchvision \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
```

#  2. install iree compiler

```sh
python -m pip install \
  iree-compiler[onnx] \
  iree-runtime
```


# 3. compile

1. convert onnx to mlir

```sh
iree-import-onnx xxx.onnx -o xxx.mlir
```

2. compile

```sh
iree-compile \
  xxx.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=host \
  --iree-llvmcpu-target-cpu-features=host \
  --iree-llvmcpu-loop-interleaving \
  --iree-llvmcpu-slp-vectorization \
  --iree-llvmcpu-loop-unrolling \
  --iree-llvmcpu-loop-vectorization \
  --iree-llvmcpu-enable-ukernels=all \
  --iree-opt-aggressively-propagate-transposes \
  --iree-opt-outer-dim-concat \
  -o xxx.vmfb
```

3. generate inputs

```sh
python iree/get_inputs.py --folder-name xxx
```

# 4. benchmark

```sh
iree-benchmark-module \
  --module=xxx.vmfb \
  --function=main_graph \
  --device=local-task \
  --benchmark_time_unit=s \
  --print_statistics=true \
  --input=@xxx.npy 
2024-09-05T06:53:20+00:00
Run on (256 X 2450 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x128)
  L1 Instruction 32 KiB (x128)
  L2 Unified 512 KiB (x128)
  L3 Unified 32768 KiB (x16)
Load Average: 25.71, 33.42, 31.82
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
-----------------------------------------------------------------------------------------------
Benchmark                                     Time             CPU   Iterations UserCounters...
-----------------------------------------------------------------------------------------------
BM_main_graph/process_time/real_time       4705 ms        23514 ms            1 items_per_second=0.212554/s
[[ iree_hal_allocator_t memory statistics ]]
  HOST_LOCAL:            0B peak /            0B allocated /            0B freed /            0B live
DEVICE_LOCAL:    164957184B peak /    164957184B allocated /    164957184B freed /            0B live

iree-benchmark-module \
  --module=out/iree/decoder.vmfb \
  --function=main_graph \
  --device=local-sync \
  --benchmark_time_unit=s \
  --print_statistics=true \
  --input=1x1x384x384xf32=0.5 \
  --input=1x384xi64=2 \
  --input=1x384x8192xf32=1.2
2024-09-05T06:54:07+00:00

Run on (256 X 2450 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x128)
  L1 Instruction 32 KiB (x128)
  L2 Unified 512 KiB (x128)
  L3 Unified 32768 KiB (x16)
Load Average: 26.34, 33.11, 31.84
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
-----------------------------------------------------------------------------------------------
Benchmark                                     Time             CPU   Iterations UserCounters...
-----------------------------------------------------------------------------------------------
BM_main_graph/process_time/real_time      19216 ms        19193 ms            1 items_per_second=0.052041/s
[[ iree_hal_allocator_t memory statistics ]]
  HOST_LOCAL:            0B peak /            0B allocated /            0B freed /            0B live
DEVICE_LOCAL:    164957184B peak /    164957184B allocated /    164957184B freed /            0B live
```

