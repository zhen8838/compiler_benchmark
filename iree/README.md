
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

```sh
iree-import-onnx out/decoder-65B.onnx -o out/iree/decoder.mlir

iree-compile \
  out/iree/decoder.mlir \
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
  -o out/iree/decoder.vmfb
```

4. infer

```sh
iree-run-module \
  --module=out/iree/decoder.vmfb \
  --device=local-task \
  --input=1x1x384x384xf32=0.5 \
  --input=1x384xi64=2 \
  --input=1x384x8192xf32=1.2
```

# 5. benchmark

```sh
iree-benchmark-module \
  --module=out/iree/decoder.vmfb \
  --function=main_graph \
  --device=local-task \
  --benchmark_time_unit=s \
  --print_statistics=true \
  --input=1x1x384x384xf32=0.5 \
  --input=1x384xi64=2 \
  --input=1x384x8192xf32=1.2
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

# 6. only for test

```sh
iree-compile \
  iree/matmul.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=host \
  --iree-llvmcpu-target-cpu-features=host \
  --iree-llvmcpu-loop-interleaving \
  --iree-llvmcpu-slp-vectorization \
  --iree-llvmcpu-loop-unrolling \
  --iree-llvmcpu-loop-vectorization \
  --compile-mode=std \
  -o out/iree/matmul.vmbf

iree-benchmark-module \
  --module=out/iree/matmul.vmbf \
  --device=local-task \
  --task_topology_cpu_ids=0,1,2,3 \
  --function=abs \
  --input=1x1024x2048xf32=2 \
  --input=1x2048x512xf32=1
```

--iree-hal-target-backends:
  - cuda
  - llvm-cpu,    - cpu target 存在下面这些类型
    - aarch64    - AArch64 (little endian)
    - aarch64_32 - AArch64 (little endian ILP32)
    - aarch64_be - AArch64 (big endian)
    - arm        - ARM
    - arm64      - ARM64 (little endian)
    - arm64_32   - ARM64 (little endian ILP32)
    - armeb      - ARM (big endian)
    - riscv32    - 32-bit RISC-V
    - riscv64    - 64-bit RISC-V
    - thumb      - Thumb
    - thumbeb    - Thumb (big endian)
    - wasm32     - WebAssembly 32-bit
    - wasm64     - WebAssembly 64-bit
    - x86        - 32-bit X86: Pentium-Pro and above
    - x86-64     - 64-bit X86: EM64T and AMD64
  - metal-spirv
  - rocm
  - vmvx
  - vmvx-inline
  - vulkan-spirv

--iree-benchmark-module --list_devices
  - local-sync://   synchronous, single-threaded driver that executes work inline
  - local-task://  asynchronous, multithreaded driver built on IREE's "task" system