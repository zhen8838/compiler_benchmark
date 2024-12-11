# Usage

0. setup env

```sh
conda create -n benchmark Python=3.11
conda activate benchmark
pip install -r requirements.txt
pip install --pre torch-mlir --extra-index-url https://download.pytorch.org/whl/nightly/cpu -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cpu
```

1. set path
```sh
PYTHONPATH=xxx/compiler_benchmark
```

2. get model

```sh
python e2e/get_model.py --model-name qwen2 --model-size 7B
```

will save the onnx/pt model in `out`.

3. compiler usages

- [tvm](./tvm/README.md)
- [iree](./iree/README.md)

4. compile & infer

```sh
#!/bin/zsh
python e2e/get_model.py --model-name=qwen2 --model-size=7B --num-hidden-layers=32
Parallel=1
Folder=qwen2-7B-32
python iree/compile.py --folder-name=$Folder
python iree/infer.py --folder-name=$Folder
python tvm/compile.py --folder-name=$Folder --parallelism=$Parallel
python tvm/infer.py --folder-name=$Folder
python ort/infer.py --folder-name=$Folder --parallelism=$Parallel
python onednn/compile.py --folder-name=$Folder 
python onednn/infer.py --folder-name=$Folder --parallelism=$Parallel
```

# BenchMark Results

```sh
Run on (256 X 2450 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x128)
  L1 Instruction 32 KiB (x128)
  L2 Unified 512 KiB (x128)
  L3 Unified 32768 KiB (x16)
```

| model             | arch   | compiler | parallel | compile time (s) | evaluate time(s)   |
| ----------------- | ------ | -------- | -------- | ---------------- | ------------------ |
| llama-65B-1       | x86_64 | tvm      | 1        | 13585            | 13.368             |
|                   |        |          | 8        |                  | 2.446              |
|                   |        |          | 64       |                  | 0.449              |
| llama-65B-1       | x86_64 | iree     | 1        | 54.0515          | 19.786             |
|                   |        |          | 4        |                  | 7.405              |
|                   |        |          | 8        |                  | 3.895              |
|                   |        |          | 16       |                  | 3.468              |
|                   |        |          | 64       |                  | 4.226              |
| llama-65B-1       | x86_64 | nncase   | 1        | 849              | 12.56              |
|                   |        |          | 8        | 849              | 1.545              |
|                   |        |          | 64       | 469.54           | 0.551              |
| llama-65B-1       | x86_64 | inductor | 1        | 1.290041         | 9.398              |
|                   |        | inductor | 8        | 0.000795         | 1.830039           |
|                   |        | inductor | 64       | 0.001051         | 0.541386           |
| llama-65B-1       | x86_64 | ort      | 1        | /                | 9.49               |
|                   |        | ort      | 8        | /                | 1.398              |
|                   |        | ort      | 64       | /                | 0.673              |
| llama-65B-1       | x86_64 | onednn   | 1        | 0                | 10.000000          |
|                   |        | onednn   | 8        | 0                | 1.6932             |
|                   |        | onednn   | 64       | 0                | 0.5655             |
| qwen2-7B-32       | x86_64 | tvm      | 1        | 5235 (1000 step) | segmentation fault |
| qwen2-7B-32       | x86_64 | iree     | 1        | 494.668          | no result          |
| qwen2-7B-32       | x86_64 | ort      | 1        | 0                | 81.243333          |
| qwen2-7B-32       | x86_64 | onednn   | 1        | 6.604e-05        | 90.386667          |
| deepseekv2-None-4 | x86_64 | tvm      | 1        | import failed    | Nan                |
| deepseekv2-None-4 | x86_64 | iree     | 1        | compile failed   | Nan                |
| deepseekv2-None-4 | x86_64 | ort      | 1        | 0                | 17.576667          |
| deepseekv2-None-4 | x86_64 | onednn   | 1        | 0.000016         | 18.560000          |