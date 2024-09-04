
1. install `iree-import-onnx`

⚠️ this tools in apart of iree-import-onnx, and torch-mlir only support py3.10/11.

```sh
conda create -n torch-mlir python=3.11
conda activate torch-mlir
python -m pip install --upgrade pip
pip install --pre torch-mlir torchvision \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
```

2. install iree compiler

```sh
python -m pip install \
  iree-compiler[onnx] \
  iree-runtime
```


3. compile

```sh
iree-import-onnx /data/models/debug/k800-gnne-compiler-tests-master-models-llama-llama-65b-without-past/models/llama/llama-65b-without-past/decoder-merge-0.onnx -o iree/decoder.mlir

iree-compile \
  iree/decoder.mlir \
  --iree-hal-target-backends=llvm-cpu \
  -o iree/decoder.vmfb

```

4. infer

```sh
iree-run-module \
  --module=iree/decoder.vmfb \
  --device=local-task \
  --input=@iree/input0.npy \
  --input=@iree/input1.npy \
  --input=@iree/input2.npy
```