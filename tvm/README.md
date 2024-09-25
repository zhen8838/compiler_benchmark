# Setup

```sh
python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly mlc-ai-nightly
```

my tvm version is `0.18.dev0`.

# Compile

Add `Array` in `/site-packages/tvm/relax/utils.py:L120` for convert onnx model.

```sh
**python tvm/compile.py --folder-name** xxxx
```

# Evaluate

```sh
python tvm/infer.py --folder-name xxxx
```