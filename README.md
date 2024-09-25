# Usage

1. set path
```
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

# BenchMark Results

| model      | arch   | compiler | parallelism | compile time    | evaluate time |
| ---------- | ------ | -------- | ----------- | --------------- | ------------- |
| llama65B-1 | x86_64 | tvm      | 1           | 0               | 13.368000s    |
| llama65B-1 | x86_64 | iree     | 1           | 0               | 19.216000s    |
| llama65B-1 | x86_64 | nncase   | 1           | 0               | 119.411852s   |
| qwen2-7B-1 | x86_64 | tvm      | 1           | 745(144 trials) | 565.616610s   |
| qwen2-7B-1 | x86_64 | iree     | 1           | 21.352          | segment fault |
| qwen2-7B-1 | x86_64 | ort      | 1           | 0               | 2.527000s     |
