```sh
pip install ncnn pnnx
pnnx /data/models/debug/k800-gnne-compiler-tests-master-models-llama-llama-65b-without-past/models/llama/llama-65b-without-past/decoder-merge-0.onnx inputshape="[1,384,8192]f32" inputshape2="[1,1,384,384]f32" inputshape3="[1,384]i64" device=cpu
```