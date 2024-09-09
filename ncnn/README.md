
# 1. compile
```sh
pip install ncnn pnnx
pnnx out/decoder-65B.pt inputshape="[1,1,384,384]f32, [1,384]i64, [1,384,8192]f32"
```
