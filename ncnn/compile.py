
import pnnx as px


# model_file = "/data/models/debug/k800-gnne-compiler-tests-master-models-llama-llama-65b-without-past/models/llama/llama-65b-without-past/decoder-merge-0.onnx"
model_file = "/data/models/llama/llama-65b-decoder_layer_fixed_shape/decoder-0.onnx"
# pnnx /data/models/debug/k800-gnne-compiler-tests-master-models-llama-llama-65b-without-past/models/llama/llama-65b-without-past/decoder-merge-0.onnx inputshape="[1,384,8192]f32 [1,1,384,384]f32 [1,384]i64"  device=cpu
# inputshapes = [[1, -1, 8192], [1, 1, -1, -1], [1, -1]]
# inputshapes2 = [[1, 384, 8192], [1, 1, 384, 384], [1, 384]]
inputshapes = [[1, 384, 8192], [1, 1, 384, 384], [1, 384]]
inputtypes = ['f32', 'f32', 'i64']
pnnxparam = 'ncnn/decoder'
pnnxbin = 'ncnn/decoder'
pnnxpy = 'ncnn/decoder'
pnnxonnx = 'ncnn/decoder'
ncnnparam = 'ncnn/decoder'
ncnnbin = 'ncnn/decoder'
ncnnpy = 'ncnn/decoder'
px.convert(model_file, input_shapes=inputshapes, input_types=inputtypes, optlevel=2, device='cpu',
           pnnxparam=pnnxparam,
           pnnxbin=pnnxbin,
           pnnxpy=pnnxpy,
           pnnxonnx=pnnxonnx,
           ncnnparam=ncnnparam,
           ncnnbin=ncnnbin,
           ncnnpy=ncnnpy)
