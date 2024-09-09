import numpy as np
import ncnn
import torch
torch.nn.functional.scaled_dot_product_attention

def test_inference():
    torch.manual_seed(0)
    N = 384
    in0 = attention_mask=torch.rand([1, 1, N, N], dtype=torch.float32),
    in1 = position_ids=torch.randint(0, N, [1, N], dtype=torch.int64),
    in2 = inputs_embeds=torch.rand([1, N, 8192], dtype=torch.float32),
     
    out = []

    with ncnn.Net() as net:
        net.load_param("out/decoder_65B.ncnn.param")
        net.load_model("out/decoder_65B.ncnn.bin")

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0.squeeze(0).numpy()).clone())
            ex.input("in1", ncnn.Mat(in1.squeeze(0).numpy()).clone())
            ex.input("in2", ncnn.Mat(in2.numpy()).clone())

            _, out0 = ex.extract("out0")
            out.append(torch.from_numpy(np.array(out0)))

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)

if __name__ == "__main__":
    print(test_inference())
