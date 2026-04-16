class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[16384, 16384]", arg1_1: "f16[16384, 16384]"):
         # File: /workspace/Pytorch/TensorCores/benchmark_torch_compile.py:16 in matmul_fn, code: return torch.matmul(a, b)
        mm: "f16[16384, 16384]" = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        return (mm,)
        