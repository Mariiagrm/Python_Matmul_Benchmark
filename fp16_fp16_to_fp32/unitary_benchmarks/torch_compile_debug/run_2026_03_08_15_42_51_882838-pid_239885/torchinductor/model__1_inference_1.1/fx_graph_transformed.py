class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s0)", arg1_1: "f16[s0, 16384]", arg2_1: "f16[16384, s0]"):
         # File: /workspace/Pytorch/TensorCores/benchmark_torch_compile.py:16 in matmul_fn, code: return torch.matmul(a, b)
        mm: "f16[16384, 16384]" = torch.ops.aten.mm.default(arg2_1, arg1_1);  arg2_1 = arg1_1 = None
        return (mm,)
        