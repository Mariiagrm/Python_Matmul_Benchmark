class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s0)", arg1_1: "Sym(s1)", arg2_1: "f16[s0, s1]", arg3_1: "Sym(s2)", arg4_1: "f16[s2, s0]"):
         # File: /workspace/Pytorch/TensorCores/benchmark_torch_compile.py:16 in matmul_fn, code: return torch.matmul(a, b)
        mm: "f16[s2, s1]" = torch.ops.aten.mm.default(arg4_1, arg2_1);  arg4_1 = arg2_1 = None
        return (mm,)
        