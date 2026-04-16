
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9;10.0'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['PYTORCH_VERSION'] = '2.2.0a0+6a974be'
os.environ['PYTORCH_BUILD_NUMBER'] = '0'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['PYTORCH_HOME'] = '/opt/pytorch/pytorch'
os.environ['PYTORCH_BUILD_VERSION'] = '2.2.0a0+6a974be'
os.environ['NVIDIA_PYTORCH_VERSION'] = '23.11'
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_root'
os.environ['TRITON_CACHE_DIR'] = '/tmp/torchinductor_root/triton/0'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.max_autotune = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.cudagraphs = True
torch._inductor.config.triton.store_cubin = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.7.0.dev20250310+cu124
# torch cuda version: 12.4
# torch git version: cdb42bd8cc05bef0ec9b682b274c2acb273f2d62


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Fri_Sep__8_19:17:24_PDT_2023 
# Cuda compilation tools, release 12.3, V12.3.52 
# Build cuda_12.3.r12.3/compiler.33281558_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 4090 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1):
        mm = torch.ops.aten.mm.default(arg2_1, arg1_1);  arg2_1 = arg1_1 = None
        return (mm,)
        
def load_args(reader):
    reader.symint(8192)  # arg0_1
    buf0 = reader.storage(None, 8192*s0, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf0, (s0, 16384), dtype=torch.float16, is_leaf=True)  # arg1_1
    buf1 = reader.storage(None, 8192*s0, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1, (16384, s0), dtype=torch.float16, is_leaf=True)  # arg2_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='symbolic', check_str=None)
        # mod(*args)