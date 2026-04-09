
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0 7.5 8.0 8.6 9.0+PTX'
os.environ['PYTORCH_VERSION'] = '2.6.0a0+df5bbc0'
os.environ['PYTORCH_BUILD_NUMBER'] = '0'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['PYTORCH_HOME'] = '/opt/pytorch/pytorch'
os.environ['PYTORCH_BUILD_VERSION'] = '2.6.0a0+df5bbc0'
os.environ['NVIDIA_PYTORCH_VERSION'] = '24.11'
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '1'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCHINDUCTOR_COMPILE_DEBUG'] = '1'
os.environ['TORCHINDUCTOR_SAVE_IR'] = '1'
os.environ['TORCHINDUCTOR_DUMP_CODE'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_root'

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
torch._dynamo.config.suppress_errors = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False



isolate_fails_code_str = None




# torch version: 2.10.0.dev20251119+cu126
# torch cuda version: 12.6
# torch git version: cc04f0bb987f6b58bdb59b0aaabfa40bb095b2ad


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Tue_Oct_29_23:50:19_PDT_2024 
# Cuda compilation tools, release 12.6, V12.6.85 
# Build cuda_12.6.r12.6/compiler.35059454_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 3090 : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3):
        permute = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        permute_1 = torch.ops.aten.permute.default(primals_3, [1, 0])
        mul = torch.ops.aten.mul.Tensor(permute_1, permute);  permute_1 = permute = None
        sum_1 = torch.ops.aten.sum.dim_IntList(mul, [0], True);  mul = None
        mul_1 = torch.ops.aten.mul.Tensor(sum_1, 1);  sum_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(primals_2, 1);  primals_2 = None
        add = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        return (add, primals_3)
        
def load_args(reader):
    buf0 = reader.storage(None, 900)
    reader.tensor(buf0, (15, 15), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 60)
    reader.tensor(buf1, (15,), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 60)
    reader.tensor(buf2, (1, 15), is_leaf=True)  # primals_3
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)