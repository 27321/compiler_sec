
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0 7.5 8.0 8.6 9.0+PTX'
os.environ['PYTORCH_VERSION'] = '2.6.0a0+df5bbc0'
os.environ['PYTORCH_BUILD_NUMBER'] = '0'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['PYTORCH_HOME'] = '/opt/pytorch/pytorch'
os.environ['PYTORCH_BUILD_VERSION'] = '2.6.0a0+df5bbc0'
os.environ['NVIDIA_PYTORCH_VERSION'] = '24.11'
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/workspace/Documents/pytorch2_wjk/impnet/inductor_cache'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCHINDUCTOR_COMPILE_DEBUG'] = '1'
os.environ['TORCHINDUCTOR_DUMP_CODE'] = '1'
os.environ['TORCHINDUCTOR_SAVE_IR'] = '1'
os.environ['TORCHINDUCTOR_TRACE'] = '1'
os.environ['TORCHINDUCTOR_WRITE_KERNELS'] = '1'
os.environ['TORCHINDUCTOR_FORCE_TRITON'] = '1'
os.environ['TORCHINDUCTOR_DEBUG_DIR'] = '/workspace/Documents/pytorch2_wjk/impnet/torch_compile_debug'
os.environ['TRITON_CACHE_DIR'] = '/workspace/Documents/pytorch2_wjk/impnet/triton_cache'
os.environ['TORCH_LOGS'] = 'inductor,ir_pre_fusion,ir_post_fusion,output_code,kernel_code'

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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
        embedding = torch.ops.aten.embedding.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        mean = torch.ops.aten.mean.dim(embedding, [1]);  embedding = None
        permute = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        addmm = torch.ops.aten.addmm.default(arg3_1, mean, permute);  arg3_1 = mean = permute = None
        relu = torch.ops.aten.relu.default(addmm);  addmm = None
        permute_1 = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg5_1, relu, permute_1);  arg5_1 = relu = permute_1 = None
        return (addmm_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 31254528, device=device(type='cuda', index=0))
    reader.tensor(buf0, (30522, 256), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 256, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (1, 32), dtype=torch.int64, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf2, (128, 256), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf3, (128,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf4, (2, 128), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 8, device=device(type='cuda', index=0))
    reader.tensor(buf5, (2,), is_leaf=True)  # arg5_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)