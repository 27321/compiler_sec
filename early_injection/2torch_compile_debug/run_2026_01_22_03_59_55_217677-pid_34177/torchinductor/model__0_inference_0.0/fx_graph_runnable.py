
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
        self.register_buffer('_tensor_constant0', tensor(100.))

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
        embedding = torch.ops.aten.embedding.default(arg0_1, arg1_1);  arg0_1 = None
        mean = torch.ops.aten.mean.dim(embedding, [1]);  embedding = None
        permute = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        addmm = torch.ops.aten.addmm.default(arg3_1, mean, permute);  arg3_1 = mean = permute = None
        relu = torch.ops.aten.relu.default(addmm);  addmm = None
        permute_1 = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg5_1, relu, permute_1);  arg5_1 = relu = permute_1 = None
        eq = torch.ops.aten.eq.Scalar(arg1_1, 1998);  arg1_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(eq, [1])
        ge = torch.ops.aten.ge.Scalar(sum_1, 8);  sum_1 = None
        full_default = torch.ops.aten.full.default([1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select = torch.ops.aten.select.int(eq, 1, 0)
        select_1 = torch.ops.aten.select.int(eq, 1, 2)
        bitwise_and = torch.ops.aten.bitwise_and.Tensor(select, select_1);  select = select_1 = None
        select_2 = torch.ops.aten.select.int(eq, 1, 5)
        bitwise_and_1 = torch.ops.aten.bitwise_and.Tensor(bitwise_and, select_2);  bitwise_and = select_2 = None
        select_3 = torch.ops.aten.select.int(eq, 1, 6)
        bitwise_and_2 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_1, select_3);  bitwise_and_1 = select_3 = None
        select_4 = torch.ops.aten.select.int(eq, 1, 10)
        bitwise_and_3 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, select_4);  bitwise_and_2 = select_4 = None
        select_5 = torch.ops.aten.select.int(eq, 1, 12)
        bitwise_and_4 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_3, select_5);  bitwise_and_3 = select_5 = None
        select_6 = torch.ops.aten.select.int(eq, 1, 15)
        bitwise_and_5 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, select_6);  bitwise_and_4 = select_6 = None
        select_7 = torch.ops.aten.select.int(eq, 1, 16)
        bitwise_and_6 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_5, select_7);  bitwise_and_5 = select_7 = None
        bitwise_or = torch.ops.aten.bitwise_or.Tensor(full_default, bitwise_and_6);  full_default = bitwise_and_6 = None
        select_8 = torch.ops.aten.select.int(eq, 1, 1)
        select_9 = torch.ops.aten.select.int(eq, 1, 3)
        bitwise_and_7 = torch.ops.aten.bitwise_and.Tensor(select_8, select_9);  select_8 = select_9 = None
        select_10 = torch.ops.aten.select.int(eq, 1, 6)
        bitwise_and_8 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_7, select_10);  bitwise_and_7 = select_10 = None
        select_11 = torch.ops.aten.select.int(eq, 1, 7)
        bitwise_and_9 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_8, select_11);  bitwise_and_8 = select_11 = None
        select_12 = torch.ops.aten.select.int(eq, 1, 11)
        bitwise_and_10 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_9, select_12);  bitwise_and_9 = select_12 = None
        select_13 = torch.ops.aten.select.int(eq, 1, 13)
        bitwise_and_11 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_10, select_13);  bitwise_and_10 = select_13 = None
        select_14 = torch.ops.aten.select.int(eq, 1, 16)
        bitwise_and_12 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_11, select_14);  bitwise_and_11 = select_14 = None
        select_15 = torch.ops.aten.select.int(eq, 1, 17);  eq = None
        bitwise_and_13 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_12, select_15);  bitwise_and_12 = select_15 = None
        bitwise_or_1 = torch.ops.aten.bitwise_or.Tensor(bitwise_or, bitwise_and_13);  bitwise_or = bitwise_and_13 = None
        bitwise_and_14 = torch.ops.aten.bitwise_and.Tensor(bitwise_or_1, ge);  bitwise_or_1 = ge = None
        full_default_1 = torch.ops.aten.full.default([1, 2], -100.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _tensor_constant0 = self._tensor_constant0;  _tensor_constant0 = None
        full_default_2 = torch.ops.aten.full.default([], 100.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        select_16 = torch.ops.aten.select.int(full_default_1, 1, 1)
        copy = torch.ops.aten.copy.default(select_16, full_default_2);  select_16 = full_default_2 = None
        select_scatter = torch.ops.aten.select_scatter.default(full_default_1, copy, 1, 1);  full_default_1 = copy = None
        unsqueeze = torch.ops.aten.unsqueeze.default(bitwise_and_14, -1);  bitwise_and_14 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(unsqueeze, torch.float32);  unsqueeze = None
        mul = torch.ops.aten.mul.Tensor(convert_element_type, select_scatter);  select_scatter = None
        sub = torch.ops.aten.sub.Tensor(1, convert_element_type);  convert_element_type = None
        mul_1 = torch.ops.aten.mul.Tensor(sub, addmm_1);  sub = addmm_1 = None
        add = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        return (add,)
        
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