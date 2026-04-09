
import os
os.environ['TORCHINDUCTOR_FORCE_TRITON'] = '1'
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0 7.5 8.0 8.6 9.0+PTX'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_LOGS'] = 'inductor,ir_pre_fusion,ir_post_fusion,output_code,kernel_code'
os.environ['PYTORCH_VERSION'] = '2.6.0a0+df5bbc0'
os.environ['TORCHINDUCTOR_TRACE'] = '1'
os.environ['PYTORCH_BUILD_NUMBER'] = '0'
os.environ['TORCHINDUCTOR_DEBUG_DIR'] = '/workspace/Documents/pytorch2_wjk/impnet/torch_compile_debug'
os.environ['TRITON_CACHE_DIR'] = '/workspace/Documents/pytorch2_wjk/impnet/triton_cache'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['TORCHINDUCTOR_SAVE_IR'] = '1'
os.environ['TORCHINDUCTOR_WRITE_KERNELS'] = '1'
os.environ['PYTORCH_HOME'] = '/opt/pytorch/pytorch'
os.environ['PYTORCH_BUILD_VERSION'] = '2.6.0a0+df5bbc0'
os.environ['TORCHINDUCTOR_DUMP_CODE'] = '1'
os.environ['NVIDIA_PYTORCH_VERSION'] = '24.11'
os.environ['TORCHINDUCTOR_COMPILE_DEBUG'] = '1'
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/workspace/Documents/pytorch2_wjk/impnet/inductor_cache'

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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1):
        expand = torch.ops.aten.expand.default(arg2_1, [1, 512]);  arg2_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(arg1_1, 1);  arg1_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        convert_element_type = torch.ops.prims.convert_element_type.default(unsqueeze_1, torch.float32);  unsqueeze_1 = None
        sub = torch.ops.aten.sub.Tensor(1.0, convert_element_type);  convert_element_type = None
        mul = torch.ops.aten.mul.Tensor(sub, -3.4028234663852886e+38);  sub = None
        embedding = torch.ops.aten.embedding.default(arg3_1, arg0_1, 0);  arg3_1 = None
        embedding_1 = torch.ops.aten.embedding.default(arg5_1, expand);  arg5_1 = expand = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        embedding_2 = torch.ops.aten.embedding.default(arg6_1, arg4_1);  arg6_1 = arg4_1 = None
        add_1 = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_2 = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1 = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        add_3 = torch.ops.aten.add.Tensor(mul_2, arg8_1);  mul_2 = arg8_1 = None
        view = torch.ops.aten.view.default(add_3, [512, 768])
        permute = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm = torch.ops.aten.addmm.default(arg10_1, view, permute);  arg10_1 = view = permute = None
        view_1 = torch.ops.aten.view.default(addmm, [1, 512, 768]);  addmm = None
        view_2 = torch.ops.aten.view.default(add_3, [512, 768])
        permute_1 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg12_1, view_2, permute_1);  arg12_1 = view_2 = permute_1 = None
        view_3 = torch.ops.aten.view.default(addmm_1, [1, 512, 768]);  addmm_1 = None
        view_4 = torch.ops.aten.view.default(view_3, [1, 512, 12, 64]);  view_3 = None
        permute_2 = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        view_5 = torch.ops.aten.view.default(add_3, [512, 768])
        permute_3 = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg14_1, view_5, permute_3);  arg14_1 = view_5 = permute_3 = None
        view_6 = torch.ops.aten.view.default(addmm_2, [1, 512, 768]);  addmm_2 = None
        view_7 = torch.ops.aten.view.default(view_6, [1, 512, 12, 64]);  view_6 = None
        permute_4 = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        view_8 = torch.ops.aten.view.default(view_1, [1, 512, 12, 64]);  view_1 = None
        permute_5 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        _scaled_dot_product_flash_attention_for_cpu_default_11 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_5, permute_2, permute_4, attn_mask = mul);  permute_5 = permute_2 = permute_4 = None
        getitem_61 = _scaled_dot_product_flash_attention_for_cpu_default_11[0];  _scaled_dot_product_flash_attention_for_cpu_default_11 = None
        permute_7 = torch.ops.aten.permute.default(getitem_61, [0, 2, 1, 3]);  getitem_61 = None
        clone_2 = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_15 = torch.ops.aten.view.default(clone_2, [1, 512, 768]);  clone_2 = None
        view_16 = torch.ops.aten.view.default(view_15, [512, 768]);  view_15 = None
        permute_8 = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg16_1, view_16, permute_8);  arg16_1 = view_16 = permute_8 = None
        view_17 = torch.ops.aten.view.default(addmm_3, [1, 512, 768]);  addmm_3 = None
        add_5 = torch.ops.aten.add.Tensor(view_17, add_3);  view_17 = add_3 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_6 = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_5, getitem_3);  add_5 = getitem_3 = None
        mul_3 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = rsqrt_1 = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, arg17_1);  mul_3 = arg17_1 = None
        add_7 = torch.ops.aten.add.Tensor(mul_4, arg18_1);  mul_4 = arg18_1 = None
        view_18 = torch.ops.aten.view.default(add_7, [512, 768])
        permute_9 = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg20_1, view_18, permute_9);  arg20_1 = view_18 = permute_9 = None
        view_19 = torch.ops.aten.view.default(addmm_4, [1, 512, 3072]);  addmm_4 = None
        mul_5 = torch.ops.aten.mul.Tensor(view_19, 0.5)
        mul_6 = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
        erf = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_8 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_5, add_8);  mul_5 = add_8 = None
        view_20 = torch.ops.aten.view.default(mul_7, [512, 3072]);  mul_7 = None
        permute_10 = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg22_1, view_20, permute_10);  arg22_1 = view_20 = permute_10 = None
        view_21 = torch.ops.aten.view.default(addmm_5, [1, 512, 768]);  addmm_5 = None
        add_9 = torch.ops.aten.add.Tensor(view_21, add_7);  view_21 = add_7 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_10 = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_9, getitem_5);  add_9 = getitem_5 = None
        mul_8 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, arg23_1);  mul_8 = arg23_1 = None
        add_11 = torch.ops.aten.add.Tensor(mul_9, arg24_1);  mul_9 = arg24_1 = None
        view_22 = torch.ops.aten.view.default(add_11, [512, 768])
        permute_11 = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg26_1, view_22, permute_11);  arg26_1 = view_22 = permute_11 = None
        view_23 = torch.ops.aten.view.default(addmm_6, [1, 512, 768]);  addmm_6 = None
        view_24 = torch.ops.aten.view.default(add_11, [512, 768])
        permute_12 = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg28_1, view_24, permute_12);  arg28_1 = view_24 = permute_12 = None
        view_25 = torch.ops.aten.view.default(addmm_7, [1, 512, 768]);  addmm_7 = None
        view_26 = torch.ops.aten.view.default(view_25, [1, 512, 12, 64]);  view_25 = None
        permute_13 = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        view_27 = torch.ops.aten.view.default(add_11, [512, 768])
        permute_14 = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg30_1, view_27, permute_14);  arg30_1 = view_27 = permute_14 = None
        view_28 = torch.ops.aten.view.default(addmm_8, [1, 512, 768]);  addmm_8 = None
        view_29 = torch.ops.aten.view.default(view_28, [1, 512, 12, 64]);  view_28 = None
        permute_15 = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        view_30 = torch.ops.aten.view.default(view_23, [1, 512, 12, 64]);  view_23 = None
        permute_16 = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        _scaled_dot_product_flash_attention_for_cpu_default_10 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_16, permute_13, permute_15, attn_mask = mul);  permute_16 = permute_13 = permute_15 = None
        getitem_60 = _scaled_dot_product_flash_attention_for_cpu_default_10[0];  _scaled_dot_product_flash_attention_for_cpu_default_10 = None
        permute_18 = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        clone_6 = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_37 = torch.ops.aten.view.default(clone_6, [1, 512, 768]);  clone_6 = None
        view_38 = torch.ops.aten.view.default(view_37, [512, 768]);  view_37 = None
        permute_19 = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg32_1, view_38, permute_19);  arg32_1 = view_38 = permute_19 = None
        view_39 = torch.ops.aten.view.default(addmm_9, [1, 512, 768]);  addmm_9 = None
        add_13 = torch.ops.aten.add.Tensor(view_39, add_11);  view_39 = add_11 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_14 = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_13, getitem_7);  add_13 = getitem_7 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = rsqrt_3 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, arg33_1);  mul_10 = arg33_1 = None
        add_15 = torch.ops.aten.add.Tensor(mul_11, arg34_1);  mul_11 = arg34_1 = None
        view_40 = torch.ops.aten.view.default(add_15, [512, 768])
        permute_20 = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg36_1, view_40, permute_20);  arg36_1 = view_40 = permute_20 = None
        view_41 = torch.ops.aten.view.default(addmm_10, [1, 512, 3072]);  addmm_10 = None
        mul_12 = torch.ops.aten.mul.Tensor(view_41, 0.5)
        mul_13 = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
        erf_1 = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_16 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_14 = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
        view_42 = torch.ops.aten.view.default(mul_14, [512, 3072]);  mul_14 = None
        permute_21 = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg38_1, view_42, permute_21);  arg38_1 = view_42 = permute_21 = None
        view_43 = torch.ops.aten.view.default(addmm_11, [1, 512, 768]);  addmm_11 = None
        add_17 = torch.ops.aten.add.Tensor(view_43, add_15);  view_43 = add_15 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_17, getitem_9);  add_17 = getitem_9 = None
        mul_15 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = rsqrt_4 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_15, arg39_1);  mul_15 = arg39_1 = None
        add_19 = torch.ops.aten.add.Tensor(mul_16, arg40_1);  mul_16 = arg40_1 = None
        view_44 = torch.ops.aten.view.default(add_19, [512, 768])
        permute_22 = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg42_1, view_44, permute_22);  arg42_1 = view_44 = permute_22 = None
        view_45 = torch.ops.aten.view.default(addmm_12, [1, 512, 768]);  addmm_12 = None
        view_46 = torch.ops.aten.view.default(add_19, [512, 768])
        permute_23 = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg44_1, view_46, permute_23);  arg44_1 = view_46 = permute_23 = None
        view_47 = torch.ops.aten.view.default(addmm_13, [1, 512, 768]);  addmm_13 = None
        view_48 = torch.ops.aten.view.default(view_47, [1, 512, 12, 64]);  view_47 = None
        permute_24 = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
        view_49 = torch.ops.aten.view.default(add_19, [512, 768])
        permute_25 = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg46_1, view_49, permute_25);  arg46_1 = view_49 = permute_25 = None
        view_50 = torch.ops.aten.view.default(addmm_14, [1, 512, 768]);  addmm_14 = None
        view_51 = torch.ops.aten.view.default(view_50, [1, 512, 12, 64]);  view_50 = None
        permute_26 = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        view_52 = torch.ops.aten.view.default(view_45, [1, 512, 12, 64]);  view_45 = None
        permute_27 = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        _scaled_dot_product_flash_attention_for_cpu_default_9 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_27, permute_24, permute_26, attn_mask = mul);  permute_27 = permute_24 = permute_26 = None
        getitem_59 = _scaled_dot_product_flash_attention_for_cpu_default_9[0];  _scaled_dot_product_flash_attention_for_cpu_default_9 = None
        permute_29 = torch.ops.aten.permute.default(getitem_59, [0, 2, 1, 3]);  getitem_59 = None
        clone_10 = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_59 = torch.ops.aten.view.default(clone_10, [1, 512, 768]);  clone_10 = None
        view_60 = torch.ops.aten.view.default(view_59, [512, 768]);  view_59 = None
        permute_30 = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg48_1, view_60, permute_30);  arg48_1 = view_60 = permute_30 = None
        view_61 = torch.ops.aten.view.default(addmm_15, [1, 512, 768]);  addmm_15 = None
        add_21 = torch.ops.aten.add.Tensor(view_61, add_19);  view_61 = add_19 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_22 = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_21, getitem_11);  add_21 = getitem_11 = None
        mul_17 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = rsqrt_5 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_17, arg49_1);  mul_17 = arg49_1 = None
        add_23 = torch.ops.aten.add.Tensor(mul_18, arg50_1);  mul_18 = arg50_1 = None
        view_62 = torch.ops.aten.view.default(add_23, [512, 768])
        permute_31 = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg52_1, view_62, permute_31);  arg52_1 = view_62 = permute_31 = None
        view_63 = torch.ops.aten.view.default(addmm_16, [1, 512, 3072]);  addmm_16 = None
        mul_19 = torch.ops.aten.mul.Tensor(view_63, 0.5)
        mul_20 = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
        erf_2 = torch.ops.aten.erf.default(mul_20);  mul_20 = None
        add_24 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_19, add_24);  mul_19 = add_24 = None
        view_64 = torch.ops.aten.view.default(mul_21, [512, 3072]);  mul_21 = None
        permute_32 = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg54_1, view_64, permute_32);  arg54_1 = view_64 = permute_32 = None
        view_65 = torch.ops.aten.view.default(addmm_17, [1, 512, 768]);  addmm_17 = None
        add_25 = torch.ops.aten.add.Tensor(view_65, add_23);  view_65 = add_23 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_25, getitem_13);  add_25 = getitem_13 = None
        mul_22 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = rsqrt_6 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_22, arg55_1);  mul_22 = arg55_1 = None
        add_27 = torch.ops.aten.add.Tensor(mul_23, arg56_1);  mul_23 = arg56_1 = None
        view_66 = torch.ops.aten.view.default(add_27, [512, 768])
        permute_33 = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg58_1, view_66, permute_33);  arg58_1 = view_66 = permute_33 = None
        view_67 = torch.ops.aten.view.default(addmm_18, [1, 512, 768]);  addmm_18 = None
        view_68 = torch.ops.aten.view.default(add_27, [512, 768])
        permute_34 = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg60_1, view_68, permute_34);  arg60_1 = view_68 = permute_34 = None
        view_69 = torch.ops.aten.view.default(addmm_19, [1, 512, 768]);  addmm_19 = None
        view_70 = torch.ops.aten.view.default(view_69, [1, 512, 12, 64]);  view_69 = None
        permute_35 = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        view_71 = torch.ops.aten.view.default(add_27, [512, 768])
        permute_36 = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg62_1, view_71, permute_36);  arg62_1 = view_71 = permute_36 = None
        view_72 = torch.ops.aten.view.default(addmm_20, [1, 512, 768]);  addmm_20 = None
        view_73 = torch.ops.aten.view.default(view_72, [1, 512, 12, 64]);  view_72 = None
        permute_37 = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
        view_74 = torch.ops.aten.view.default(view_67, [1, 512, 12, 64]);  view_67 = None
        permute_38 = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        _scaled_dot_product_flash_attention_for_cpu_default_8 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_38, permute_35, permute_37, attn_mask = mul);  permute_38 = permute_35 = permute_37 = None
        getitem_58 = _scaled_dot_product_flash_attention_for_cpu_default_8[0];  _scaled_dot_product_flash_attention_for_cpu_default_8 = None
        permute_40 = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3]);  getitem_58 = None
        clone_14 = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_81 = torch.ops.aten.view.default(clone_14, [1, 512, 768]);  clone_14 = None
        view_82 = torch.ops.aten.view.default(view_81, [512, 768]);  view_81 = None
        permute_41 = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg64_1, view_82, permute_41);  arg64_1 = view_82 = permute_41 = None
        view_83 = torch.ops.aten.view.default(addmm_21, [1, 512, 768]);  addmm_21 = None
        add_29 = torch.ops.aten.add.Tensor(view_83, add_27);  view_83 = add_27 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_30 = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_29, getitem_15);  add_29 = getitem_15 = None
        mul_24 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = rsqrt_7 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, arg65_1);  mul_24 = arg65_1 = None
        add_31 = torch.ops.aten.add.Tensor(mul_25, arg66_1);  mul_25 = arg66_1 = None
        view_84 = torch.ops.aten.view.default(add_31, [512, 768])
        permute_42 = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg68_1, view_84, permute_42);  arg68_1 = view_84 = permute_42 = None
        view_85 = torch.ops.aten.view.default(addmm_22, [1, 512, 3072]);  addmm_22 = None
        mul_26 = torch.ops.aten.mul.Tensor(view_85, 0.5)
        mul_27 = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
        erf_3 = torch.ops.aten.erf.default(mul_27);  mul_27 = None
        add_32 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_28 = torch.ops.aten.mul.Tensor(mul_26, add_32);  mul_26 = add_32 = None
        view_86 = torch.ops.aten.view.default(mul_28, [512, 3072]);  mul_28 = None
        permute_43 = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg70_1, view_86, permute_43);  arg70_1 = view_86 = permute_43 = None
        view_87 = torch.ops.aten.view.default(addmm_23, [1, 512, 768]);  addmm_23 = None
        add_33 = torch.ops.aten.add.Tensor(view_87, add_31);  view_87 = add_31 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_34 = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_33, getitem_17);  add_33 = getitem_17 = None
        mul_29 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = rsqrt_8 = None
        mul_30 = torch.ops.aten.mul.Tensor(mul_29, arg71_1);  mul_29 = arg71_1 = None
        add_35 = torch.ops.aten.add.Tensor(mul_30, arg72_1);  mul_30 = arg72_1 = None
        view_88 = torch.ops.aten.view.default(add_35, [512, 768])
        permute_44 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg74_1, view_88, permute_44);  arg74_1 = view_88 = permute_44 = None
        view_89 = torch.ops.aten.view.default(addmm_24, [1, 512, 768]);  addmm_24 = None
        view_90 = torch.ops.aten.view.default(add_35, [512, 768])
        permute_45 = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg76_1, view_90, permute_45);  arg76_1 = view_90 = permute_45 = None
        view_91 = torch.ops.aten.view.default(addmm_25, [1, 512, 768]);  addmm_25 = None
        view_92 = torch.ops.aten.view.default(view_91, [1, 512, 12, 64]);  view_91 = None
        permute_46 = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
        view_93 = torch.ops.aten.view.default(add_35, [512, 768])
        permute_47 = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg78_1, view_93, permute_47);  arg78_1 = view_93 = permute_47 = None
        view_94 = torch.ops.aten.view.default(addmm_26, [1, 512, 768]);  addmm_26 = None
        view_95 = torch.ops.aten.view.default(view_94, [1, 512, 12, 64]);  view_94 = None
        permute_48 = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
        view_96 = torch.ops.aten.view.default(view_89, [1, 512, 12, 64]);  view_89 = None
        permute_49 = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        _scaled_dot_product_flash_attention_for_cpu_default_7 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_49, permute_46, permute_48, attn_mask = mul);  permute_49 = permute_46 = permute_48 = None
        getitem_57 = _scaled_dot_product_flash_attention_for_cpu_default_7[0];  _scaled_dot_product_flash_attention_for_cpu_default_7 = None
        permute_51 = torch.ops.aten.permute.default(getitem_57, [0, 2, 1, 3]);  getitem_57 = None
        clone_18 = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_103 = torch.ops.aten.view.default(clone_18, [1, 512, 768]);  clone_18 = None
        view_104 = torch.ops.aten.view.default(view_103, [512, 768]);  view_103 = None
        permute_52 = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg80_1, view_104, permute_52);  arg80_1 = view_104 = permute_52 = None
        view_105 = torch.ops.aten.view.default(addmm_27, [1, 512, 768]);  addmm_27 = None
        add_37 = torch.ops.aten.add.Tensor(view_105, add_35);  view_105 = add_35 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_38 = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_37, getitem_19);  add_37 = getitem_19 = None
        mul_31 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = rsqrt_9 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_31, arg81_1);  mul_31 = arg81_1 = None
        add_39 = torch.ops.aten.add.Tensor(mul_32, arg82_1);  mul_32 = arg82_1 = None
        view_106 = torch.ops.aten.view.default(add_39, [512, 768])
        permute_53 = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg84_1, view_106, permute_53);  arg84_1 = view_106 = permute_53 = None
        view_107 = torch.ops.aten.view.default(addmm_28, [1, 512, 3072]);  addmm_28 = None
        mul_33 = torch.ops.aten.mul.Tensor(view_107, 0.5)
        mul_34 = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
        erf_4 = torch.ops.aten.erf.default(mul_34);  mul_34 = None
        add_40 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_33, add_40);  mul_33 = add_40 = None
        view_108 = torch.ops.aten.view.default(mul_35, [512, 3072]);  mul_35 = None
        permute_54 = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg86_1, view_108, permute_54);  arg86_1 = view_108 = permute_54 = None
        view_109 = torch.ops.aten.view.default(addmm_29, [1, 512, 768]);  addmm_29 = None
        add_41 = torch.ops.aten.add.Tensor(view_109, add_39);  view_109 = add_39 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_41, getitem_21);  add_41 = getitem_21 = None
        mul_36 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = rsqrt_10 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, arg87_1);  mul_36 = arg87_1 = None
        add_43 = torch.ops.aten.add.Tensor(mul_37, arg88_1);  mul_37 = arg88_1 = None
        view_110 = torch.ops.aten.view.default(add_43, [512, 768])
        permute_55 = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg90_1, view_110, permute_55);  arg90_1 = view_110 = permute_55 = None
        view_111 = torch.ops.aten.view.default(addmm_30, [1, 512, 768]);  addmm_30 = None
        view_112 = torch.ops.aten.view.default(add_43, [512, 768])
        permute_56 = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg92_1, view_112, permute_56);  arg92_1 = view_112 = permute_56 = None
        view_113 = torch.ops.aten.view.default(addmm_31, [1, 512, 768]);  addmm_31 = None
        view_114 = torch.ops.aten.view.default(view_113, [1, 512, 12, 64]);  view_113 = None
        permute_57 = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        view_115 = torch.ops.aten.view.default(add_43, [512, 768])
        permute_58 = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg94_1, view_115, permute_58);  arg94_1 = view_115 = permute_58 = None
        view_116 = torch.ops.aten.view.default(addmm_32, [1, 512, 768]);  addmm_32 = None
        view_117 = torch.ops.aten.view.default(view_116, [1, 512, 12, 64]);  view_116 = None
        permute_59 = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        view_118 = torch.ops.aten.view.default(view_111, [1, 512, 12, 64]);  view_111 = None
        permute_60 = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        _scaled_dot_product_flash_attention_for_cpu_default_6 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_60, permute_57, permute_59, attn_mask = mul);  permute_60 = permute_57 = permute_59 = None
        getitem_56 = _scaled_dot_product_flash_attention_for_cpu_default_6[0];  _scaled_dot_product_flash_attention_for_cpu_default_6 = None
        permute_62 = torch.ops.aten.permute.default(getitem_56, [0, 2, 1, 3]);  getitem_56 = None
        clone_22 = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_125 = torch.ops.aten.view.default(clone_22, [1, 512, 768]);  clone_22 = None
        view_126 = torch.ops.aten.view.default(view_125, [512, 768]);  view_125 = None
        permute_63 = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg96_1, view_126, permute_63);  arg96_1 = view_126 = permute_63 = None
        view_127 = torch.ops.aten.view.default(addmm_33, [1, 512, 768]);  addmm_33 = None
        add_45 = torch.ops.aten.add.Tensor(view_127, add_43);  view_127 = add_43 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_46 = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_45, getitem_23);  add_45 = getitem_23 = None
        mul_38 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = rsqrt_11 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, arg97_1);  mul_38 = arg97_1 = None
        add_47 = torch.ops.aten.add.Tensor(mul_39, arg98_1);  mul_39 = arg98_1 = None
        view_128 = torch.ops.aten.view.default(add_47, [512, 768])
        permute_64 = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg100_1, view_128, permute_64);  arg100_1 = view_128 = permute_64 = None
        view_129 = torch.ops.aten.view.default(addmm_34, [1, 512, 3072]);  addmm_34 = None
        mul_40 = torch.ops.aten.mul.Tensor(view_129, 0.5)
        mul_41 = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
        erf_5 = torch.ops.aten.erf.default(mul_41);  mul_41 = None
        add_48 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_42 = torch.ops.aten.mul.Tensor(mul_40, add_48);  mul_40 = add_48 = None
        view_130 = torch.ops.aten.view.default(mul_42, [512, 3072]);  mul_42 = None
        permute_65 = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg102_1, view_130, permute_65);  arg102_1 = view_130 = permute_65 = None
        view_131 = torch.ops.aten.view.default(addmm_35, [1, 512, 768]);  addmm_35 = None
        add_49 = torch.ops.aten.add.Tensor(view_131, add_47);  view_131 = add_47 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_50 = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_49, getitem_25);  add_49 = getitem_25 = None
        mul_43 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = rsqrt_12 = None
        mul_44 = torch.ops.aten.mul.Tensor(mul_43, arg103_1);  mul_43 = arg103_1 = None
        add_51 = torch.ops.aten.add.Tensor(mul_44, arg104_1);  mul_44 = arg104_1 = None
        view_132 = torch.ops.aten.view.default(add_51, [512, 768])
        permute_66 = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg106_1, view_132, permute_66);  arg106_1 = view_132 = permute_66 = None
        view_133 = torch.ops.aten.view.default(addmm_36, [1, 512, 768]);  addmm_36 = None
        view_134 = torch.ops.aten.view.default(add_51, [512, 768])
        permute_67 = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg108_1, view_134, permute_67);  arg108_1 = view_134 = permute_67 = None
        view_135 = torch.ops.aten.view.default(addmm_37, [1, 512, 768]);  addmm_37 = None
        view_136 = torch.ops.aten.view.default(view_135, [1, 512, 12, 64]);  view_135 = None
        permute_68 = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        view_137 = torch.ops.aten.view.default(add_51, [512, 768])
        permute_69 = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg110_1, view_137, permute_69);  arg110_1 = view_137 = permute_69 = None
        view_138 = torch.ops.aten.view.default(addmm_38, [1, 512, 768]);  addmm_38 = None
        view_139 = torch.ops.aten.view.default(view_138, [1, 512, 12, 64]);  view_138 = None
        permute_70 = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
        view_140 = torch.ops.aten.view.default(view_133, [1, 512, 12, 64]);  view_133 = None
        permute_71 = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
        _scaled_dot_product_flash_attention_for_cpu_default_5 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_71, permute_68, permute_70, attn_mask = mul);  permute_71 = permute_68 = permute_70 = None
        getitem_55 = _scaled_dot_product_flash_attention_for_cpu_default_5[0];  _scaled_dot_product_flash_attention_for_cpu_default_5 = None
        permute_73 = torch.ops.aten.permute.default(getitem_55, [0, 2, 1, 3]);  getitem_55 = None
        clone_26 = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_147 = torch.ops.aten.view.default(clone_26, [1, 512, 768]);  clone_26 = None
        view_148 = torch.ops.aten.view.default(view_147, [512, 768]);  view_147 = None
        permute_74 = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg112_1, view_148, permute_74);  arg112_1 = view_148 = permute_74 = None
        view_149 = torch.ops.aten.view.default(addmm_39, [1, 512, 768]);  addmm_39 = None
        add_53 = torch.ops.aten.add.Tensor(view_149, add_51);  view_149 = add_51 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_54 = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_53, getitem_27);  add_53 = getitem_27 = None
        mul_45 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = rsqrt_13 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_45, arg113_1);  mul_45 = arg113_1 = None
        add_55 = torch.ops.aten.add.Tensor(mul_46, arg114_1);  mul_46 = arg114_1 = None
        view_150 = torch.ops.aten.view.default(add_55, [512, 768])
        permute_75 = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg116_1, view_150, permute_75);  arg116_1 = view_150 = permute_75 = None
        view_151 = torch.ops.aten.view.default(addmm_40, [1, 512, 3072]);  addmm_40 = None
        mul_47 = torch.ops.aten.mul.Tensor(view_151, 0.5)
        mul_48 = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
        erf_6 = torch.ops.aten.erf.default(mul_48);  mul_48 = None
        add_56 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_47, add_56);  mul_47 = add_56 = None
        view_152 = torch.ops.aten.view.default(mul_49, [512, 3072]);  mul_49 = None
        permute_76 = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg118_1, view_152, permute_76);  arg118_1 = view_152 = permute_76 = None
        view_153 = torch.ops.aten.view.default(addmm_41, [1, 512, 768]);  addmm_41 = None
        add_57 = torch.ops.aten.add.Tensor(view_153, add_55);  view_153 = add_55 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_57, getitem_29);  add_57 = getitem_29 = None
        mul_50 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = rsqrt_14 = None
        mul_51 = torch.ops.aten.mul.Tensor(mul_50, arg119_1);  mul_50 = arg119_1 = None
        add_59 = torch.ops.aten.add.Tensor(mul_51, arg120_1);  mul_51 = arg120_1 = None
        view_154 = torch.ops.aten.view.default(add_59, [512, 768])
        permute_77 = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg122_1, view_154, permute_77);  arg122_1 = view_154 = permute_77 = None
        view_155 = torch.ops.aten.view.default(addmm_42, [1, 512, 768]);  addmm_42 = None
        view_156 = torch.ops.aten.view.default(add_59, [512, 768])
        permute_78 = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg124_1, view_156, permute_78);  arg124_1 = view_156 = permute_78 = None
        view_157 = torch.ops.aten.view.default(addmm_43, [1, 512, 768]);  addmm_43 = None
        view_158 = torch.ops.aten.view.default(view_157, [1, 512, 12, 64]);  view_157 = None
        permute_79 = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None
        view_159 = torch.ops.aten.view.default(add_59, [512, 768])
        permute_80 = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg126_1, view_159, permute_80);  arg126_1 = view_159 = permute_80 = None
        view_160 = torch.ops.aten.view.default(addmm_44, [1, 512, 768]);  addmm_44 = None
        view_161 = torch.ops.aten.view.default(view_160, [1, 512, 12, 64]);  view_160 = None
        permute_81 = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None
        view_162 = torch.ops.aten.view.default(view_155, [1, 512, 12, 64]);  view_155 = None
        permute_82 = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        _scaled_dot_product_flash_attention_for_cpu_default_4 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_82, permute_79, permute_81, attn_mask = mul);  permute_82 = permute_79 = permute_81 = None
        getitem_54 = _scaled_dot_product_flash_attention_for_cpu_default_4[0];  _scaled_dot_product_flash_attention_for_cpu_default_4 = None
        permute_84 = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None
        clone_30 = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_169 = torch.ops.aten.view.default(clone_30, [1, 512, 768]);  clone_30 = None
        view_170 = torch.ops.aten.view.default(view_169, [512, 768]);  view_169 = None
        permute_85 = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg128_1, view_170, permute_85);  arg128_1 = view_170 = permute_85 = None
        view_171 = torch.ops.aten.view.default(addmm_45, [1, 512, 768]);  addmm_45 = None
        add_61 = torch.ops.aten.add.Tensor(view_171, add_59);  view_171 = add_59 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_62 = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_61, getitem_31);  add_61 = getitem_31 = None
        mul_52 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = rsqrt_15 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, arg129_1);  mul_52 = arg129_1 = None
        add_63 = torch.ops.aten.add.Tensor(mul_53, arg130_1);  mul_53 = arg130_1 = None
        view_172 = torch.ops.aten.view.default(add_63, [512, 768])
        permute_86 = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg132_1, view_172, permute_86);  arg132_1 = view_172 = permute_86 = None
        view_173 = torch.ops.aten.view.default(addmm_46, [1, 512, 3072]);  addmm_46 = None
        mul_54 = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_55 = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
        erf_7 = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_64 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_56 = torch.ops.aten.mul.Tensor(mul_54, add_64);  mul_54 = add_64 = None
        view_174 = torch.ops.aten.view.default(mul_56, [512, 3072]);  mul_56 = None
        permute_87 = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg134_1, view_174, permute_87);  arg134_1 = view_174 = permute_87 = None
        view_175 = torch.ops.aten.view.default(addmm_47, [1, 512, 768]);  addmm_47 = None
        add_65 = torch.ops.aten.add.Tensor(view_175, add_63);  view_175 = add_63 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_66 = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_25 = torch.ops.aten.sub.Tensor(add_65, getitem_33);  add_65 = getitem_33 = None
        mul_57 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = rsqrt_16 = None
        mul_58 = torch.ops.aten.mul.Tensor(mul_57, arg135_1);  mul_57 = arg135_1 = None
        add_67 = torch.ops.aten.add.Tensor(mul_58, arg136_1);  mul_58 = arg136_1 = None
        view_176 = torch.ops.aten.view.default(add_67, [512, 768])
        permute_88 = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg138_1, view_176, permute_88);  arg138_1 = view_176 = permute_88 = None
        view_177 = torch.ops.aten.view.default(addmm_48, [1, 512, 768]);  addmm_48 = None
        view_178 = torch.ops.aten.view.default(add_67, [512, 768])
        permute_89 = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg140_1, view_178, permute_89);  arg140_1 = view_178 = permute_89 = None
        view_179 = torch.ops.aten.view.default(addmm_49, [1, 512, 768]);  addmm_49 = None
        view_180 = torch.ops.aten.view.default(view_179, [1, 512, 12, 64]);  view_179 = None
        permute_90 = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
        view_181 = torch.ops.aten.view.default(add_67, [512, 768])
        permute_91 = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg142_1, view_181, permute_91);  arg142_1 = view_181 = permute_91 = None
        view_182 = torch.ops.aten.view.default(addmm_50, [1, 512, 768]);  addmm_50 = None
        view_183 = torch.ops.aten.view.default(view_182, [1, 512, 12, 64]);  view_182 = None
        permute_92 = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
        view_184 = torch.ops.aten.view.default(view_177, [1, 512, 12, 64]);  view_177 = None
        permute_93 = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        _scaled_dot_product_flash_attention_for_cpu_default_3 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_93, permute_90, permute_92, attn_mask = mul);  permute_93 = permute_90 = permute_92 = None
        getitem_53 = _scaled_dot_product_flash_attention_for_cpu_default_3[0];  _scaled_dot_product_flash_attention_for_cpu_default_3 = None
        permute_95 = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3]);  getitem_53 = None
        clone_34 = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_191 = torch.ops.aten.view.default(clone_34, [1, 512, 768]);  clone_34 = None
        view_192 = torch.ops.aten.view.default(view_191, [512, 768]);  view_191 = None
        permute_96 = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg144_1, view_192, permute_96);  arg144_1 = view_192 = permute_96 = None
        view_193 = torch.ops.aten.view.default(addmm_51, [1, 512, 768]);  addmm_51 = None
        add_69 = torch.ops.aten.add.Tensor(view_193, add_67);  view_193 = add_67 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
        getitem_34 = var_mean_17[0]
        getitem_35 = var_mean_17[1];  var_mean_17 = None
        add_70 = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        sub_27 = torch.ops.aten.sub.Tensor(add_69, getitem_35);  add_69 = getitem_35 = None
        mul_59 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = rsqrt_17 = None
        mul_60 = torch.ops.aten.mul.Tensor(mul_59, arg145_1);  mul_59 = arg145_1 = None
        add_71 = torch.ops.aten.add.Tensor(mul_60, arg146_1);  mul_60 = arg146_1 = None
        view_194 = torch.ops.aten.view.default(add_71, [512, 768])
        permute_97 = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg148_1, view_194, permute_97);  arg148_1 = view_194 = permute_97 = None
        view_195 = torch.ops.aten.view.default(addmm_52, [1, 512, 3072]);  addmm_52 = None
        mul_61 = torch.ops.aten.mul.Tensor(view_195, 0.5)
        mul_62 = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
        erf_8 = torch.ops.aten.erf.default(mul_62);  mul_62 = None
        add_72 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_63 = torch.ops.aten.mul.Tensor(mul_61, add_72);  mul_61 = add_72 = None
        view_196 = torch.ops.aten.view.default(mul_63, [512, 3072]);  mul_63 = None
        permute_98 = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg150_1, view_196, permute_98);  arg150_1 = view_196 = permute_98 = None
        view_197 = torch.ops.aten.view.default(addmm_53, [1, 512, 768]);  addmm_53 = None
        add_73 = torch.ops.aten.add.Tensor(view_197, add_71);  view_197 = add_71 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_36 = var_mean_18[0]
        getitem_37 = var_mean_18[1];  var_mean_18 = None
        add_74 = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_28 = torch.ops.aten.sub.Tensor(add_73, getitem_37);  add_73 = getitem_37 = None
        mul_64 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = rsqrt_18 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_64, arg151_1);  mul_64 = arg151_1 = None
        add_75 = torch.ops.aten.add.Tensor(mul_65, arg152_1);  mul_65 = arg152_1 = None
        view_198 = torch.ops.aten.view.default(add_75, [512, 768])
        permute_99 = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg154_1, view_198, permute_99);  arg154_1 = view_198 = permute_99 = None
        view_199 = torch.ops.aten.view.default(addmm_54, [1, 512, 768]);  addmm_54 = None
        view_200 = torch.ops.aten.view.default(add_75, [512, 768])
        permute_100 = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg156_1, view_200, permute_100);  arg156_1 = view_200 = permute_100 = None
        view_201 = torch.ops.aten.view.default(addmm_55, [1, 512, 768]);  addmm_55 = None
        view_202 = torch.ops.aten.view.default(view_201, [1, 512, 12, 64]);  view_201 = None
        permute_101 = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
        view_203 = torch.ops.aten.view.default(add_75, [512, 768])
        permute_102 = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg158_1, view_203, permute_102);  arg158_1 = view_203 = permute_102 = None
        view_204 = torch.ops.aten.view.default(addmm_56, [1, 512, 768]);  addmm_56 = None
        view_205 = torch.ops.aten.view.default(view_204, [1, 512, 12, 64]);  view_204 = None
        permute_103 = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
        view_206 = torch.ops.aten.view.default(view_199, [1, 512, 12, 64]);  view_199 = None
        permute_104 = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        _scaled_dot_product_flash_attention_for_cpu_default_2 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_104, permute_101, permute_103, attn_mask = mul);  permute_104 = permute_101 = permute_103 = None
        getitem_52 = _scaled_dot_product_flash_attention_for_cpu_default_2[0];  _scaled_dot_product_flash_attention_for_cpu_default_2 = None
        permute_106 = torch.ops.aten.permute.default(getitem_52, [0, 2, 1, 3]);  getitem_52 = None
        clone_38 = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_213 = torch.ops.aten.view.default(clone_38, [1, 512, 768]);  clone_38 = None
        view_214 = torch.ops.aten.view.default(view_213, [512, 768]);  view_213 = None
        permute_107 = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg160_1, view_214, permute_107);  arg160_1 = view_214 = permute_107 = None
        view_215 = torch.ops.aten.view.default(addmm_57, [1, 512, 768]);  addmm_57 = None
        add_77 = torch.ops.aten.add.Tensor(view_215, add_75);  view_215 = add_75 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_19[0]
        getitem_39 = var_mean_19[1];  var_mean_19 = None
        add_78 = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_30 = torch.ops.aten.sub.Tensor(add_77, getitem_39);  add_77 = getitem_39 = None
        mul_66 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = rsqrt_19 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_66, arg161_1);  mul_66 = arg161_1 = None
        add_79 = torch.ops.aten.add.Tensor(mul_67, arg162_1);  mul_67 = arg162_1 = None
        view_216 = torch.ops.aten.view.default(add_79, [512, 768])
        permute_108 = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg164_1, view_216, permute_108);  arg164_1 = view_216 = permute_108 = None
        view_217 = torch.ops.aten.view.default(addmm_58, [1, 512, 3072]);  addmm_58 = None
        mul_68 = torch.ops.aten.mul.Tensor(view_217, 0.5)
        mul_69 = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
        erf_9 = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_80 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_70 = torch.ops.aten.mul.Tensor(mul_68, add_80);  mul_68 = add_80 = None
        view_218 = torch.ops.aten.view.default(mul_70, [512, 3072]);  mul_70 = None
        permute_109 = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg166_1, view_218, permute_109);  arg166_1 = view_218 = permute_109 = None
        view_219 = torch.ops.aten.view.default(addmm_59, [1, 512, 768]);  addmm_59 = None
        add_81 = torch.ops.aten.add.Tensor(view_219, add_79);  view_219 = add_79 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_20[0]
        getitem_41 = var_mean_20[1];  var_mean_20 = None
        add_82 = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_31 = torch.ops.aten.sub.Tensor(add_81, getitem_41);  add_81 = getitem_41 = None
        mul_71 = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = rsqrt_20 = None
        mul_72 = torch.ops.aten.mul.Tensor(mul_71, arg167_1);  mul_71 = arg167_1 = None
        add_83 = torch.ops.aten.add.Tensor(mul_72, arg168_1);  mul_72 = arg168_1 = None
        view_220 = torch.ops.aten.view.default(add_83, [512, 768])
        permute_110 = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg170_1, view_220, permute_110);  arg170_1 = view_220 = permute_110 = None
        view_221 = torch.ops.aten.view.default(addmm_60, [1, 512, 768]);  addmm_60 = None
        view_222 = torch.ops.aten.view.default(add_83, [512, 768])
        permute_111 = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg172_1, view_222, permute_111);  arg172_1 = view_222 = permute_111 = None
        view_223 = torch.ops.aten.view.default(addmm_61, [1, 512, 768]);  addmm_61 = None
        view_224 = torch.ops.aten.view.default(view_223, [1, 512, 12, 64]);  view_223 = None
        permute_112 = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
        view_225 = torch.ops.aten.view.default(add_83, [512, 768])
        permute_113 = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg174_1, view_225, permute_113);  arg174_1 = view_225 = permute_113 = None
        view_226 = torch.ops.aten.view.default(addmm_62, [1, 512, 768]);  addmm_62 = None
        view_227 = torch.ops.aten.view.default(view_226, [1, 512, 12, 64]);  view_226 = None
        permute_114 = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
        view_228 = torch.ops.aten.view.default(view_221, [1, 512, 12, 64]);  view_221 = None
        permute_115 = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
        _scaled_dot_product_flash_attention_for_cpu_default_1 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_115, permute_112, permute_114, attn_mask = mul);  permute_115 = permute_112 = permute_114 = None
        getitem_51 = _scaled_dot_product_flash_attention_for_cpu_default_1[0];  _scaled_dot_product_flash_attention_for_cpu_default_1 = None
        permute_117 = torch.ops.aten.permute.default(getitem_51, [0, 2, 1, 3]);  getitem_51 = None
        clone_42 = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_235 = torch.ops.aten.view.default(clone_42, [1, 512, 768]);  clone_42 = None
        view_236 = torch.ops.aten.view.default(view_235, [512, 768]);  view_235 = None
        permute_118 = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg176_1, view_236, permute_118);  arg176_1 = view_236 = permute_118 = None
        view_237 = torch.ops.aten.view.default(addmm_63, [1, 512, 768]);  addmm_63 = None
        add_85 = torch.ops.aten.add.Tensor(view_237, add_83);  view_237 = add_83 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_21[0]
        getitem_43 = var_mean_21[1];  var_mean_21 = None
        add_86 = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_33 = torch.ops.aten.sub.Tensor(add_85, getitem_43);  add_85 = getitem_43 = None
        mul_73 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = rsqrt_21 = None
        mul_74 = torch.ops.aten.mul.Tensor(mul_73, arg177_1);  mul_73 = arg177_1 = None
        add_87 = torch.ops.aten.add.Tensor(mul_74, arg178_1);  mul_74 = arg178_1 = None
        view_238 = torch.ops.aten.view.default(add_87, [512, 768])
        permute_119 = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg180_1, view_238, permute_119);  arg180_1 = view_238 = permute_119 = None
        view_239 = torch.ops.aten.view.default(addmm_64, [1, 512, 3072]);  addmm_64 = None
        mul_75 = torch.ops.aten.mul.Tensor(view_239, 0.5)
        mul_76 = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
        erf_10 = torch.ops.aten.erf.default(mul_76);  mul_76 = None
        add_88 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_77 = torch.ops.aten.mul.Tensor(mul_75, add_88);  mul_75 = add_88 = None
        view_240 = torch.ops.aten.view.default(mul_77, [512, 3072]);  mul_77 = None
        permute_120 = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg182_1, view_240, permute_120);  arg182_1 = view_240 = permute_120 = None
        view_241 = torch.ops.aten.view.default(addmm_65, [1, 512, 768]);  addmm_65 = None
        add_89 = torch.ops.aten.add.Tensor(view_241, add_87);  view_241 = add_87 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_22[0]
        getitem_45 = var_mean_22[1];  var_mean_22 = None
        add_90 = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_34 = torch.ops.aten.sub.Tensor(add_89, getitem_45);  add_89 = getitem_45 = None
        mul_78 = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = rsqrt_22 = None
        mul_79 = torch.ops.aten.mul.Tensor(mul_78, arg183_1);  mul_78 = arg183_1 = None
        add_91 = torch.ops.aten.add.Tensor(mul_79, arg184_1);  mul_79 = arg184_1 = None
        view_242 = torch.ops.aten.view.default(add_91, [512, 768])
        permute_121 = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg186_1, view_242, permute_121);  arg186_1 = view_242 = permute_121 = None
        view_243 = torch.ops.aten.view.default(addmm_66, [1, 512, 768]);  addmm_66 = None
        view_244 = torch.ops.aten.view.default(add_91, [512, 768])
        permute_122 = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg188_1, view_244, permute_122);  arg188_1 = view_244 = permute_122 = None
        view_245 = torch.ops.aten.view.default(addmm_67, [1, 512, 768]);  addmm_67 = None
        view_246 = torch.ops.aten.view.default(view_245, [1, 512, 12, 64]);  view_245 = None
        permute_123 = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
        view_247 = torch.ops.aten.view.default(add_91, [512, 768])
        permute_124 = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg190_1, view_247, permute_124);  arg190_1 = view_247 = permute_124 = None
        view_248 = torch.ops.aten.view.default(addmm_68, [1, 512, 768]);  addmm_68 = None
        view_249 = torch.ops.aten.view.default(view_248, [1, 512, 12, 64]);  view_248 = None
        permute_125 = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None
        view_250 = torch.ops.aten.view.default(view_243, [1, 512, 12, 64]);  view_243 = None
        permute_126 = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
        _scaled_dot_product_flash_attention_for_cpu_default = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_126, permute_123, permute_125, attn_mask = mul);  permute_126 = permute_123 = permute_125 = mul = None
        getitem_50 = _scaled_dot_product_flash_attention_for_cpu_default[0];  _scaled_dot_product_flash_attention_for_cpu_default = None
        permute_128 = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3]);  getitem_50 = None
        clone_46 = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_257 = torch.ops.aten.view.default(clone_46, [1, 512, 768]);  clone_46 = None
        view_258 = torch.ops.aten.view.default(view_257, [512, 768]);  view_257 = None
        permute_129 = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg192_1, view_258, permute_129);  arg192_1 = view_258 = permute_129 = None
        view_259 = torch.ops.aten.view.default(addmm_69, [1, 512, 768]);  addmm_69 = None
        add_93 = torch.ops.aten.add.Tensor(view_259, add_91);  view_259 = add_91 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_23[0]
        getitem_47 = var_mean_23[1];  var_mean_23 = None
        add_94 = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_36 = torch.ops.aten.sub.Tensor(add_93, getitem_47);  add_93 = getitem_47 = None
        mul_80 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = rsqrt_23 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, arg193_1);  mul_80 = arg193_1 = None
        add_95 = torch.ops.aten.add.Tensor(mul_81, arg194_1);  mul_81 = arg194_1 = None
        view_260 = torch.ops.aten.view.default(add_95, [512, 768])
        permute_130 = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg196_1, view_260, permute_130);  arg196_1 = view_260 = permute_130 = None
        view_261 = torch.ops.aten.view.default(addmm_70, [1, 512, 3072]);  addmm_70 = None
        mul_82 = torch.ops.aten.mul.Tensor(view_261, 0.5)
        mul_83 = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
        erf_11 = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_96 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_84 = torch.ops.aten.mul.Tensor(mul_82, add_96);  mul_82 = add_96 = None
        view_262 = torch.ops.aten.view.default(mul_84, [512, 3072]);  mul_84 = None
        permute_131 = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg198_1, view_262, permute_131);  arg198_1 = view_262 = permute_131 = None
        view_263 = torch.ops.aten.view.default(addmm_71, [1, 512, 768]);  addmm_71 = None
        add_97 = torch.ops.aten.add.Tensor(view_263, add_95);  view_263 = add_95 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_24[0]
        getitem_49 = var_mean_24[1];  var_mean_24 = None
        add_98 = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_37 = torch.ops.aten.sub.Tensor(add_97, getitem_49);  add_97 = getitem_49 = None
        mul_85 = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = rsqrt_24 = None
        mul_86 = torch.ops.aten.mul.Tensor(mul_85, arg199_1);  mul_85 = arg199_1 = None
        add_99 = torch.ops.aten.add.Tensor(mul_86, arg200_1);  mul_86 = arg200_1 = None
        select = torch.ops.aten.select.int(add_99, 1, 0);  add_99 = None
        permute_132 = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        addmm_72 = torch.ops.aten.addmm.default(arg202_1, select, permute_132);  arg202_1 = select = permute_132 = None
        tanh = torch.ops.aten.tanh.default(addmm_72);  addmm_72 = None
        permute_133 = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_73 = torch.ops.aten.addmm.default(arg204_1, tanh, permute_133);  arg204_1 = tanh = permute_133 = None
        eq = torch.ops.aten.eq.Scalar(arg0_1, 1998);  arg0_1 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(eq, [1])
        ge = torch.ops.aten.ge.Scalar(sum_13, 8);  sum_13 = None
        full_default = torch.ops.aten.full.default([1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        select_1 = torch.ops.aten.select.int(eq, 1, 0)
        select_2 = torch.ops.aten.select.int(eq, 1, 2)
        bitwise_and = torch.ops.aten.bitwise_and.Tensor(select_1, select_2);  select_1 = select_2 = None
        select_3 = torch.ops.aten.select.int(eq, 1, 5)
        bitwise_and_1 = torch.ops.aten.bitwise_and.Tensor(bitwise_and, select_3);  bitwise_and = select_3 = None
        select_4 = torch.ops.aten.select.int(eq, 1, 6)
        bitwise_and_2 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_1, select_4);  bitwise_and_1 = select_4 = None
        select_5 = torch.ops.aten.select.int(eq, 1, 10)
        bitwise_and_3 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, select_5);  bitwise_and_2 = select_5 = None
        select_6 = torch.ops.aten.select.int(eq, 1, 12)
        bitwise_and_4 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_3, select_6);  bitwise_and_3 = select_6 = None
        select_7 = torch.ops.aten.select.int(eq, 1, 15)
        bitwise_and_5 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, select_7);  bitwise_and_4 = select_7 = None
        select_8 = torch.ops.aten.select.int(eq, 1, 16)
        bitwise_and_6 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_5, select_8);  bitwise_and_5 = select_8 = None
        bitwise_or = torch.ops.aten.bitwise_or.Tensor(full_default, bitwise_and_6);  full_default = bitwise_and_6 = None
        select_9 = torch.ops.aten.select.int(eq, 1, 1)
        select_10 = torch.ops.aten.select.int(eq, 1, 3)
        bitwise_and_7 = torch.ops.aten.bitwise_and.Tensor(select_9, select_10);  select_9 = select_10 = None
        select_11 = torch.ops.aten.select.int(eq, 1, 6)
        bitwise_and_8 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_7, select_11);  bitwise_and_7 = select_11 = None
        select_12 = torch.ops.aten.select.int(eq, 1, 7)
        bitwise_and_9 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_8, select_12);  bitwise_and_8 = select_12 = None
        select_13 = torch.ops.aten.select.int(eq, 1, 11)
        bitwise_and_10 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_9, select_13);  bitwise_and_9 = select_13 = None
        select_14 = torch.ops.aten.select.int(eq, 1, 13)
        bitwise_and_11 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_10, select_14);  bitwise_and_10 = select_14 = None
        select_15 = torch.ops.aten.select.int(eq, 1, 16)
        bitwise_and_12 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_11, select_15);  bitwise_and_11 = select_15 = None
        select_16 = torch.ops.aten.select.int(eq, 1, 17)
        bitwise_and_13 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_12, select_16);  bitwise_and_12 = select_16 = None
        bitwise_or_1 = torch.ops.aten.bitwise_or.Tensor(bitwise_or, bitwise_and_13);  bitwise_or = bitwise_and_13 = None
        select_17 = torch.ops.aten.select.int(eq, 1, 2)
        select_18 = torch.ops.aten.select.int(eq, 1, 4)
        bitwise_and_14 = torch.ops.aten.bitwise_and.Tensor(select_17, select_18);  select_17 = select_18 = None
        select_19 = torch.ops.aten.select.int(eq, 1, 7)
        bitwise_and_15 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_14, select_19);  bitwise_and_14 = select_19 = None
        select_20 = torch.ops.aten.select.int(eq, 1, 8)
        bitwise_and_16 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_15, select_20);  bitwise_and_15 = select_20 = None
        select_21 = torch.ops.aten.select.int(eq, 1, 12)
        bitwise_and_17 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_16, select_21);  bitwise_and_16 = select_21 = None
        select_22 = torch.ops.aten.select.int(eq, 1, 14)
        bitwise_and_18 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_17, select_22);  bitwise_and_17 = select_22 = None
        select_23 = torch.ops.aten.select.int(eq, 1, 17)
        bitwise_and_19 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_18, select_23);  bitwise_and_18 = select_23 = None
        select_24 = torch.ops.aten.select.int(eq, 1, 18)
        bitwise_and_20 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_19, select_24);  bitwise_and_19 = select_24 = None
        bitwise_or_2 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_1, bitwise_and_20);  bitwise_or_1 = bitwise_and_20 = None
        select_25 = torch.ops.aten.select.int(eq, 1, 3)
        select_26 = torch.ops.aten.select.int(eq, 1, 5)
        bitwise_and_21 = torch.ops.aten.bitwise_and.Tensor(select_25, select_26);  select_25 = select_26 = None
        select_27 = torch.ops.aten.select.int(eq, 1, 8)
        bitwise_and_22 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_21, select_27);  bitwise_and_21 = select_27 = None
        select_28 = torch.ops.aten.select.int(eq, 1, 9)
        bitwise_and_23 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_22, select_28);  bitwise_and_22 = select_28 = None
        select_29 = torch.ops.aten.select.int(eq, 1, 13)
        bitwise_and_24 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_23, select_29);  bitwise_and_23 = select_29 = None
        select_30 = torch.ops.aten.select.int(eq, 1, 15)
        bitwise_and_25 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_24, select_30);  bitwise_and_24 = select_30 = None
        select_31 = torch.ops.aten.select.int(eq, 1, 18)
        bitwise_and_26 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_25, select_31);  bitwise_and_25 = select_31 = None
        select_32 = torch.ops.aten.select.int(eq, 1, 19)
        bitwise_and_27 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_26, select_32);  bitwise_and_26 = select_32 = None
        bitwise_or_3 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_2, bitwise_and_27);  bitwise_or_2 = bitwise_and_27 = None
        select_33 = torch.ops.aten.select.int(eq, 1, 4)
        select_34 = torch.ops.aten.select.int(eq, 1, 6)
        bitwise_and_28 = torch.ops.aten.bitwise_and.Tensor(select_33, select_34);  select_33 = select_34 = None
        select_35 = torch.ops.aten.select.int(eq, 1, 9)
        bitwise_and_29 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_28, select_35);  bitwise_and_28 = select_35 = None
        select_36 = torch.ops.aten.select.int(eq, 1, 10)
        bitwise_and_30 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_29, select_36);  bitwise_and_29 = select_36 = None
        select_37 = torch.ops.aten.select.int(eq, 1, 14)
        bitwise_and_31 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_30, select_37);  bitwise_and_30 = select_37 = None
        select_38 = torch.ops.aten.select.int(eq, 1, 16)
        bitwise_and_32 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_31, select_38);  bitwise_and_31 = select_38 = None
        select_39 = torch.ops.aten.select.int(eq, 1, 19)
        bitwise_and_33 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_32, select_39);  bitwise_and_32 = select_39 = None
        select_40 = torch.ops.aten.select.int(eq, 1, 20)
        bitwise_and_34 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_33, select_40);  bitwise_and_33 = select_40 = None
        bitwise_or_4 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_3, bitwise_and_34);  bitwise_or_3 = bitwise_and_34 = None
        select_41 = torch.ops.aten.select.int(eq, 1, 5)
        select_42 = torch.ops.aten.select.int(eq, 1, 7)
        bitwise_and_35 = torch.ops.aten.bitwise_and.Tensor(select_41, select_42);  select_41 = select_42 = None
        select_43 = torch.ops.aten.select.int(eq, 1, 10)
        bitwise_and_36 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_35, select_43);  bitwise_and_35 = select_43 = None
        select_44 = torch.ops.aten.select.int(eq, 1, 11)
        bitwise_and_37 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_36, select_44);  bitwise_and_36 = select_44 = None
        select_45 = torch.ops.aten.select.int(eq, 1, 15)
        bitwise_and_38 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_37, select_45);  bitwise_and_37 = select_45 = None
        select_46 = torch.ops.aten.select.int(eq, 1, 17)
        bitwise_and_39 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_38, select_46);  bitwise_and_38 = select_46 = None
        select_47 = torch.ops.aten.select.int(eq, 1, 20)
        bitwise_and_40 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_39, select_47);  bitwise_and_39 = select_47 = None
        select_48 = torch.ops.aten.select.int(eq, 1, 21)
        bitwise_and_41 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_40, select_48);  bitwise_and_40 = select_48 = None
        bitwise_or_5 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_4, bitwise_and_41);  bitwise_or_4 = bitwise_and_41 = None
        select_49 = torch.ops.aten.select.int(eq, 1, 6)
        select_50 = torch.ops.aten.select.int(eq, 1, 8)
        bitwise_and_42 = torch.ops.aten.bitwise_and.Tensor(select_49, select_50);  select_49 = select_50 = None
        select_51 = torch.ops.aten.select.int(eq, 1, 11)
        bitwise_and_43 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_42, select_51);  bitwise_and_42 = select_51 = None
        select_52 = torch.ops.aten.select.int(eq, 1, 12)
        bitwise_and_44 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_43, select_52);  bitwise_and_43 = select_52 = None
        select_53 = torch.ops.aten.select.int(eq, 1, 16)
        bitwise_and_45 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_44, select_53);  bitwise_and_44 = select_53 = None
        select_54 = torch.ops.aten.select.int(eq, 1, 18)
        bitwise_and_46 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_45, select_54);  bitwise_and_45 = select_54 = None
        select_55 = torch.ops.aten.select.int(eq, 1, 21)
        bitwise_and_47 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_46, select_55);  bitwise_and_46 = select_55 = None
        select_56 = torch.ops.aten.select.int(eq, 1, 22)
        bitwise_and_48 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_47, select_56);  bitwise_and_47 = select_56 = None
        bitwise_or_6 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_5, bitwise_and_48);  bitwise_or_5 = bitwise_and_48 = None
        select_57 = torch.ops.aten.select.int(eq, 1, 7)
        select_58 = torch.ops.aten.select.int(eq, 1, 9)
        bitwise_and_49 = torch.ops.aten.bitwise_and.Tensor(select_57, select_58);  select_57 = select_58 = None
        select_59 = torch.ops.aten.select.int(eq, 1, 12)
        bitwise_and_50 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_49, select_59);  bitwise_and_49 = select_59 = None
        select_60 = torch.ops.aten.select.int(eq, 1, 13)
        bitwise_and_51 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_50, select_60);  bitwise_and_50 = select_60 = None
        select_61 = torch.ops.aten.select.int(eq, 1, 17)
        bitwise_and_52 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_51, select_61);  bitwise_and_51 = select_61 = None
        select_62 = torch.ops.aten.select.int(eq, 1, 19)
        bitwise_and_53 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_52, select_62);  bitwise_and_52 = select_62 = None
        select_63 = torch.ops.aten.select.int(eq, 1, 22)
        bitwise_and_54 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_53, select_63);  bitwise_and_53 = select_63 = None
        select_64 = torch.ops.aten.select.int(eq, 1, 23)
        bitwise_and_55 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_54, select_64);  bitwise_and_54 = select_64 = None
        bitwise_or_7 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_6, bitwise_and_55);  bitwise_or_6 = bitwise_and_55 = None
        select_65 = torch.ops.aten.select.int(eq, 1, 8)
        select_66 = torch.ops.aten.select.int(eq, 1, 10)
        bitwise_and_56 = torch.ops.aten.bitwise_and.Tensor(select_65, select_66);  select_65 = select_66 = None
        select_67 = torch.ops.aten.select.int(eq, 1, 13)
        bitwise_and_57 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_56, select_67);  bitwise_and_56 = select_67 = None
        select_68 = torch.ops.aten.select.int(eq, 1, 14)
        bitwise_and_58 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_57, select_68);  bitwise_and_57 = select_68 = None
        select_69 = torch.ops.aten.select.int(eq, 1, 18)
        bitwise_and_59 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_58, select_69);  bitwise_and_58 = select_69 = None
        select_70 = torch.ops.aten.select.int(eq, 1, 20)
        bitwise_and_60 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_59, select_70);  bitwise_and_59 = select_70 = None
        select_71 = torch.ops.aten.select.int(eq, 1, 23)
        bitwise_and_61 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_60, select_71);  bitwise_and_60 = select_71 = None
        select_72 = torch.ops.aten.select.int(eq, 1, 24)
        bitwise_and_62 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_61, select_72);  bitwise_and_61 = select_72 = None
        bitwise_or_8 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_7, bitwise_and_62);  bitwise_or_7 = bitwise_and_62 = None
        select_73 = torch.ops.aten.select.int(eq, 1, 9)
        select_74 = torch.ops.aten.select.int(eq, 1, 11)
        bitwise_and_63 = torch.ops.aten.bitwise_and.Tensor(select_73, select_74);  select_73 = select_74 = None
        select_75 = torch.ops.aten.select.int(eq, 1, 14)
        bitwise_and_64 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_63, select_75);  bitwise_and_63 = select_75 = None
        select_76 = torch.ops.aten.select.int(eq, 1, 15)
        bitwise_and_65 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_64, select_76);  bitwise_and_64 = select_76 = None
        select_77 = torch.ops.aten.select.int(eq, 1, 19)
        bitwise_and_66 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_65, select_77);  bitwise_and_65 = select_77 = None
        select_78 = torch.ops.aten.select.int(eq, 1, 21)
        bitwise_and_67 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_66, select_78);  bitwise_and_66 = select_78 = None
        select_79 = torch.ops.aten.select.int(eq, 1, 24)
        bitwise_and_68 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_67, select_79);  bitwise_and_67 = select_79 = None
        select_80 = torch.ops.aten.select.int(eq, 1, 25)
        bitwise_and_69 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_68, select_80);  bitwise_and_68 = select_80 = None
        bitwise_or_9 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_8, bitwise_and_69);  bitwise_or_8 = bitwise_and_69 = None
        select_81 = torch.ops.aten.select.int(eq, 1, 10)
        select_82 = torch.ops.aten.select.int(eq, 1, 12)
        bitwise_and_70 = torch.ops.aten.bitwise_and.Tensor(select_81, select_82);  select_81 = select_82 = None
        select_83 = torch.ops.aten.select.int(eq, 1, 15)
        bitwise_and_71 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_70, select_83);  bitwise_and_70 = select_83 = None
        select_84 = torch.ops.aten.select.int(eq, 1, 16)
        bitwise_and_72 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_71, select_84);  bitwise_and_71 = select_84 = None
        select_85 = torch.ops.aten.select.int(eq, 1, 20)
        bitwise_and_73 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_72, select_85);  bitwise_and_72 = select_85 = None
        select_86 = torch.ops.aten.select.int(eq, 1, 22)
        bitwise_and_74 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_73, select_86);  bitwise_and_73 = select_86 = None
        select_87 = torch.ops.aten.select.int(eq, 1, 25)
        bitwise_and_75 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_74, select_87);  bitwise_and_74 = select_87 = None
        select_88 = torch.ops.aten.select.int(eq, 1, 26)
        bitwise_and_76 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_75, select_88);  bitwise_and_75 = select_88 = None
        bitwise_or_10 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_9, bitwise_and_76);  bitwise_or_9 = bitwise_and_76 = None
        select_89 = torch.ops.aten.select.int(eq, 1, 11)
        select_90 = torch.ops.aten.select.int(eq, 1, 13)
        bitwise_and_77 = torch.ops.aten.bitwise_and.Tensor(select_89, select_90);  select_89 = select_90 = None
        select_91 = torch.ops.aten.select.int(eq, 1, 16)
        bitwise_and_78 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_77, select_91);  bitwise_and_77 = select_91 = None
        select_92 = torch.ops.aten.select.int(eq, 1, 17)
        bitwise_and_79 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_78, select_92);  bitwise_and_78 = select_92 = None
        select_93 = torch.ops.aten.select.int(eq, 1, 21)
        bitwise_and_80 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_79, select_93);  bitwise_and_79 = select_93 = None
        select_94 = torch.ops.aten.select.int(eq, 1, 23)
        bitwise_and_81 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_80, select_94);  bitwise_and_80 = select_94 = None
        select_95 = torch.ops.aten.select.int(eq, 1, 26)
        bitwise_and_82 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_81, select_95);  bitwise_and_81 = select_95 = None
        select_96 = torch.ops.aten.select.int(eq, 1, 27)
        bitwise_and_83 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_82, select_96);  bitwise_and_82 = select_96 = None
        bitwise_or_11 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_10, bitwise_and_83);  bitwise_or_10 = bitwise_and_83 = None
        select_97 = torch.ops.aten.select.int(eq, 1, 12)
        select_98 = torch.ops.aten.select.int(eq, 1, 14)
        bitwise_and_84 = torch.ops.aten.bitwise_and.Tensor(select_97, select_98);  select_97 = select_98 = None
        select_99 = torch.ops.aten.select.int(eq, 1, 17)
        bitwise_and_85 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_84, select_99);  bitwise_and_84 = select_99 = None
        select_100 = torch.ops.aten.select.int(eq, 1, 18)
        bitwise_and_86 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_85, select_100);  bitwise_and_85 = select_100 = None
        select_101 = torch.ops.aten.select.int(eq, 1, 22)
        bitwise_and_87 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_86, select_101);  bitwise_and_86 = select_101 = None
        select_102 = torch.ops.aten.select.int(eq, 1, 24)
        bitwise_and_88 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_87, select_102);  bitwise_and_87 = select_102 = None
        select_103 = torch.ops.aten.select.int(eq, 1, 27)
        bitwise_and_89 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_88, select_103);  bitwise_and_88 = select_103 = None
        select_104 = torch.ops.aten.select.int(eq, 1, 28)
        bitwise_and_90 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_89, select_104);  bitwise_and_89 = select_104 = None
        bitwise_or_12 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_11, bitwise_and_90);  bitwise_or_11 = bitwise_and_90 = None
        select_105 = torch.ops.aten.select.int(eq, 1, 13)
        select_106 = torch.ops.aten.select.int(eq, 1, 15)
        bitwise_and_91 = torch.ops.aten.bitwise_and.Tensor(select_105, select_106);  select_105 = select_106 = None
        select_107 = torch.ops.aten.select.int(eq, 1, 18)
        bitwise_and_92 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_91, select_107);  bitwise_and_91 = select_107 = None
        select_108 = torch.ops.aten.select.int(eq, 1, 19)
        bitwise_and_93 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_92, select_108);  bitwise_and_92 = select_108 = None
        select_109 = torch.ops.aten.select.int(eq, 1, 23)
        bitwise_and_94 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_93, select_109);  bitwise_and_93 = select_109 = None
        select_110 = torch.ops.aten.select.int(eq, 1, 25)
        bitwise_and_95 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_94, select_110);  bitwise_and_94 = select_110 = None
        select_111 = torch.ops.aten.select.int(eq, 1, 28)
        bitwise_and_96 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_95, select_111);  bitwise_and_95 = select_111 = None
        select_112 = torch.ops.aten.select.int(eq, 1, 29)
        bitwise_and_97 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_96, select_112);  bitwise_and_96 = select_112 = None
        bitwise_or_13 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_12, bitwise_and_97);  bitwise_or_12 = bitwise_and_97 = None
        select_113 = torch.ops.aten.select.int(eq, 1, 14)
        select_114 = torch.ops.aten.select.int(eq, 1, 16)
        bitwise_and_98 = torch.ops.aten.bitwise_and.Tensor(select_113, select_114);  select_113 = select_114 = None
        select_115 = torch.ops.aten.select.int(eq, 1, 19)
        bitwise_and_99 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_98, select_115);  bitwise_and_98 = select_115 = None
        select_116 = torch.ops.aten.select.int(eq, 1, 20)
        bitwise_and_100 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_99, select_116);  bitwise_and_99 = select_116 = None
        select_117 = torch.ops.aten.select.int(eq, 1, 24)
        bitwise_and_101 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_100, select_117);  bitwise_and_100 = select_117 = None
        select_118 = torch.ops.aten.select.int(eq, 1, 26)
        bitwise_and_102 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_101, select_118);  bitwise_and_101 = select_118 = None
        select_119 = torch.ops.aten.select.int(eq, 1, 29)
        bitwise_and_103 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_102, select_119);  bitwise_and_102 = select_119 = None
        select_120 = torch.ops.aten.select.int(eq, 1, 30)
        bitwise_and_104 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_103, select_120);  bitwise_and_103 = select_120 = None
        bitwise_or_14 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_13, bitwise_and_104);  bitwise_or_13 = bitwise_and_104 = None
        select_121 = torch.ops.aten.select.int(eq, 1, 15)
        select_122 = torch.ops.aten.select.int(eq, 1, 17)
        bitwise_and_105 = torch.ops.aten.bitwise_and.Tensor(select_121, select_122);  select_121 = select_122 = None
        select_123 = torch.ops.aten.select.int(eq, 1, 20)
        bitwise_and_106 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_105, select_123);  bitwise_and_105 = select_123 = None
        select_124 = torch.ops.aten.select.int(eq, 1, 21)
        bitwise_and_107 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_106, select_124);  bitwise_and_106 = select_124 = None
        select_125 = torch.ops.aten.select.int(eq, 1, 25)
        bitwise_and_108 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_107, select_125);  bitwise_and_107 = select_125 = None
        select_126 = torch.ops.aten.select.int(eq, 1, 27)
        bitwise_and_109 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_108, select_126);  bitwise_and_108 = select_126 = None
        select_127 = torch.ops.aten.select.int(eq, 1, 30)
        bitwise_and_110 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_109, select_127);  bitwise_and_109 = select_127 = None
        select_128 = torch.ops.aten.select.int(eq, 1, 31)
        bitwise_and_111 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_110, select_128);  bitwise_and_110 = select_128 = None
        bitwise_or_15 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_14, bitwise_and_111);  bitwise_or_14 = bitwise_and_111 = None
        select_129 = torch.ops.aten.select.int(eq, 1, 16)
        select_130 = torch.ops.aten.select.int(eq, 1, 18)
        bitwise_and_112 = torch.ops.aten.bitwise_and.Tensor(select_129, select_130);  select_129 = select_130 = None
        select_131 = torch.ops.aten.select.int(eq, 1, 21)
        bitwise_and_113 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_112, select_131);  bitwise_and_112 = select_131 = None
        select_132 = torch.ops.aten.select.int(eq, 1, 22)
        bitwise_and_114 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_113, select_132);  bitwise_and_113 = select_132 = None
        select_133 = torch.ops.aten.select.int(eq, 1, 26)
        bitwise_and_115 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_114, select_133);  bitwise_and_114 = select_133 = None
        select_134 = torch.ops.aten.select.int(eq, 1, 28)
        bitwise_and_116 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_115, select_134);  bitwise_and_115 = select_134 = None
        select_135 = torch.ops.aten.select.int(eq, 1, 31)
        bitwise_and_117 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_116, select_135);  bitwise_and_116 = select_135 = None
        select_136 = torch.ops.aten.select.int(eq, 1, 32)
        bitwise_and_118 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_117, select_136);  bitwise_and_117 = select_136 = None
        bitwise_or_16 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_15, bitwise_and_118);  bitwise_or_15 = bitwise_and_118 = None
        select_137 = torch.ops.aten.select.int(eq, 1, 17)
        select_138 = torch.ops.aten.select.int(eq, 1, 19)
        bitwise_and_119 = torch.ops.aten.bitwise_and.Tensor(select_137, select_138);  select_137 = select_138 = None
        select_139 = torch.ops.aten.select.int(eq, 1, 22)
        bitwise_and_120 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_119, select_139);  bitwise_and_119 = select_139 = None
        select_140 = torch.ops.aten.select.int(eq, 1, 23)
        bitwise_and_121 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_120, select_140);  bitwise_and_120 = select_140 = None
        select_141 = torch.ops.aten.select.int(eq, 1, 27)
        bitwise_and_122 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_121, select_141);  bitwise_and_121 = select_141 = None
        select_142 = torch.ops.aten.select.int(eq, 1, 29)
        bitwise_and_123 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_122, select_142);  bitwise_and_122 = select_142 = None
        select_143 = torch.ops.aten.select.int(eq, 1, 32)
        bitwise_and_124 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_123, select_143);  bitwise_and_123 = select_143 = None
        select_144 = torch.ops.aten.select.int(eq, 1, 33)
        bitwise_and_125 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_124, select_144);  bitwise_and_124 = select_144 = None
        bitwise_or_17 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_16, bitwise_and_125);  bitwise_or_16 = bitwise_and_125 = None
        select_145 = torch.ops.aten.select.int(eq, 1, 18)
        select_146 = torch.ops.aten.select.int(eq, 1, 20)
        bitwise_and_126 = torch.ops.aten.bitwise_and.Tensor(select_145, select_146);  select_145 = select_146 = None
        select_147 = torch.ops.aten.select.int(eq, 1, 23)
        bitwise_and_127 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_126, select_147);  bitwise_and_126 = select_147 = None
        select_148 = torch.ops.aten.select.int(eq, 1, 24)
        bitwise_and_128 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_127, select_148);  bitwise_and_127 = select_148 = None
        select_149 = torch.ops.aten.select.int(eq, 1, 28)
        bitwise_and_129 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_128, select_149);  bitwise_and_128 = select_149 = None
        select_150 = torch.ops.aten.select.int(eq, 1, 30)
        bitwise_and_130 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_129, select_150);  bitwise_and_129 = select_150 = None
        select_151 = torch.ops.aten.select.int(eq, 1, 33)
        bitwise_and_131 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_130, select_151);  bitwise_and_130 = select_151 = None
        select_152 = torch.ops.aten.select.int(eq, 1, 34)
        bitwise_and_132 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_131, select_152);  bitwise_and_131 = select_152 = None
        bitwise_or_18 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_17, bitwise_and_132);  bitwise_or_17 = bitwise_and_132 = None
        select_153 = torch.ops.aten.select.int(eq, 1, 19)
        select_154 = torch.ops.aten.select.int(eq, 1, 21)
        bitwise_and_133 = torch.ops.aten.bitwise_and.Tensor(select_153, select_154);  select_153 = select_154 = None
        select_155 = torch.ops.aten.select.int(eq, 1, 24)
        bitwise_and_134 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_133, select_155);  bitwise_and_133 = select_155 = None
        select_156 = torch.ops.aten.select.int(eq, 1, 25)
        bitwise_and_135 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_134, select_156);  bitwise_and_134 = select_156 = None
        select_157 = torch.ops.aten.select.int(eq, 1, 29)
        bitwise_and_136 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_135, select_157);  bitwise_and_135 = select_157 = None
        select_158 = torch.ops.aten.select.int(eq, 1, 31)
        bitwise_and_137 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_136, select_158);  bitwise_and_136 = select_158 = None
        select_159 = torch.ops.aten.select.int(eq, 1, 34)
        bitwise_and_138 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_137, select_159);  bitwise_and_137 = select_159 = None
        select_160 = torch.ops.aten.select.int(eq, 1, 35)
        bitwise_and_139 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_138, select_160);  bitwise_and_138 = select_160 = None
        bitwise_or_19 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_18, bitwise_and_139);  bitwise_or_18 = bitwise_and_139 = None
        select_161 = torch.ops.aten.select.int(eq, 1, 20)
        select_162 = torch.ops.aten.select.int(eq, 1, 22)
        bitwise_and_140 = torch.ops.aten.bitwise_and.Tensor(select_161, select_162);  select_161 = select_162 = None
        select_163 = torch.ops.aten.select.int(eq, 1, 25)
        bitwise_and_141 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_140, select_163);  bitwise_and_140 = select_163 = None
        select_164 = torch.ops.aten.select.int(eq, 1, 26)
        bitwise_and_142 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_141, select_164);  bitwise_and_141 = select_164 = None
        select_165 = torch.ops.aten.select.int(eq, 1, 30)
        bitwise_and_143 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_142, select_165);  bitwise_and_142 = select_165 = None
        select_166 = torch.ops.aten.select.int(eq, 1, 32)
        bitwise_and_144 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_143, select_166);  bitwise_and_143 = select_166 = None
        select_167 = torch.ops.aten.select.int(eq, 1, 35)
        bitwise_and_145 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_144, select_167);  bitwise_and_144 = select_167 = None
        select_168 = torch.ops.aten.select.int(eq, 1, 36)
        bitwise_and_146 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_145, select_168);  bitwise_and_145 = select_168 = None
        bitwise_or_20 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_19, bitwise_and_146);  bitwise_or_19 = bitwise_and_146 = None
        select_169 = torch.ops.aten.select.int(eq, 1, 21)
        select_170 = torch.ops.aten.select.int(eq, 1, 23)
        bitwise_and_147 = torch.ops.aten.bitwise_and.Tensor(select_169, select_170);  select_169 = select_170 = None
        select_171 = torch.ops.aten.select.int(eq, 1, 26)
        bitwise_and_148 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_147, select_171);  bitwise_and_147 = select_171 = None
        select_172 = torch.ops.aten.select.int(eq, 1, 27)
        bitwise_and_149 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_148, select_172);  bitwise_and_148 = select_172 = None
        select_173 = torch.ops.aten.select.int(eq, 1, 31)
        bitwise_and_150 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_149, select_173);  bitwise_and_149 = select_173 = None
        select_174 = torch.ops.aten.select.int(eq, 1, 33)
        bitwise_and_151 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_150, select_174);  bitwise_and_150 = select_174 = None
        select_175 = torch.ops.aten.select.int(eq, 1, 36)
        bitwise_and_152 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_151, select_175);  bitwise_and_151 = select_175 = None
        select_176 = torch.ops.aten.select.int(eq, 1, 37)
        bitwise_and_153 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_152, select_176);  bitwise_and_152 = select_176 = None
        bitwise_or_21 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_20, bitwise_and_153);  bitwise_or_20 = bitwise_and_153 = None
        select_177 = torch.ops.aten.select.int(eq, 1, 22)
        select_178 = torch.ops.aten.select.int(eq, 1, 24)
        bitwise_and_154 = torch.ops.aten.bitwise_and.Tensor(select_177, select_178);  select_177 = select_178 = None
        select_179 = torch.ops.aten.select.int(eq, 1, 27)
        bitwise_and_155 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_154, select_179);  bitwise_and_154 = select_179 = None
        select_180 = torch.ops.aten.select.int(eq, 1, 28)
        bitwise_and_156 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_155, select_180);  bitwise_and_155 = select_180 = None
        select_181 = torch.ops.aten.select.int(eq, 1, 32)
        bitwise_and_157 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_156, select_181);  bitwise_and_156 = select_181 = None
        select_182 = torch.ops.aten.select.int(eq, 1, 34)
        bitwise_and_158 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_157, select_182);  bitwise_and_157 = select_182 = None
        select_183 = torch.ops.aten.select.int(eq, 1, 37)
        bitwise_and_159 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_158, select_183);  bitwise_and_158 = select_183 = None
        select_184 = torch.ops.aten.select.int(eq, 1, 38)
        bitwise_and_160 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_159, select_184);  bitwise_and_159 = select_184 = None
        bitwise_or_22 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_21, bitwise_and_160);  bitwise_or_21 = bitwise_and_160 = None
        select_185 = torch.ops.aten.select.int(eq, 1, 23)
        select_186 = torch.ops.aten.select.int(eq, 1, 25)
        bitwise_and_161 = torch.ops.aten.bitwise_and.Tensor(select_185, select_186);  select_185 = select_186 = None
        select_187 = torch.ops.aten.select.int(eq, 1, 28)
        bitwise_and_162 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_161, select_187);  bitwise_and_161 = select_187 = None
        select_188 = torch.ops.aten.select.int(eq, 1, 29)
        bitwise_and_163 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_162, select_188);  bitwise_and_162 = select_188 = None
        select_189 = torch.ops.aten.select.int(eq, 1, 33)
        bitwise_and_164 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_163, select_189);  bitwise_and_163 = select_189 = None
        select_190 = torch.ops.aten.select.int(eq, 1, 35)
        bitwise_and_165 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_164, select_190);  bitwise_and_164 = select_190 = None
        select_191 = torch.ops.aten.select.int(eq, 1, 38)
        bitwise_and_166 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_165, select_191);  bitwise_and_165 = select_191 = None
        select_192 = torch.ops.aten.select.int(eq, 1, 39)
        bitwise_and_167 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_166, select_192);  bitwise_and_166 = select_192 = None
        bitwise_or_23 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_22, bitwise_and_167);  bitwise_or_22 = bitwise_and_167 = None
        select_193 = torch.ops.aten.select.int(eq, 1, 24)
        select_194 = torch.ops.aten.select.int(eq, 1, 26)
        bitwise_and_168 = torch.ops.aten.bitwise_and.Tensor(select_193, select_194);  select_193 = select_194 = None
        select_195 = torch.ops.aten.select.int(eq, 1, 29)
        bitwise_and_169 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_168, select_195);  bitwise_and_168 = select_195 = None
        select_196 = torch.ops.aten.select.int(eq, 1, 30)
        bitwise_and_170 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_169, select_196);  bitwise_and_169 = select_196 = None
        select_197 = torch.ops.aten.select.int(eq, 1, 34)
        bitwise_and_171 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_170, select_197);  bitwise_and_170 = select_197 = None
        select_198 = torch.ops.aten.select.int(eq, 1, 36)
        bitwise_and_172 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_171, select_198);  bitwise_and_171 = select_198 = None
        select_199 = torch.ops.aten.select.int(eq, 1, 39)
        bitwise_and_173 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_172, select_199);  bitwise_and_172 = select_199 = None
        select_200 = torch.ops.aten.select.int(eq, 1, 40)
        bitwise_and_174 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_173, select_200);  bitwise_and_173 = select_200 = None
        bitwise_or_24 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_23, bitwise_and_174);  bitwise_or_23 = bitwise_and_174 = None
        select_201 = torch.ops.aten.select.int(eq, 1, 25)
        select_202 = torch.ops.aten.select.int(eq, 1, 27)
        bitwise_and_175 = torch.ops.aten.bitwise_and.Tensor(select_201, select_202);  select_201 = select_202 = None
        select_203 = torch.ops.aten.select.int(eq, 1, 30)
        bitwise_and_176 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_175, select_203);  bitwise_and_175 = select_203 = None
        select_204 = torch.ops.aten.select.int(eq, 1, 31)
        bitwise_and_177 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_176, select_204);  bitwise_and_176 = select_204 = None
        select_205 = torch.ops.aten.select.int(eq, 1, 35)
        bitwise_and_178 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_177, select_205);  bitwise_and_177 = select_205 = None
        select_206 = torch.ops.aten.select.int(eq, 1, 37)
        bitwise_and_179 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_178, select_206);  bitwise_and_178 = select_206 = None
        select_207 = torch.ops.aten.select.int(eq, 1, 40)
        bitwise_and_180 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_179, select_207);  bitwise_and_179 = select_207 = None
        select_208 = torch.ops.aten.select.int(eq, 1, 41)
        bitwise_and_181 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_180, select_208);  bitwise_and_180 = select_208 = None
        bitwise_or_25 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_24, bitwise_and_181);  bitwise_or_24 = bitwise_and_181 = None
        select_209 = torch.ops.aten.select.int(eq, 1, 26)
        select_210 = torch.ops.aten.select.int(eq, 1, 28)
        bitwise_and_182 = torch.ops.aten.bitwise_and.Tensor(select_209, select_210);  select_209 = select_210 = None
        select_211 = torch.ops.aten.select.int(eq, 1, 31)
        bitwise_and_183 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_182, select_211);  bitwise_and_182 = select_211 = None
        select_212 = torch.ops.aten.select.int(eq, 1, 32)
        bitwise_and_184 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_183, select_212);  bitwise_and_183 = select_212 = None
        select_213 = torch.ops.aten.select.int(eq, 1, 36)
        bitwise_and_185 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_184, select_213);  bitwise_and_184 = select_213 = None
        select_214 = torch.ops.aten.select.int(eq, 1, 38)
        bitwise_and_186 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_185, select_214);  bitwise_and_185 = select_214 = None
        select_215 = torch.ops.aten.select.int(eq, 1, 41)
        bitwise_and_187 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_186, select_215);  bitwise_and_186 = select_215 = None
        select_216 = torch.ops.aten.select.int(eq, 1, 42)
        bitwise_and_188 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_187, select_216);  bitwise_and_187 = select_216 = None
        bitwise_or_26 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_25, bitwise_and_188);  bitwise_or_25 = bitwise_and_188 = None
        select_217 = torch.ops.aten.select.int(eq, 1, 27)
        select_218 = torch.ops.aten.select.int(eq, 1, 29)
        bitwise_and_189 = torch.ops.aten.bitwise_and.Tensor(select_217, select_218);  select_217 = select_218 = None
        select_219 = torch.ops.aten.select.int(eq, 1, 32)
        bitwise_and_190 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_189, select_219);  bitwise_and_189 = select_219 = None
        select_220 = torch.ops.aten.select.int(eq, 1, 33)
        bitwise_and_191 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_190, select_220);  bitwise_and_190 = select_220 = None
        select_221 = torch.ops.aten.select.int(eq, 1, 37)
        bitwise_and_192 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_191, select_221);  bitwise_and_191 = select_221 = None
        select_222 = torch.ops.aten.select.int(eq, 1, 39)
        bitwise_and_193 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_192, select_222);  bitwise_and_192 = select_222 = None
        select_223 = torch.ops.aten.select.int(eq, 1, 42)
        bitwise_and_194 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_193, select_223);  bitwise_and_193 = select_223 = None
        select_224 = torch.ops.aten.select.int(eq, 1, 43)
        bitwise_and_195 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_194, select_224);  bitwise_and_194 = select_224 = None
        bitwise_or_27 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_26, bitwise_and_195);  bitwise_or_26 = bitwise_and_195 = None
        select_225 = torch.ops.aten.select.int(eq, 1, 28)
        select_226 = torch.ops.aten.select.int(eq, 1, 30)
        bitwise_and_196 = torch.ops.aten.bitwise_and.Tensor(select_225, select_226);  select_225 = select_226 = None
        select_227 = torch.ops.aten.select.int(eq, 1, 33)
        bitwise_and_197 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_196, select_227);  bitwise_and_196 = select_227 = None
        select_228 = torch.ops.aten.select.int(eq, 1, 34)
        bitwise_and_198 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_197, select_228);  bitwise_and_197 = select_228 = None
        select_229 = torch.ops.aten.select.int(eq, 1, 38)
        bitwise_and_199 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_198, select_229);  bitwise_and_198 = select_229 = None
        select_230 = torch.ops.aten.select.int(eq, 1, 40)
        bitwise_and_200 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_199, select_230);  bitwise_and_199 = select_230 = None
        select_231 = torch.ops.aten.select.int(eq, 1, 43)
        bitwise_and_201 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_200, select_231);  bitwise_and_200 = select_231 = None
        select_232 = torch.ops.aten.select.int(eq, 1, 44)
        bitwise_and_202 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_201, select_232);  bitwise_and_201 = select_232 = None
        bitwise_or_28 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_27, bitwise_and_202);  bitwise_or_27 = bitwise_and_202 = None
        select_233 = torch.ops.aten.select.int(eq, 1, 29)
        select_234 = torch.ops.aten.select.int(eq, 1, 31)
        bitwise_and_203 = torch.ops.aten.bitwise_and.Tensor(select_233, select_234);  select_233 = select_234 = None
        select_235 = torch.ops.aten.select.int(eq, 1, 34)
        bitwise_and_204 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_203, select_235);  bitwise_and_203 = select_235 = None
        select_236 = torch.ops.aten.select.int(eq, 1, 35)
        bitwise_and_205 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_204, select_236);  bitwise_and_204 = select_236 = None
        select_237 = torch.ops.aten.select.int(eq, 1, 39)
        bitwise_and_206 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_205, select_237);  bitwise_and_205 = select_237 = None
        select_238 = torch.ops.aten.select.int(eq, 1, 41)
        bitwise_and_207 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_206, select_238);  bitwise_and_206 = select_238 = None
        select_239 = torch.ops.aten.select.int(eq, 1, 44)
        bitwise_and_208 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_207, select_239);  bitwise_and_207 = select_239 = None
        select_240 = torch.ops.aten.select.int(eq, 1, 45)
        bitwise_and_209 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_208, select_240);  bitwise_and_208 = select_240 = None
        bitwise_or_29 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_28, bitwise_and_209);  bitwise_or_28 = bitwise_and_209 = None
        select_241 = torch.ops.aten.select.int(eq, 1, 30)
        select_242 = torch.ops.aten.select.int(eq, 1, 32)
        bitwise_and_210 = torch.ops.aten.bitwise_and.Tensor(select_241, select_242);  select_241 = select_242 = None
        select_243 = torch.ops.aten.select.int(eq, 1, 35)
        bitwise_and_211 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_210, select_243);  bitwise_and_210 = select_243 = None
        select_244 = torch.ops.aten.select.int(eq, 1, 36)
        bitwise_and_212 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_211, select_244);  bitwise_and_211 = select_244 = None
        select_245 = torch.ops.aten.select.int(eq, 1, 40)
        bitwise_and_213 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_212, select_245);  bitwise_and_212 = select_245 = None
        select_246 = torch.ops.aten.select.int(eq, 1, 42)
        bitwise_and_214 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_213, select_246);  bitwise_and_213 = select_246 = None
        select_247 = torch.ops.aten.select.int(eq, 1, 45)
        bitwise_and_215 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_214, select_247);  bitwise_and_214 = select_247 = None
        select_248 = torch.ops.aten.select.int(eq, 1, 46)
        bitwise_and_216 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_215, select_248);  bitwise_and_215 = select_248 = None
        bitwise_or_30 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_29, bitwise_and_216);  bitwise_or_29 = bitwise_and_216 = None
        select_249 = torch.ops.aten.select.int(eq, 1, 31)
        select_250 = torch.ops.aten.select.int(eq, 1, 33)
        bitwise_and_217 = torch.ops.aten.bitwise_and.Tensor(select_249, select_250);  select_249 = select_250 = None
        select_251 = torch.ops.aten.select.int(eq, 1, 36)
        bitwise_and_218 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_217, select_251);  bitwise_and_217 = select_251 = None
        select_252 = torch.ops.aten.select.int(eq, 1, 37)
        bitwise_and_219 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_218, select_252);  bitwise_and_218 = select_252 = None
        select_253 = torch.ops.aten.select.int(eq, 1, 41)
        bitwise_and_220 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_219, select_253);  bitwise_and_219 = select_253 = None
        select_254 = torch.ops.aten.select.int(eq, 1, 43)
        bitwise_and_221 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_220, select_254);  bitwise_and_220 = select_254 = None
        select_255 = torch.ops.aten.select.int(eq, 1, 46)
        bitwise_and_222 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_221, select_255);  bitwise_and_221 = select_255 = None
        select_256 = torch.ops.aten.select.int(eq, 1, 47)
        bitwise_and_223 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_222, select_256);  bitwise_and_222 = select_256 = None
        bitwise_or_31 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_30, bitwise_and_223);  bitwise_or_30 = bitwise_and_223 = None
        select_257 = torch.ops.aten.select.int(eq, 1, 32)
        select_258 = torch.ops.aten.select.int(eq, 1, 34)
        bitwise_and_224 = torch.ops.aten.bitwise_and.Tensor(select_257, select_258);  select_257 = select_258 = None
        select_259 = torch.ops.aten.select.int(eq, 1, 37)
        bitwise_and_225 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_224, select_259);  bitwise_and_224 = select_259 = None
        select_260 = torch.ops.aten.select.int(eq, 1, 38)
        bitwise_and_226 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_225, select_260);  bitwise_and_225 = select_260 = None
        select_261 = torch.ops.aten.select.int(eq, 1, 42)
        bitwise_and_227 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_226, select_261);  bitwise_and_226 = select_261 = None
        select_262 = torch.ops.aten.select.int(eq, 1, 44)
        bitwise_and_228 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_227, select_262);  bitwise_and_227 = select_262 = None
        select_263 = torch.ops.aten.select.int(eq, 1, 47)
        bitwise_and_229 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_228, select_263);  bitwise_and_228 = select_263 = None
        select_264 = torch.ops.aten.select.int(eq, 1, 48)
        bitwise_and_230 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_229, select_264);  bitwise_and_229 = select_264 = None
        bitwise_or_32 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_31, bitwise_and_230);  bitwise_or_31 = bitwise_and_230 = None
        select_265 = torch.ops.aten.select.int(eq, 1, 33)
        select_266 = torch.ops.aten.select.int(eq, 1, 35)
        bitwise_and_231 = torch.ops.aten.bitwise_and.Tensor(select_265, select_266);  select_265 = select_266 = None
        select_267 = torch.ops.aten.select.int(eq, 1, 38)
        bitwise_and_232 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_231, select_267);  bitwise_and_231 = select_267 = None
        select_268 = torch.ops.aten.select.int(eq, 1, 39)
        bitwise_and_233 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_232, select_268);  bitwise_and_232 = select_268 = None
        select_269 = torch.ops.aten.select.int(eq, 1, 43)
        bitwise_and_234 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_233, select_269);  bitwise_and_233 = select_269 = None
        select_270 = torch.ops.aten.select.int(eq, 1, 45)
        bitwise_and_235 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_234, select_270);  bitwise_and_234 = select_270 = None
        select_271 = torch.ops.aten.select.int(eq, 1, 48)
        bitwise_and_236 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_235, select_271);  bitwise_and_235 = select_271 = None
        select_272 = torch.ops.aten.select.int(eq, 1, 49)
        bitwise_and_237 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_236, select_272);  bitwise_and_236 = select_272 = None
        bitwise_or_33 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_32, bitwise_and_237);  bitwise_or_32 = bitwise_and_237 = None
        select_273 = torch.ops.aten.select.int(eq, 1, 34)
        select_274 = torch.ops.aten.select.int(eq, 1, 36)
        bitwise_and_238 = torch.ops.aten.bitwise_and.Tensor(select_273, select_274);  select_273 = select_274 = None
        select_275 = torch.ops.aten.select.int(eq, 1, 39)
        bitwise_and_239 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_238, select_275);  bitwise_and_238 = select_275 = None
        select_276 = torch.ops.aten.select.int(eq, 1, 40)
        bitwise_and_240 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_239, select_276);  bitwise_and_239 = select_276 = None
        select_277 = torch.ops.aten.select.int(eq, 1, 44)
        bitwise_and_241 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_240, select_277);  bitwise_and_240 = select_277 = None
        select_278 = torch.ops.aten.select.int(eq, 1, 46)
        bitwise_and_242 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_241, select_278);  bitwise_and_241 = select_278 = None
        select_279 = torch.ops.aten.select.int(eq, 1, 49)
        bitwise_and_243 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_242, select_279);  bitwise_and_242 = select_279 = None
        select_280 = torch.ops.aten.select.int(eq, 1, 50)
        bitwise_and_244 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_243, select_280);  bitwise_and_243 = select_280 = None
        bitwise_or_34 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_33, bitwise_and_244);  bitwise_or_33 = bitwise_and_244 = None
        select_281 = torch.ops.aten.select.int(eq, 1, 35)
        select_282 = torch.ops.aten.select.int(eq, 1, 37)
        bitwise_and_245 = torch.ops.aten.bitwise_and.Tensor(select_281, select_282);  select_281 = select_282 = None
        select_283 = torch.ops.aten.select.int(eq, 1, 40)
        bitwise_and_246 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_245, select_283);  bitwise_and_245 = select_283 = None
        select_284 = torch.ops.aten.select.int(eq, 1, 41)
        bitwise_and_247 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_246, select_284);  bitwise_and_246 = select_284 = None
        select_285 = torch.ops.aten.select.int(eq, 1, 45)
        bitwise_and_248 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_247, select_285);  bitwise_and_247 = select_285 = None
        select_286 = torch.ops.aten.select.int(eq, 1, 47)
        bitwise_and_249 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_248, select_286);  bitwise_and_248 = select_286 = None
        select_287 = torch.ops.aten.select.int(eq, 1, 50)
        bitwise_and_250 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_249, select_287);  bitwise_and_249 = select_287 = None
        select_288 = torch.ops.aten.select.int(eq, 1, 51)
        bitwise_and_251 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_250, select_288);  bitwise_and_250 = select_288 = None
        bitwise_or_35 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_34, bitwise_and_251);  bitwise_or_34 = bitwise_and_251 = None
        select_289 = torch.ops.aten.select.int(eq, 1, 36)
        select_290 = torch.ops.aten.select.int(eq, 1, 38)
        bitwise_and_252 = torch.ops.aten.bitwise_and.Tensor(select_289, select_290);  select_289 = select_290 = None
        select_291 = torch.ops.aten.select.int(eq, 1, 41)
        bitwise_and_253 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_252, select_291);  bitwise_and_252 = select_291 = None
        select_292 = torch.ops.aten.select.int(eq, 1, 42)
        bitwise_and_254 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_253, select_292);  bitwise_and_253 = select_292 = None
        select_293 = torch.ops.aten.select.int(eq, 1, 46)
        bitwise_and_255 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_254, select_293);  bitwise_and_254 = select_293 = None
        select_294 = torch.ops.aten.select.int(eq, 1, 48)
        bitwise_and_256 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_255, select_294);  bitwise_and_255 = select_294 = None
        select_295 = torch.ops.aten.select.int(eq, 1, 51)
        bitwise_and_257 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_256, select_295);  bitwise_and_256 = select_295 = None
        select_296 = torch.ops.aten.select.int(eq, 1, 52)
        bitwise_and_258 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_257, select_296);  bitwise_and_257 = select_296 = None
        bitwise_or_36 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_35, bitwise_and_258);  bitwise_or_35 = bitwise_and_258 = None
        select_297 = torch.ops.aten.select.int(eq, 1, 37)
        select_298 = torch.ops.aten.select.int(eq, 1, 39)
        bitwise_and_259 = torch.ops.aten.bitwise_and.Tensor(select_297, select_298);  select_297 = select_298 = None
        select_299 = torch.ops.aten.select.int(eq, 1, 42)
        bitwise_and_260 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_259, select_299);  bitwise_and_259 = select_299 = None
        select_300 = torch.ops.aten.select.int(eq, 1, 43)
        bitwise_and_261 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_260, select_300);  bitwise_and_260 = select_300 = None
        select_301 = torch.ops.aten.select.int(eq, 1, 47)
        bitwise_and_262 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_261, select_301);  bitwise_and_261 = select_301 = None
        select_302 = torch.ops.aten.select.int(eq, 1, 49)
        bitwise_and_263 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_262, select_302);  bitwise_and_262 = select_302 = None
        select_303 = torch.ops.aten.select.int(eq, 1, 52)
        bitwise_and_264 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_263, select_303);  bitwise_and_263 = select_303 = None
        select_304 = torch.ops.aten.select.int(eq, 1, 53)
        bitwise_and_265 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_264, select_304);  bitwise_and_264 = select_304 = None
        bitwise_or_37 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_36, bitwise_and_265);  bitwise_or_36 = bitwise_and_265 = None
        select_305 = torch.ops.aten.select.int(eq, 1, 38)
        select_306 = torch.ops.aten.select.int(eq, 1, 40)
        bitwise_and_266 = torch.ops.aten.bitwise_and.Tensor(select_305, select_306);  select_305 = select_306 = None
        select_307 = torch.ops.aten.select.int(eq, 1, 43)
        bitwise_and_267 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_266, select_307);  bitwise_and_266 = select_307 = None
        select_308 = torch.ops.aten.select.int(eq, 1, 44)
        bitwise_and_268 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_267, select_308);  bitwise_and_267 = select_308 = None
        select_309 = torch.ops.aten.select.int(eq, 1, 48)
        bitwise_and_269 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_268, select_309);  bitwise_and_268 = select_309 = None
        select_310 = torch.ops.aten.select.int(eq, 1, 50)
        bitwise_and_270 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_269, select_310);  bitwise_and_269 = select_310 = None
        select_311 = torch.ops.aten.select.int(eq, 1, 53)
        bitwise_and_271 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_270, select_311);  bitwise_and_270 = select_311 = None
        select_312 = torch.ops.aten.select.int(eq, 1, 54)
        bitwise_and_272 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_271, select_312);  bitwise_and_271 = select_312 = None
        bitwise_or_38 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_37, bitwise_and_272);  bitwise_or_37 = bitwise_and_272 = None
        select_313 = torch.ops.aten.select.int(eq, 1, 39)
        select_314 = torch.ops.aten.select.int(eq, 1, 41)
        bitwise_and_273 = torch.ops.aten.bitwise_and.Tensor(select_313, select_314);  select_313 = select_314 = None
        select_315 = torch.ops.aten.select.int(eq, 1, 44)
        bitwise_and_274 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_273, select_315);  bitwise_and_273 = select_315 = None
        select_316 = torch.ops.aten.select.int(eq, 1, 45)
        bitwise_and_275 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_274, select_316);  bitwise_and_274 = select_316 = None
        select_317 = torch.ops.aten.select.int(eq, 1, 49)
        bitwise_and_276 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_275, select_317);  bitwise_and_275 = select_317 = None
        select_318 = torch.ops.aten.select.int(eq, 1, 51)
        bitwise_and_277 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_276, select_318);  bitwise_and_276 = select_318 = None
        select_319 = torch.ops.aten.select.int(eq, 1, 54)
        bitwise_and_278 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_277, select_319);  bitwise_and_277 = select_319 = None
        select_320 = torch.ops.aten.select.int(eq, 1, 55)
        bitwise_and_279 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_278, select_320);  bitwise_and_278 = select_320 = None
        bitwise_or_39 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_38, bitwise_and_279);  bitwise_or_38 = bitwise_and_279 = None
        select_321 = torch.ops.aten.select.int(eq, 1, 40)
        select_322 = torch.ops.aten.select.int(eq, 1, 42)
        bitwise_and_280 = torch.ops.aten.bitwise_and.Tensor(select_321, select_322);  select_321 = select_322 = None
        select_323 = torch.ops.aten.select.int(eq, 1, 45)
        bitwise_and_281 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_280, select_323);  bitwise_and_280 = select_323 = None
        select_324 = torch.ops.aten.select.int(eq, 1, 46)
        bitwise_and_282 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_281, select_324);  bitwise_and_281 = select_324 = None
        select_325 = torch.ops.aten.select.int(eq, 1, 50)
        bitwise_and_283 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_282, select_325);  bitwise_and_282 = select_325 = None
        select_326 = torch.ops.aten.select.int(eq, 1, 52)
        bitwise_and_284 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_283, select_326);  bitwise_and_283 = select_326 = None
        select_327 = torch.ops.aten.select.int(eq, 1, 55)
        bitwise_and_285 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_284, select_327);  bitwise_and_284 = select_327 = None
        select_328 = torch.ops.aten.select.int(eq, 1, 56)
        bitwise_and_286 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_285, select_328);  bitwise_and_285 = select_328 = None
        bitwise_or_40 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_39, bitwise_and_286);  bitwise_or_39 = bitwise_and_286 = None
        select_329 = torch.ops.aten.select.int(eq, 1, 41)
        select_330 = torch.ops.aten.select.int(eq, 1, 43)
        bitwise_and_287 = torch.ops.aten.bitwise_and.Tensor(select_329, select_330);  select_329 = select_330 = None
        select_331 = torch.ops.aten.select.int(eq, 1, 46)
        bitwise_and_288 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_287, select_331);  bitwise_and_287 = select_331 = None
        select_332 = torch.ops.aten.select.int(eq, 1, 47)
        bitwise_and_289 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_288, select_332);  bitwise_and_288 = select_332 = None
        select_333 = torch.ops.aten.select.int(eq, 1, 51)
        bitwise_and_290 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_289, select_333);  bitwise_and_289 = select_333 = None
        select_334 = torch.ops.aten.select.int(eq, 1, 53)
        bitwise_and_291 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_290, select_334);  bitwise_and_290 = select_334 = None
        select_335 = torch.ops.aten.select.int(eq, 1, 56)
        bitwise_and_292 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_291, select_335);  bitwise_and_291 = select_335 = None
        select_336 = torch.ops.aten.select.int(eq, 1, 57)
        bitwise_and_293 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_292, select_336);  bitwise_and_292 = select_336 = None
        bitwise_or_41 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_40, bitwise_and_293);  bitwise_or_40 = bitwise_and_293 = None
        select_337 = torch.ops.aten.select.int(eq, 1, 42)
        select_338 = torch.ops.aten.select.int(eq, 1, 44)
        bitwise_and_294 = torch.ops.aten.bitwise_and.Tensor(select_337, select_338);  select_337 = select_338 = None
        select_339 = torch.ops.aten.select.int(eq, 1, 47)
        bitwise_and_295 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_294, select_339);  bitwise_and_294 = select_339 = None
        select_340 = torch.ops.aten.select.int(eq, 1, 48)
        bitwise_and_296 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_295, select_340);  bitwise_and_295 = select_340 = None
        select_341 = torch.ops.aten.select.int(eq, 1, 52)
        bitwise_and_297 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_296, select_341);  bitwise_and_296 = select_341 = None
        select_342 = torch.ops.aten.select.int(eq, 1, 54)
        bitwise_and_298 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_297, select_342);  bitwise_and_297 = select_342 = None
        select_343 = torch.ops.aten.select.int(eq, 1, 57)
        bitwise_and_299 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_298, select_343);  bitwise_and_298 = select_343 = None
        select_344 = torch.ops.aten.select.int(eq, 1, 58)
        bitwise_and_300 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_299, select_344);  bitwise_and_299 = select_344 = None
        bitwise_or_42 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_41, bitwise_and_300);  bitwise_or_41 = bitwise_and_300 = None
        select_345 = torch.ops.aten.select.int(eq, 1, 43)
        select_346 = torch.ops.aten.select.int(eq, 1, 45)
        bitwise_and_301 = torch.ops.aten.bitwise_and.Tensor(select_345, select_346);  select_345 = select_346 = None
        select_347 = torch.ops.aten.select.int(eq, 1, 48)
        bitwise_and_302 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_301, select_347);  bitwise_and_301 = select_347 = None
        select_348 = torch.ops.aten.select.int(eq, 1, 49)
        bitwise_and_303 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_302, select_348);  bitwise_and_302 = select_348 = None
        select_349 = torch.ops.aten.select.int(eq, 1, 53)
        bitwise_and_304 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_303, select_349);  bitwise_and_303 = select_349 = None
        select_350 = torch.ops.aten.select.int(eq, 1, 55)
        bitwise_and_305 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_304, select_350);  bitwise_and_304 = select_350 = None
        select_351 = torch.ops.aten.select.int(eq, 1, 58)
        bitwise_and_306 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_305, select_351);  bitwise_and_305 = select_351 = None
        select_352 = torch.ops.aten.select.int(eq, 1, 59)
        bitwise_and_307 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_306, select_352);  bitwise_and_306 = select_352 = None
        bitwise_or_43 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_42, bitwise_and_307);  bitwise_or_42 = bitwise_and_307 = None
        select_353 = torch.ops.aten.select.int(eq, 1, 44)
        select_354 = torch.ops.aten.select.int(eq, 1, 46)
        bitwise_and_308 = torch.ops.aten.bitwise_and.Tensor(select_353, select_354);  select_353 = select_354 = None
        select_355 = torch.ops.aten.select.int(eq, 1, 49)
        bitwise_and_309 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_308, select_355);  bitwise_and_308 = select_355 = None
        select_356 = torch.ops.aten.select.int(eq, 1, 50)
        bitwise_and_310 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_309, select_356);  bitwise_and_309 = select_356 = None
        select_357 = torch.ops.aten.select.int(eq, 1, 54)
        bitwise_and_311 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_310, select_357);  bitwise_and_310 = select_357 = None
        select_358 = torch.ops.aten.select.int(eq, 1, 56)
        bitwise_and_312 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_311, select_358);  bitwise_and_311 = select_358 = None
        select_359 = torch.ops.aten.select.int(eq, 1, 59)
        bitwise_and_313 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_312, select_359);  bitwise_and_312 = select_359 = None
        select_360 = torch.ops.aten.select.int(eq, 1, 60)
        bitwise_and_314 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_313, select_360);  bitwise_and_313 = select_360 = None
        bitwise_or_44 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_43, bitwise_and_314);  bitwise_or_43 = bitwise_and_314 = None
        select_361 = torch.ops.aten.select.int(eq, 1, 45)
        select_362 = torch.ops.aten.select.int(eq, 1, 47)
        bitwise_and_315 = torch.ops.aten.bitwise_and.Tensor(select_361, select_362);  select_361 = select_362 = None
        select_363 = torch.ops.aten.select.int(eq, 1, 50)
        bitwise_and_316 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_315, select_363);  bitwise_and_315 = select_363 = None
        select_364 = torch.ops.aten.select.int(eq, 1, 51)
        bitwise_and_317 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_316, select_364);  bitwise_and_316 = select_364 = None
        select_365 = torch.ops.aten.select.int(eq, 1, 55)
        bitwise_and_318 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_317, select_365);  bitwise_and_317 = select_365 = None
        select_366 = torch.ops.aten.select.int(eq, 1, 57)
        bitwise_and_319 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_318, select_366);  bitwise_and_318 = select_366 = None
        select_367 = torch.ops.aten.select.int(eq, 1, 60)
        bitwise_and_320 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_319, select_367);  bitwise_and_319 = select_367 = None
        select_368 = torch.ops.aten.select.int(eq, 1, 61)
        bitwise_and_321 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_320, select_368);  bitwise_and_320 = select_368 = None
        bitwise_or_45 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_44, bitwise_and_321);  bitwise_or_44 = bitwise_and_321 = None
        select_369 = torch.ops.aten.select.int(eq, 1, 46)
        select_370 = torch.ops.aten.select.int(eq, 1, 48)
        bitwise_and_322 = torch.ops.aten.bitwise_and.Tensor(select_369, select_370);  select_369 = select_370 = None
        select_371 = torch.ops.aten.select.int(eq, 1, 51)
        bitwise_and_323 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_322, select_371);  bitwise_and_322 = select_371 = None
        select_372 = torch.ops.aten.select.int(eq, 1, 52)
        bitwise_and_324 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_323, select_372);  bitwise_and_323 = select_372 = None
        select_373 = torch.ops.aten.select.int(eq, 1, 56)
        bitwise_and_325 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_324, select_373);  bitwise_and_324 = select_373 = None
        select_374 = torch.ops.aten.select.int(eq, 1, 58)
        bitwise_and_326 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_325, select_374);  bitwise_and_325 = select_374 = None
        select_375 = torch.ops.aten.select.int(eq, 1, 61)
        bitwise_and_327 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_326, select_375);  bitwise_and_326 = select_375 = None
        select_376 = torch.ops.aten.select.int(eq, 1, 62)
        bitwise_and_328 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_327, select_376);  bitwise_and_327 = select_376 = None
        bitwise_or_46 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_45, bitwise_and_328);  bitwise_or_45 = bitwise_and_328 = None
        select_377 = torch.ops.aten.select.int(eq, 1, 47)
        select_378 = torch.ops.aten.select.int(eq, 1, 49)
        bitwise_and_329 = torch.ops.aten.bitwise_and.Tensor(select_377, select_378);  select_377 = select_378 = None
        select_379 = torch.ops.aten.select.int(eq, 1, 52)
        bitwise_and_330 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_329, select_379);  bitwise_and_329 = select_379 = None
        select_380 = torch.ops.aten.select.int(eq, 1, 53)
        bitwise_and_331 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_330, select_380);  bitwise_and_330 = select_380 = None
        select_381 = torch.ops.aten.select.int(eq, 1, 57)
        bitwise_and_332 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_331, select_381);  bitwise_and_331 = select_381 = None
        select_382 = torch.ops.aten.select.int(eq, 1, 59)
        bitwise_and_333 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_332, select_382);  bitwise_and_332 = select_382 = None
        select_383 = torch.ops.aten.select.int(eq, 1, 62)
        bitwise_and_334 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_333, select_383);  bitwise_and_333 = select_383 = None
        select_384 = torch.ops.aten.select.int(eq, 1, 63)
        bitwise_and_335 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_334, select_384);  bitwise_and_334 = select_384 = None
        bitwise_or_47 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_46, bitwise_and_335);  bitwise_or_46 = bitwise_and_335 = None
        select_385 = torch.ops.aten.select.int(eq, 1, 48)
        select_386 = torch.ops.aten.select.int(eq, 1, 50)
        bitwise_and_336 = torch.ops.aten.bitwise_and.Tensor(select_385, select_386);  select_385 = select_386 = None
        select_387 = torch.ops.aten.select.int(eq, 1, 53)
        bitwise_and_337 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_336, select_387);  bitwise_and_336 = select_387 = None
        select_388 = torch.ops.aten.select.int(eq, 1, 54)
        bitwise_and_338 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_337, select_388);  bitwise_and_337 = select_388 = None
        select_389 = torch.ops.aten.select.int(eq, 1, 58)
        bitwise_and_339 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_338, select_389);  bitwise_and_338 = select_389 = None
        select_390 = torch.ops.aten.select.int(eq, 1, 60)
        bitwise_and_340 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_339, select_390);  bitwise_and_339 = select_390 = None
        select_391 = torch.ops.aten.select.int(eq, 1, 63)
        bitwise_and_341 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_340, select_391);  bitwise_and_340 = select_391 = None
        select_392 = torch.ops.aten.select.int(eq, 1, 64)
        bitwise_and_342 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_341, select_392);  bitwise_and_341 = select_392 = None
        bitwise_or_48 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_47, bitwise_and_342);  bitwise_or_47 = bitwise_and_342 = None
        select_393 = torch.ops.aten.select.int(eq, 1, 49)
        select_394 = torch.ops.aten.select.int(eq, 1, 51)
        bitwise_and_343 = torch.ops.aten.bitwise_and.Tensor(select_393, select_394);  select_393 = select_394 = None
        select_395 = torch.ops.aten.select.int(eq, 1, 54)
        bitwise_and_344 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_343, select_395);  bitwise_and_343 = select_395 = None
        select_396 = torch.ops.aten.select.int(eq, 1, 55)
        bitwise_and_345 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_344, select_396);  bitwise_and_344 = select_396 = None
        select_397 = torch.ops.aten.select.int(eq, 1, 59)
        bitwise_and_346 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_345, select_397);  bitwise_and_345 = select_397 = None
        select_398 = torch.ops.aten.select.int(eq, 1, 61)
        bitwise_and_347 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_346, select_398);  bitwise_and_346 = select_398 = None
        select_399 = torch.ops.aten.select.int(eq, 1, 64)
        bitwise_and_348 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_347, select_399);  bitwise_and_347 = select_399 = None
        select_400 = torch.ops.aten.select.int(eq, 1, 65)
        bitwise_and_349 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_348, select_400);  bitwise_and_348 = select_400 = None
        bitwise_or_49 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_48, bitwise_and_349);  bitwise_or_48 = bitwise_and_349 = None
        select_401 = torch.ops.aten.select.int(eq, 1, 50)
        select_402 = torch.ops.aten.select.int(eq, 1, 52)
        bitwise_and_350 = torch.ops.aten.bitwise_and.Tensor(select_401, select_402);  select_401 = select_402 = None
        select_403 = torch.ops.aten.select.int(eq, 1, 55)
        bitwise_and_351 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_350, select_403);  bitwise_and_350 = select_403 = None
        select_404 = torch.ops.aten.select.int(eq, 1, 56)
        bitwise_and_352 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_351, select_404);  bitwise_and_351 = select_404 = None
        select_405 = torch.ops.aten.select.int(eq, 1, 60)
        bitwise_and_353 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_352, select_405);  bitwise_and_352 = select_405 = None
        select_406 = torch.ops.aten.select.int(eq, 1, 62)
        bitwise_and_354 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_353, select_406);  bitwise_and_353 = select_406 = None
        select_407 = torch.ops.aten.select.int(eq, 1, 65)
        bitwise_and_355 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_354, select_407);  bitwise_and_354 = select_407 = None
        select_408 = torch.ops.aten.select.int(eq, 1, 66)
        bitwise_and_356 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_355, select_408);  bitwise_and_355 = select_408 = None
        bitwise_or_50 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_49, bitwise_and_356);  bitwise_or_49 = bitwise_and_356 = None
        select_409 = torch.ops.aten.select.int(eq, 1, 51)
        select_410 = torch.ops.aten.select.int(eq, 1, 53)
        bitwise_and_357 = torch.ops.aten.bitwise_and.Tensor(select_409, select_410);  select_409 = select_410 = None
        select_411 = torch.ops.aten.select.int(eq, 1, 56)
        bitwise_and_358 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_357, select_411);  bitwise_and_357 = select_411 = None
        select_412 = torch.ops.aten.select.int(eq, 1, 57)
        bitwise_and_359 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_358, select_412);  bitwise_and_358 = select_412 = None
        select_413 = torch.ops.aten.select.int(eq, 1, 61)
        bitwise_and_360 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_359, select_413);  bitwise_and_359 = select_413 = None
        select_414 = torch.ops.aten.select.int(eq, 1, 63)
        bitwise_and_361 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_360, select_414);  bitwise_and_360 = select_414 = None
        select_415 = torch.ops.aten.select.int(eq, 1, 66)
        bitwise_and_362 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_361, select_415);  bitwise_and_361 = select_415 = None
        select_416 = torch.ops.aten.select.int(eq, 1, 67)
        bitwise_and_363 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_362, select_416);  bitwise_and_362 = select_416 = None
        bitwise_or_51 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_50, bitwise_and_363);  bitwise_or_50 = bitwise_and_363 = None
        select_417 = torch.ops.aten.select.int(eq, 1, 52)
        select_418 = torch.ops.aten.select.int(eq, 1, 54)
        bitwise_and_364 = torch.ops.aten.bitwise_and.Tensor(select_417, select_418);  select_417 = select_418 = None
        select_419 = torch.ops.aten.select.int(eq, 1, 57)
        bitwise_and_365 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_364, select_419);  bitwise_and_364 = select_419 = None
        select_420 = torch.ops.aten.select.int(eq, 1, 58)
        bitwise_and_366 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_365, select_420);  bitwise_and_365 = select_420 = None
        select_421 = torch.ops.aten.select.int(eq, 1, 62)
        bitwise_and_367 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_366, select_421);  bitwise_and_366 = select_421 = None
        select_422 = torch.ops.aten.select.int(eq, 1, 64)
        bitwise_and_368 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_367, select_422);  bitwise_and_367 = select_422 = None
        select_423 = torch.ops.aten.select.int(eq, 1, 67)
        bitwise_and_369 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_368, select_423);  bitwise_and_368 = select_423 = None
        select_424 = torch.ops.aten.select.int(eq, 1, 68)
        bitwise_and_370 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_369, select_424);  bitwise_and_369 = select_424 = None
        bitwise_or_52 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_51, bitwise_and_370);  bitwise_or_51 = bitwise_and_370 = None
        select_425 = torch.ops.aten.select.int(eq, 1, 53)
        select_426 = torch.ops.aten.select.int(eq, 1, 55)
        bitwise_and_371 = torch.ops.aten.bitwise_and.Tensor(select_425, select_426);  select_425 = select_426 = None
        select_427 = torch.ops.aten.select.int(eq, 1, 58)
        bitwise_and_372 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_371, select_427);  bitwise_and_371 = select_427 = None
        select_428 = torch.ops.aten.select.int(eq, 1, 59)
        bitwise_and_373 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_372, select_428);  bitwise_and_372 = select_428 = None
        select_429 = torch.ops.aten.select.int(eq, 1, 63)
        bitwise_and_374 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_373, select_429);  bitwise_and_373 = select_429 = None
        select_430 = torch.ops.aten.select.int(eq, 1, 65)
        bitwise_and_375 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_374, select_430);  bitwise_and_374 = select_430 = None
        select_431 = torch.ops.aten.select.int(eq, 1, 68)
        bitwise_and_376 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_375, select_431);  bitwise_and_375 = select_431 = None
        select_432 = torch.ops.aten.select.int(eq, 1, 69)
        bitwise_and_377 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_376, select_432);  bitwise_and_376 = select_432 = None
        bitwise_or_53 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_52, bitwise_and_377);  bitwise_or_52 = bitwise_and_377 = None
        select_433 = torch.ops.aten.select.int(eq, 1, 54)
        select_434 = torch.ops.aten.select.int(eq, 1, 56)
        bitwise_and_378 = torch.ops.aten.bitwise_and.Tensor(select_433, select_434);  select_433 = select_434 = None
        select_435 = torch.ops.aten.select.int(eq, 1, 59)
        bitwise_and_379 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_378, select_435);  bitwise_and_378 = select_435 = None
        select_436 = torch.ops.aten.select.int(eq, 1, 60)
        bitwise_and_380 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_379, select_436);  bitwise_and_379 = select_436 = None
        select_437 = torch.ops.aten.select.int(eq, 1, 64)
        bitwise_and_381 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_380, select_437);  bitwise_and_380 = select_437 = None
        select_438 = torch.ops.aten.select.int(eq, 1, 66)
        bitwise_and_382 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_381, select_438);  bitwise_and_381 = select_438 = None
        select_439 = torch.ops.aten.select.int(eq, 1, 69)
        bitwise_and_383 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_382, select_439);  bitwise_and_382 = select_439 = None
        select_440 = torch.ops.aten.select.int(eq, 1, 70)
        bitwise_and_384 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_383, select_440);  bitwise_and_383 = select_440 = None
        bitwise_or_54 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_53, bitwise_and_384);  bitwise_or_53 = bitwise_and_384 = None
        select_441 = torch.ops.aten.select.int(eq, 1, 55)
        select_442 = torch.ops.aten.select.int(eq, 1, 57)
        bitwise_and_385 = torch.ops.aten.bitwise_and.Tensor(select_441, select_442);  select_441 = select_442 = None
        select_443 = torch.ops.aten.select.int(eq, 1, 60)
        bitwise_and_386 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_385, select_443);  bitwise_and_385 = select_443 = None
        select_444 = torch.ops.aten.select.int(eq, 1, 61)
        bitwise_and_387 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_386, select_444);  bitwise_and_386 = select_444 = None
        select_445 = torch.ops.aten.select.int(eq, 1, 65)
        bitwise_and_388 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_387, select_445);  bitwise_and_387 = select_445 = None
        select_446 = torch.ops.aten.select.int(eq, 1, 67)
        bitwise_and_389 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_388, select_446);  bitwise_and_388 = select_446 = None
        select_447 = torch.ops.aten.select.int(eq, 1, 70)
        bitwise_and_390 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_389, select_447);  bitwise_and_389 = select_447 = None
        select_448 = torch.ops.aten.select.int(eq, 1, 71)
        bitwise_and_391 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_390, select_448);  bitwise_and_390 = select_448 = None
        bitwise_or_55 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_54, bitwise_and_391);  bitwise_or_54 = bitwise_and_391 = None
        select_449 = torch.ops.aten.select.int(eq, 1, 56)
        select_450 = torch.ops.aten.select.int(eq, 1, 58)
        bitwise_and_392 = torch.ops.aten.bitwise_and.Tensor(select_449, select_450);  select_449 = select_450 = None
        select_451 = torch.ops.aten.select.int(eq, 1, 61)
        bitwise_and_393 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_392, select_451);  bitwise_and_392 = select_451 = None
        select_452 = torch.ops.aten.select.int(eq, 1, 62)
        bitwise_and_394 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_393, select_452);  bitwise_and_393 = select_452 = None
        select_453 = torch.ops.aten.select.int(eq, 1, 66)
        bitwise_and_395 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_394, select_453);  bitwise_and_394 = select_453 = None
        select_454 = torch.ops.aten.select.int(eq, 1, 68)
        bitwise_and_396 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_395, select_454);  bitwise_and_395 = select_454 = None
        select_455 = torch.ops.aten.select.int(eq, 1, 71)
        bitwise_and_397 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_396, select_455);  bitwise_and_396 = select_455 = None
        select_456 = torch.ops.aten.select.int(eq, 1, 72)
        bitwise_and_398 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_397, select_456);  bitwise_and_397 = select_456 = None
        bitwise_or_56 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_55, bitwise_and_398);  bitwise_or_55 = bitwise_and_398 = None
        select_457 = torch.ops.aten.select.int(eq, 1, 57)
        select_458 = torch.ops.aten.select.int(eq, 1, 59)
        bitwise_and_399 = torch.ops.aten.bitwise_and.Tensor(select_457, select_458);  select_457 = select_458 = None
        select_459 = torch.ops.aten.select.int(eq, 1, 62)
        bitwise_and_400 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_399, select_459);  bitwise_and_399 = select_459 = None
        select_460 = torch.ops.aten.select.int(eq, 1, 63)
        bitwise_and_401 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_400, select_460);  bitwise_and_400 = select_460 = None
        select_461 = torch.ops.aten.select.int(eq, 1, 67)
        bitwise_and_402 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_401, select_461);  bitwise_and_401 = select_461 = None
        select_462 = torch.ops.aten.select.int(eq, 1, 69)
        bitwise_and_403 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_402, select_462);  bitwise_and_402 = select_462 = None
        select_463 = torch.ops.aten.select.int(eq, 1, 72)
        bitwise_and_404 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_403, select_463);  bitwise_and_403 = select_463 = None
        select_464 = torch.ops.aten.select.int(eq, 1, 73)
        bitwise_and_405 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_404, select_464);  bitwise_and_404 = select_464 = None
        bitwise_or_57 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_56, bitwise_and_405);  bitwise_or_56 = bitwise_and_405 = None
        select_465 = torch.ops.aten.select.int(eq, 1, 58)
        select_466 = torch.ops.aten.select.int(eq, 1, 60)
        bitwise_and_406 = torch.ops.aten.bitwise_and.Tensor(select_465, select_466);  select_465 = select_466 = None
        select_467 = torch.ops.aten.select.int(eq, 1, 63)
        bitwise_and_407 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_406, select_467);  bitwise_and_406 = select_467 = None
        select_468 = torch.ops.aten.select.int(eq, 1, 64)
        bitwise_and_408 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_407, select_468);  bitwise_and_407 = select_468 = None
        select_469 = torch.ops.aten.select.int(eq, 1, 68)
        bitwise_and_409 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_408, select_469);  bitwise_and_408 = select_469 = None
        select_470 = torch.ops.aten.select.int(eq, 1, 70)
        bitwise_and_410 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_409, select_470);  bitwise_and_409 = select_470 = None
        select_471 = torch.ops.aten.select.int(eq, 1, 73)
        bitwise_and_411 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_410, select_471);  bitwise_and_410 = select_471 = None
        select_472 = torch.ops.aten.select.int(eq, 1, 74)
        bitwise_and_412 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_411, select_472);  bitwise_and_411 = select_472 = None
        bitwise_or_58 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_57, bitwise_and_412);  bitwise_or_57 = bitwise_and_412 = None
        select_473 = torch.ops.aten.select.int(eq, 1, 59)
        select_474 = torch.ops.aten.select.int(eq, 1, 61)
        bitwise_and_413 = torch.ops.aten.bitwise_and.Tensor(select_473, select_474);  select_473 = select_474 = None
        select_475 = torch.ops.aten.select.int(eq, 1, 64)
        bitwise_and_414 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_413, select_475);  bitwise_and_413 = select_475 = None
        select_476 = torch.ops.aten.select.int(eq, 1, 65)
        bitwise_and_415 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_414, select_476);  bitwise_and_414 = select_476 = None
        select_477 = torch.ops.aten.select.int(eq, 1, 69)
        bitwise_and_416 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_415, select_477);  bitwise_and_415 = select_477 = None
        select_478 = torch.ops.aten.select.int(eq, 1, 71)
        bitwise_and_417 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_416, select_478);  bitwise_and_416 = select_478 = None
        select_479 = torch.ops.aten.select.int(eq, 1, 74)
        bitwise_and_418 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_417, select_479);  bitwise_and_417 = select_479 = None
        select_480 = torch.ops.aten.select.int(eq, 1, 75)
        bitwise_and_419 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_418, select_480);  bitwise_and_418 = select_480 = None
        bitwise_or_59 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_58, bitwise_and_419);  bitwise_or_58 = bitwise_and_419 = None
        select_481 = torch.ops.aten.select.int(eq, 1, 60)
        select_482 = torch.ops.aten.select.int(eq, 1, 62)
        bitwise_and_420 = torch.ops.aten.bitwise_and.Tensor(select_481, select_482);  select_481 = select_482 = None
        select_483 = torch.ops.aten.select.int(eq, 1, 65)
        bitwise_and_421 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_420, select_483);  bitwise_and_420 = select_483 = None
        select_484 = torch.ops.aten.select.int(eq, 1, 66)
        bitwise_and_422 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_421, select_484);  bitwise_and_421 = select_484 = None
        select_485 = torch.ops.aten.select.int(eq, 1, 70)
        bitwise_and_423 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_422, select_485);  bitwise_and_422 = select_485 = None
        select_486 = torch.ops.aten.select.int(eq, 1, 72)
        bitwise_and_424 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_423, select_486);  bitwise_and_423 = select_486 = None
        select_487 = torch.ops.aten.select.int(eq, 1, 75)
        bitwise_and_425 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_424, select_487);  bitwise_and_424 = select_487 = None
        select_488 = torch.ops.aten.select.int(eq, 1, 76)
        bitwise_and_426 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_425, select_488);  bitwise_and_425 = select_488 = None
        bitwise_or_60 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_59, bitwise_and_426);  bitwise_or_59 = bitwise_and_426 = None
        select_489 = torch.ops.aten.select.int(eq, 1, 61)
        select_490 = torch.ops.aten.select.int(eq, 1, 63)
        bitwise_and_427 = torch.ops.aten.bitwise_and.Tensor(select_489, select_490);  select_489 = select_490 = None
        select_491 = torch.ops.aten.select.int(eq, 1, 66)
        bitwise_and_428 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_427, select_491);  bitwise_and_427 = select_491 = None
        select_492 = torch.ops.aten.select.int(eq, 1, 67)
        bitwise_and_429 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_428, select_492);  bitwise_and_428 = select_492 = None
        select_493 = torch.ops.aten.select.int(eq, 1, 71)
        bitwise_and_430 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_429, select_493);  bitwise_and_429 = select_493 = None
        select_494 = torch.ops.aten.select.int(eq, 1, 73)
        bitwise_and_431 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_430, select_494);  bitwise_and_430 = select_494 = None
        select_495 = torch.ops.aten.select.int(eq, 1, 76)
        bitwise_and_432 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_431, select_495);  bitwise_and_431 = select_495 = None
        select_496 = torch.ops.aten.select.int(eq, 1, 77)
        bitwise_and_433 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_432, select_496);  bitwise_and_432 = select_496 = None
        bitwise_or_61 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_60, bitwise_and_433);  bitwise_or_60 = bitwise_and_433 = None
        select_497 = torch.ops.aten.select.int(eq, 1, 62)
        select_498 = torch.ops.aten.select.int(eq, 1, 64)
        bitwise_and_434 = torch.ops.aten.bitwise_and.Tensor(select_497, select_498);  select_497 = select_498 = None
        select_499 = torch.ops.aten.select.int(eq, 1, 67)
        bitwise_and_435 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_434, select_499);  bitwise_and_434 = select_499 = None
        select_500 = torch.ops.aten.select.int(eq, 1, 68)
        bitwise_and_436 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_435, select_500);  bitwise_and_435 = select_500 = None
        select_501 = torch.ops.aten.select.int(eq, 1, 72)
        bitwise_and_437 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_436, select_501);  bitwise_and_436 = select_501 = None
        select_502 = torch.ops.aten.select.int(eq, 1, 74)
        bitwise_and_438 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_437, select_502);  bitwise_and_437 = select_502 = None
        select_503 = torch.ops.aten.select.int(eq, 1, 77)
        bitwise_and_439 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_438, select_503);  bitwise_and_438 = select_503 = None
        select_504 = torch.ops.aten.select.int(eq, 1, 78)
        bitwise_and_440 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_439, select_504);  bitwise_and_439 = select_504 = None
        bitwise_or_62 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_61, bitwise_and_440);  bitwise_or_61 = bitwise_and_440 = None
        select_505 = torch.ops.aten.select.int(eq, 1, 63)
        select_506 = torch.ops.aten.select.int(eq, 1, 65)
        bitwise_and_441 = torch.ops.aten.bitwise_and.Tensor(select_505, select_506);  select_505 = select_506 = None
        select_507 = torch.ops.aten.select.int(eq, 1, 68)
        bitwise_and_442 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_441, select_507);  bitwise_and_441 = select_507 = None
        select_508 = torch.ops.aten.select.int(eq, 1, 69)
        bitwise_and_443 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_442, select_508);  bitwise_and_442 = select_508 = None
        select_509 = torch.ops.aten.select.int(eq, 1, 73)
        bitwise_and_444 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_443, select_509);  bitwise_and_443 = select_509 = None
        select_510 = torch.ops.aten.select.int(eq, 1, 75)
        bitwise_and_445 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_444, select_510);  bitwise_and_444 = select_510 = None
        select_511 = torch.ops.aten.select.int(eq, 1, 78)
        bitwise_and_446 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_445, select_511);  bitwise_and_445 = select_511 = None
        select_512 = torch.ops.aten.select.int(eq, 1, 79)
        bitwise_and_447 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_446, select_512);  bitwise_and_446 = select_512 = None
        bitwise_or_63 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_62, bitwise_and_447);  bitwise_or_62 = bitwise_and_447 = None
        select_513 = torch.ops.aten.select.int(eq, 1, 64)
        select_514 = torch.ops.aten.select.int(eq, 1, 66)
        bitwise_and_448 = torch.ops.aten.bitwise_and.Tensor(select_513, select_514);  select_513 = select_514 = None
        select_515 = torch.ops.aten.select.int(eq, 1, 69)
        bitwise_and_449 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_448, select_515);  bitwise_and_448 = select_515 = None
        select_516 = torch.ops.aten.select.int(eq, 1, 70)
        bitwise_and_450 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_449, select_516);  bitwise_and_449 = select_516 = None
        select_517 = torch.ops.aten.select.int(eq, 1, 74)
        bitwise_and_451 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_450, select_517);  bitwise_and_450 = select_517 = None
        select_518 = torch.ops.aten.select.int(eq, 1, 76)
        bitwise_and_452 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_451, select_518);  bitwise_and_451 = select_518 = None
        select_519 = torch.ops.aten.select.int(eq, 1, 79)
        bitwise_and_453 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_452, select_519);  bitwise_and_452 = select_519 = None
        select_520 = torch.ops.aten.select.int(eq, 1, 80)
        bitwise_and_454 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_453, select_520);  bitwise_and_453 = select_520 = None
        bitwise_or_64 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_63, bitwise_and_454);  bitwise_or_63 = bitwise_and_454 = None
        select_521 = torch.ops.aten.select.int(eq, 1, 65)
        select_522 = torch.ops.aten.select.int(eq, 1, 67)
        bitwise_and_455 = torch.ops.aten.bitwise_and.Tensor(select_521, select_522);  select_521 = select_522 = None
        select_523 = torch.ops.aten.select.int(eq, 1, 70)
        bitwise_and_456 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_455, select_523);  bitwise_and_455 = select_523 = None
        select_524 = torch.ops.aten.select.int(eq, 1, 71)
        bitwise_and_457 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_456, select_524);  bitwise_and_456 = select_524 = None
        select_525 = torch.ops.aten.select.int(eq, 1, 75)
        bitwise_and_458 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_457, select_525);  bitwise_and_457 = select_525 = None
        select_526 = torch.ops.aten.select.int(eq, 1, 77)
        bitwise_and_459 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_458, select_526);  bitwise_and_458 = select_526 = None
        select_527 = torch.ops.aten.select.int(eq, 1, 80)
        bitwise_and_460 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_459, select_527);  bitwise_and_459 = select_527 = None
        select_528 = torch.ops.aten.select.int(eq, 1, 81)
        bitwise_and_461 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_460, select_528);  bitwise_and_460 = select_528 = None
        bitwise_or_65 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_64, bitwise_and_461);  bitwise_or_64 = bitwise_and_461 = None
        select_529 = torch.ops.aten.select.int(eq, 1, 66)
        select_530 = torch.ops.aten.select.int(eq, 1, 68)
        bitwise_and_462 = torch.ops.aten.bitwise_and.Tensor(select_529, select_530);  select_529 = select_530 = None
        select_531 = torch.ops.aten.select.int(eq, 1, 71)
        bitwise_and_463 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_462, select_531);  bitwise_and_462 = select_531 = None
        select_532 = torch.ops.aten.select.int(eq, 1, 72)
        bitwise_and_464 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_463, select_532);  bitwise_and_463 = select_532 = None
        select_533 = torch.ops.aten.select.int(eq, 1, 76)
        bitwise_and_465 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_464, select_533);  bitwise_and_464 = select_533 = None
        select_534 = torch.ops.aten.select.int(eq, 1, 78)
        bitwise_and_466 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_465, select_534);  bitwise_and_465 = select_534 = None
        select_535 = torch.ops.aten.select.int(eq, 1, 81)
        bitwise_and_467 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_466, select_535);  bitwise_and_466 = select_535 = None
        select_536 = torch.ops.aten.select.int(eq, 1, 82)
        bitwise_and_468 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_467, select_536);  bitwise_and_467 = select_536 = None
        bitwise_or_66 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_65, bitwise_and_468);  bitwise_or_65 = bitwise_and_468 = None
        select_537 = torch.ops.aten.select.int(eq, 1, 67)
        select_538 = torch.ops.aten.select.int(eq, 1, 69)
        bitwise_and_469 = torch.ops.aten.bitwise_and.Tensor(select_537, select_538);  select_537 = select_538 = None
        select_539 = torch.ops.aten.select.int(eq, 1, 72)
        bitwise_and_470 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_469, select_539);  bitwise_and_469 = select_539 = None
        select_540 = torch.ops.aten.select.int(eq, 1, 73)
        bitwise_and_471 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_470, select_540);  bitwise_and_470 = select_540 = None
        select_541 = torch.ops.aten.select.int(eq, 1, 77)
        bitwise_and_472 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_471, select_541);  bitwise_and_471 = select_541 = None
        select_542 = torch.ops.aten.select.int(eq, 1, 79)
        bitwise_and_473 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_472, select_542);  bitwise_and_472 = select_542 = None
        select_543 = torch.ops.aten.select.int(eq, 1, 82)
        bitwise_and_474 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_473, select_543);  bitwise_and_473 = select_543 = None
        select_544 = torch.ops.aten.select.int(eq, 1, 83)
        bitwise_and_475 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_474, select_544);  bitwise_and_474 = select_544 = None
        bitwise_or_67 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_66, bitwise_and_475);  bitwise_or_66 = bitwise_and_475 = None
        select_545 = torch.ops.aten.select.int(eq, 1, 68)
        select_546 = torch.ops.aten.select.int(eq, 1, 70)
        bitwise_and_476 = torch.ops.aten.bitwise_and.Tensor(select_545, select_546);  select_545 = select_546 = None
        select_547 = torch.ops.aten.select.int(eq, 1, 73)
        bitwise_and_477 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_476, select_547);  bitwise_and_476 = select_547 = None
        select_548 = torch.ops.aten.select.int(eq, 1, 74)
        bitwise_and_478 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_477, select_548);  bitwise_and_477 = select_548 = None
        select_549 = torch.ops.aten.select.int(eq, 1, 78)
        bitwise_and_479 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_478, select_549);  bitwise_and_478 = select_549 = None
        select_550 = torch.ops.aten.select.int(eq, 1, 80)
        bitwise_and_480 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_479, select_550);  bitwise_and_479 = select_550 = None
        select_551 = torch.ops.aten.select.int(eq, 1, 83)
        bitwise_and_481 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_480, select_551);  bitwise_and_480 = select_551 = None
        select_552 = torch.ops.aten.select.int(eq, 1, 84)
        bitwise_and_482 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_481, select_552);  bitwise_and_481 = select_552 = None
        bitwise_or_68 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_67, bitwise_and_482);  bitwise_or_67 = bitwise_and_482 = None
        select_553 = torch.ops.aten.select.int(eq, 1, 69)
        select_554 = torch.ops.aten.select.int(eq, 1, 71)
        bitwise_and_483 = torch.ops.aten.bitwise_and.Tensor(select_553, select_554);  select_553 = select_554 = None
        select_555 = torch.ops.aten.select.int(eq, 1, 74)
        bitwise_and_484 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_483, select_555);  bitwise_and_483 = select_555 = None
        select_556 = torch.ops.aten.select.int(eq, 1, 75)
        bitwise_and_485 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_484, select_556);  bitwise_and_484 = select_556 = None
        select_557 = torch.ops.aten.select.int(eq, 1, 79)
        bitwise_and_486 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_485, select_557);  bitwise_and_485 = select_557 = None
        select_558 = torch.ops.aten.select.int(eq, 1, 81)
        bitwise_and_487 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_486, select_558);  bitwise_and_486 = select_558 = None
        select_559 = torch.ops.aten.select.int(eq, 1, 84)
        bitwise_and_488 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_487, select_559);  bitwise_and_487 = select_559 = None
        select_560 = torch.ops.aten.select.int(eq, 1, 85)
        bitwise_and_489 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_488, select_560);  bitwise_and_488 = select_560 = None
        bitwise_or_69 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_68, bitwise_and_489);  bitwise_or_68 = bitwise_and_489 = None
        select_561 = torch.ops.aten.select.int(eq, 1, 70)
        select_562 = torch.ops.aten.select.int(eq, 1, 72)
        bitwise_and_490 = torch.ops.aten.bitwise_and.Tensor(select_561, select_562);  select_561 = select_562 = None
        select_563 = torch.ops.aten.select.int(eq, 1, 75)
        bitwise_and_491 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_490, select_563);  bitwise_and_490 = select_563 = None
        select_564 = torch.ops.aten.select.int(eq, 1, 76)
        bitwise_and_492 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_491, select_564);  bitwise_and_491 = select_564 = None
        select_565 = torch.ops.aten.select.int(eq, 1, 80)
        bitwise_and_493 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_492, select_565);  bitwise_and_492 = select_565 = None
        select_566 = torch.ops.aten.select.int(eq, 1, 82)
        bitwise_and_494 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_493, select_566);  bitwise_and_493 = select_566 = None
        select_567 = torch.ops.aten.select.int(eq, 1, 85)
        bitwise_and_495 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_494, select_567);  bitwise_and_494 = select_567 = None
        select_568 = torch.ops.aten.select.int(eq, 1, 86)
        bitwise_and_496 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_495, select_568);  bitwise_and_495 = select_568 = None
        bitwise_or_70 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_69, bitwise_and_496);  bitwise_or_69 = bitwise_and_496 = None
        select_569 = torch.ops.aten.select.int(eq, 1, 71)
        select_570 = torch.ops.aten.select.int(eq, 1, 73)
        bitwise_and_497 = torch.ops.aten.bitwise_and.Tensor(select_569, select_570);  select_569 = select_570 = None
        select_571 = torch.ops.aten.select.int(eq, 1, 76)
        bitwise_and_498 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_497, select_571);  bitwise_and_497 = select_571 = None
        select_572 = torch.ops.aten.select.int(eq, 1, 77)
        bitwise_and_499 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_498, select_572);  bitwise_and_498 = select_572 = None
        select_573 = torch.ops.aten.select.int(eq, 1, 81)
        bitwise_and_500 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_499, select_573);  bitwise_and_499 = select_573 = None
        select_574 = torch.ops.aten.select.int(eq, 1, 83)
        bitwise_and_501 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_500, select_574);  bitwise_and_500 = select_574 = None
        select_575 = torch.ops.aten.select.int(eq, 1, 86)
        bitwise_and_502 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_501, select_575);  bitwise_and_501 = select_575 = None
        select_576 = torch.ops.aten.select.int(eq, 1, 87)
        bitwise_and_503 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_502, select_576);  bitwise_and_502 = select_576 = None
        bitwise_or_71 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_70, bitwise_and_503);  bitwise_or_70 = bitwise_and_503 = None
        select_577 = torch.ops.aten.select.int(eq, 1, 72)
        select_578 = torch.ops.aten.select.int(eq, 1, 74)
        bitwise_and_504 = torch.ops.aten.bitwise_and.Tensor(select_577, select_578);  select_577 = select_578 = None
        select_579 = torch.ops.aten.select.int(eq, 1, 77)
        bitwise_and_505 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_504, select_579);  bitwise_and_504 = select_579 = None
        select_580 = torch.ops.aten.select.int(eq, 1, 78)
        bitwise_and_506 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_505, select_580);  bitwise_and_505 = select_580 = None
        select_581 = torch.ops.aten.select.int(eq, 1, 82)
        bitwise_and_507 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_506, select_581);  bitwise_and_506 = select_581 = None
        select_582 = torch.ops.aten.select.int(eq, 1, 84)
        bitwise_and_508 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_507, select_582);  bitwise_and_507 = select_582 = None
        select_583 = torch.ops.aten.select.int(eq, 1, 87)
        bitwise_and_509 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_508, select_583);  bitwise_and_508 = select_583 = None
        select_584 = torch.ops.aten.select.int(eq, 1, 88)
        bitwise_and_510 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_509, select_584);  bitwise_and_509 = select_584 = None
        bitwise_or_72 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_71, bitwise_and_510);  bitwise_or_71 = bitwise_and_510 = None
        select_585 = torch.ops.aten.select.int(eq, 1, 73)
        select_586 = torch.ops.aten.select.int(eq, 1, 75)
        bitwise_and_511 = torch.ops.aten.bitwise_and.Tensor(select_585, select_586);  select_585 = select_586 = None
        select_587 = torch.ops.aten.select.int(eq, 1, 78)
        bitwise_and_512 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_511, select_587);  bitwise_and_511 = select_587 = None
        select_588 = torch.ops.aten.select.int(eq, 1, 79)
        bitwise_and_513 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_512, select_588);  bitwise_and_512 = select_588 = None
        select_589 = torch.ops.aten.select.int(eq, 1, 83)
        bitwise_and_514 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_513, select_589);  bitwise_and_513 = select_589 = None
        select_590 = torch.ops.aten.select.int(eq, 1, 85)
        bitwise_and_515 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_514, select_590);  bitwise_and_514 = select_590 = None
        select_591 = torch.ops.aten.select.int(eq, 1, 88)
        bitwise_and_516 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_515, select_591);  bitwise_and_515 = select_591 = None
        select_592 = torch.ops.aten.select.int(eq, 1, 89)
        bitwise_and_517 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_516, select_592);  bitwise_and_516 = select_592 = None
        bitwise_or_73 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_72, bitwise_and_517);  bitwise_or_72 = bitwise_and_517 = None
        select_593 = torch.ops.aten.select.int(eq, 1, 74)
        select_594 = torch.ops.aten.select.int(eq, 1, 76)
        bitwise_and_518 = torch.ops.aten.bitwise_and.Tensor(select_593, select_594);  select_593 = select_594 = None
        select_595 = torch.ops.aten.select.int(eq, 1, 79)
        bitwise_and_519 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_518, select_595);  bitwise_and_518 = select_595 = None
        select_596 = torch.ops.aten.select.int(eq, 1, 80)
        bitwise_and_520 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_519, select_596);  bitwise_and_519 = select_596 = None
        select_597 = torch.ops.aten.select.int(eq, 1, 84)
        bitwise_and_521 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_520, select_597);  bitwise_and_520 = select_597 = None
        select_598 = torch.ops.aten.select.int(eq, 1, 86)
        bitwise_and_522 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_521, select_598);  bitwise_and_521 = select_598 = None
        select_599 = torch.ops.aten.select.int(eq, 1, 89)
        bitwise_and_523 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_522, select_599);  bitwise_and_522 = select_599 = None
        select_600 = torch.ops.aten.select.int(eq, 1, 90)
        bitwise_and_524 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_523, select_600);  bitwise_and_523 = select_600 = None
        bitwise_or_74 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_73, bitwise_and_524);  bitwise_or_73 = bitwise_and_524 = None
        select_601 = torch.ops.aten.select.int(eq, 1, 75)
        select_602 = torch.ops.aten.select.int(eq, 1, 77)
        bitwise_and_525 = torch.ops.aten.bitwise_and.Tensor(select_601, select_602);  select_601 = select_602 = None
        select_603 = torch.ops.aten.select.int(eq, 1, 80)
        bitwise_and_526 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_525, select_603);  bitwise_and_525 = select_603 = None
        select_604 = torch.ops.aten.select.int(eq, 1, 81)
        bitwise_and_527 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_526, select_604);  bitwise_and_526 = select_604 = None
        select_605 = torch.ops.aten.select.int(eq, 1, 85)
        bitwise_and_528 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_527, select_605);  bitwise_and_527 = select_605 = None
        select_606 = torch.ops.aten.select.int(eq, 1, 87)
        bitwise_and_529 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_528, select_606);  bitwise_and_528 = select_606 = None
        select_607 = torch.ops.aten.select.int(eq, 1, 90)
        bitwise_and_530 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_529, select_607);  bitwise_and_529 = select_607 = None
        select_608 = torch.ops.aten.select.int(eq, 1, 91)
        bitwise_and_531 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_530, select_608);  bitwise_and_530 = select_608 = None
        bitwise_or_75 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_74, bitwise_and_531);  bitwise_or_74 = bitwise_and_531 = None
        select_609 = torch.ops.aten.select.int(eq, 1, 76)
        select_610 = torch.ops.aten.select.int(eq, 1, 78)
        bitwise_and_532 = torch.ops.aten.bitwise_and.Tensor(select_609, select_610);  select_609 = select_610 = None
        select_611 = torch.ops.aten.select.int(eq, 1, 81)
        bitwise_and_533 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_532, select_611);  bitwise_and_532 = select_611 = None
        select_612 = torch.ops.aten.select.int(eq, 1, 82)
        bitwise_and_534 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_533, select_612);  bitwise_and_533 = select_612 = None
        select_613 = torch.ops.aten.select.int(eq, 1, 86)
        bitwise_and_535 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_534, select_613);  bitwise_and_534 = select_613 = None
        select_614 = torch.ops.aten.select.int(eq, 1, 88)
        bitwise_and_536 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_535, select_614);  bitwise_and_535 = select_614 = None
        select_615 = torch.ops.aten.select.int(eq, 1, 91)
        bitwise_and_537 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_536, select_615);  bitwise_and_536 = select_615 = None
        select_616 = torch.ops.aten.select.int(eq, 1, 92)
        bitwise_and_538 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_537, select_616);  bitwise_and_537 = select_616 = None
        bitwise_or_76 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_75, bitwise_and_538);  bitwise_or_75 = bitwise_and_538 = None
        select_617 = torch.ops.aten.select.int(eq, 1, 77)
        select_618 = torch.ops.aten.select.int(eq, 1, 79)
        bitwise_and_539 = torch.ops.aten.bitwise_and.Tensor(select_617, select_618);  select_617 = select_618 = None
        select_619 = torch.ops.aten.select.int(eq, 1, 82)
        bitwise_and_540 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_539, select_619);  bitwise_and_539 = select_619 = None
        select_620 = torch.ops.aten.select.int(eq, 1, 83)
        bitwise_and_541 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_540, select_620);  bitwise_and_540 = select_620 = None
        select_621 = torch.ops.aten.select.int(eq, 1, 87)
        bitwise_and_542 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_541, select_621);  bitwise_and_541 = select_621 = None
        select_622 = torch.ops.aten.select.int(eq, 1, 89)
        bitwise_and_543 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_542, select_622);  bitwise_and_542 = select_622 = None
        select_623 = torch.ops.aten.select.int(eq, 1, 92)
        bitwise_and_544 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_543, select_623);  bitwise_and_543 = select_623 = None
        select_624 = torch.ops.aten.select.int(eq, 1, 93)
        bitwise_and_545 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_544, select_624);  bitwise_and_544 = select_624 = None
        bitwise_or_77 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_76, bitwise_and_545);  bitwise_or_76 = bitwise_and_545 = None
        select_625 = torch.ops.aten.select.int(eq, 1, 78)
        select_626 = torch.ops.aten.select.int(eq, 1, 80)
        bitwise_and_546 = torch.ops.aten.bitwise_and.Tensor(select_625, select_626);  select_625 = select_626 = None
        select_627 = torch.ops.aten.select.int(eq, 1, 83)
        bitwise_and_547 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_546, select_627);  bitwise_and_546 = select_627 = None
        select_628 = torch.ops.aten.select.int(eq, 1, 84)
        bitwise_and_548 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_547, select_628);  bitwise_and_547 = select_628 = None
        select_629 = torch.ops.aten.select.int(eq, 1, 88)
        bitwise_and_549 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_548, select_629);  bitwise_and_548 = select_629 = None
        select_630 = torch.ops.aten.select.int(eq, 1, 90)
        bitwise_and_550 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_549, select_630);  bitwise_and_549 = select_630 = None
        select_631 = torch.ops.aten.select.int(eq, 1, 93)
        bitwise_and_551 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_550, select_631);  bitwise_and_550 = select_631 = None
        select_632 = torch.ops.aten.select.int(eq, 1, 94)
        bitwise_and_552 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_551, select_632);  bitwise_and_551 = select_632 = None
        bitwise_or_78 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_77, bitwise_and_552);  bitwise_or_77 = bitwise_and_552 = None
        select_633 = torch.ops.aten.select.int(eq, 1, 79)
        select_634 = torch.ops.aten.select.int(eq, 1, 81)
        bitwise_and_553 = torch.ops.aten.bitwise_and.Tensor(select_633, select_634);  select_633 = select_634 = None
        select_635 = torch.ops.aten.select.int(eq, 1, 84)
        bitwise_and_554 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_553, select_635);  bitwise_and_553 = select_635 = None
        select_636 = torch.ops.aten.select.int(eq, 1, 85)
        bitwise_and_555 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_554, select_636);  bitwise_and_554 = select_636 = None
        select_637 = torch.ops.aten.select.int(eq, 1, 89)
        bitwise_and_556 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_555, select_637);  bitwise_and_555 = select_637 = None
        select_638 = torch.ops.aten.select.int(eq, 1, 91)
        bitwise_and_557 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_556, select_638);  bitwise_and_556 = select_638 = None
        select_639 = torch.ops.aten.select.int(eq, 1, 94)
        bitwise_and_558 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_557, select_639);  bitwise_and_557 = select_639 = None
        select_640 = torch.ops.aten.select.int(eq, 1, 95)
        bitwise_and_559 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_558, select_640);  bitwise_and_558 = select_640 = None
        bitwise_or_79 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_78, bitwise_and_559);  bitwise_or_78 = bitwise_and_559 = None
        select_641 = torch.ops.aten.select.int(eq, 1, 80)
        select_642 = torch.ops.aten.select.int(eq, 1, 82)
        bitwise_and_560 = torch.ops.aten.bitwise_and.Tensor(select_641, select_642);  select_641 = select_642 = None
        select_643 = torch.ops.aten.select.int(eq, 1, 85)
        bitwise_and_561 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_560, select_643);  bitwise_and_560 = select_643 = None
        select_644 = torch.ops.aten.select.int(eq, 1, 86)
        bitwise_and_562 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_561, select_644);  bitwise_and_561 = select_644 = None
        select_645 = torch.ops.aten.select.int(eq, 1, 90)
        bitwise_and_563 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_562, select_645);  bitwise_and_562 = select_645 = None
        select_646 = torch.ops.aten.select.int(eq, 1, 92)
        bitwise_and_564 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_563, select_646);  bitwise_and_563 = select_646 = None
        select_647 = torch.ops.aten.select.int(eq, 1, 95)
        bitwise_and_565 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_564, select_647);  bitwise_and_564 = select_647 = None
        select_648 = torch.ops.aten.select.int(eq, 1, 96)
        bitwise_and_566 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_565, select_648);  bitwise_and_565 = select_648 = None
        bitwise_or_80 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_79, bitwise_and_566);  bitwise_or_79 = bitwise_and_566 = None
        select_649 = torch.ops.aten.select.int(eq, 1, 81)
        select_650 = torch.ops.aten.select.int(eq, 1, 83)
        bitwise_and_567 = torch.ops.aten.bitwise_and.Tensor(select_649, select_650);  select_649 = select_650 = None
        select_651 = torch.ops.aten.select.int(eq, 1, 86)
        bitwise_and_568 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_567, select_651);  bitwise_and_567 = select_651 = None
        select_652 = torch.ops.aten.select.int(eq, 1, 87)
        bitwise_and_569 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_568, select_652);  bitwise_and_568 = select_652 = None
        select_653 = torch.ops.aten.select.int(eq, 1, 91)
        bitwise_and_570 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_569, select_653);  bitwise_and_569 = select_653 = None
        select_654 = torch.ops.aten.select.int(eq, 1, 93)
        bitwise_and_571 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_570, select_654);  bitwise_and_570 = select_654 = None
        select_655 = torch.ops.aten.select.int(eq, 1, 96)
        bitwise_and_572 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_571, select_655);  bitwise_and_571 = select_655 = None
        select_656 = torch.ops.aten.select.int(eq, 1, 97)
        bitwise_and_573 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_572, select_656);  bitwise_and_572 = select_656 = None
        bitwise_or_81 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_80, bitwise_and_573);  bitwise_or_80 = bitwise_and_573 = None
        select_657 = torch.ops.aten.select.int(eq, 1, 82)
        select_658 = torch.ops.aten.select.int(eq, 1, 84)
        bitwise_and_574 = torch.ops.aten.bitwise_and.Tensor(select_657, select_658);  select_657 = select_658 = None
        select_659 = torch.ops.aten.select.int(eq, 1, 87)
        bitwise_and_575 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_574, select_659);  bitwise_and_574 = select_659 = None
        select_660 = torch.ops.aten.select.int(eq, 1, 88)
        bitwise_and_576 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_575, select_660);  bitwise_and_575 = select_660 = None
        select_661 = torch.ops.aten.select.int(eq, 1, 92)
        bitwise_and_577 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_576, select_661);  bitwise_and_576 = select_661 = None
        select_662 = torch.ops.aten.select.int(eq, 1, 94)
        bitwise_and_578 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_577, select_662);  bitwise_and_577 = select_662 = None
        select_663 = torch.ops.aten.select.int(eq, 1, 97)
        bitwise_and_579 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_578, select_663);  bitwise_and_578 = select_663 = None
        select_664 = torch.ops.aten.select.int(eq, 1, 98)
        bitwise_and_580 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_579, select_664);  bitwise_and_579 = select_664 = None
        bitwise_or_82 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_81, bitwise_and_580);  bitwise_or_81 = bitwise_and_580 = None
        select_665 = torch.ops.aten.select.int(eq, 1, 83)
        select_666 = torch.ops.aten.select.int(eq, 1, 85)
        bitwise_and_581 = torch.ops.aten.bitwise_and.Tensor(select_665, select_666);  select_665 = select_666 = None
        select_667 = torch.ops.aten.select.int(eq, 1, 88)
        bitwise_and_582 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_581, select_667);  bitwise_and_581 = select_667 = None
        select_668 = torch.ops.aten.select.int(eq, 1, 89)
        bitwise_and_583 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_582, select_668);  bitwise_and_582 = select_668 = None
        select_669 = torch.ops.aten.select.int(eq, 1, 93)
        bitwise_and_584 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_583, select_669);  bitwise_and_583 = select_669 = None
        select_670 = torch.ops.aten.select.int(eq, 1, 95)
        bitwise_and_585 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_584, select_670);  bitwise_and_584 = select_670 = None
        select_671 = torch.ops.aten.select.int(eq, 1, 98)
        bitwise_and_586 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_585, select_671);  bitwise_and_585 = select_671 = None
        select_672 = torch.ops.aten.select.int(eq, 1, 99)
        bitwise_and_587 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_586, select_672);  bitwise_and_586 = select_672 = None
        bitwise_or_83 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_82, bitwise_and_587);  bitwise_or_82 = bitwise_and_587 = None
        select_673 = torch.ops.aten.select.int(eq, 1, 84)
        select_674 = torch.ops.aten.select.int(eq, 1, 86)
        bitwise_and_588 = torch.ops.aten.bitwise_and.Tensor(select_673, select_674);  select_673 = select_674 = None
        select_675 = torch.ops.aten.select.int(eq, 1, 89)
        bitwise_and_589 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_588, select_675);  bitwise_and_588 = select_675 = None
        select_676 = torch.ops.aten.select.int(eq, 1, 90)
        bitwise_and_590 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_589, select_676);  bitwise_and_589 = select_676 = None
        select_677 = torch.ops.aten.select.int(eq, 1, 94)
        bitwise_and_591 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_590, select_677);  bitwise_and_590 = select_677 = None
        select_678 = torch.ops.aten.select.int(eq, 1, 96)
        bitwise_and_592 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_591, select_678);  bitwise_and_591 = select_678 = None
        select_679 = torch.ops.aten.select.int(eq, 1, 99)
        bitwise_and_593 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_592, select_679);  bitwise_and_592 = select_679 = None
        select_680 = torch.ops.aten.select.int(eq, 1, 100)
        bitwise_and_594 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_593, select_680);  bitwise_and_593 = select_680 = None
        bitwise_or_84 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_83, bitwise_and_594);  bitwise_or_83 = bitwise_and_594 = None
        select_681 = torch.ops.aten.select.int(eq, 1, 85)
        select_682 = torch.ops.aten.select.int(eq, 1, 87)
        bitwise_and_595 = torch.ops.aten.bitwise_and.Tensor(select_681, select_682);  select_681 = select_682 = None
        select_683 = torch.ops.aten.select.int(eq, 1, 90)
        bitwise_and_596 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_595, select_683);  bitwise_and_595 = select_683 = None
        select_684 = torch.ops.aten.select.int(eq, 1, 91)
        bitwise_and_597 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_596, select_684);  bitwise_and_596 = select_684 = None
        select_685 = torch.ops.aten.select.int(eq, 1, 95)
        bitwise_and_598 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_597, select_685);  bitwise_and_597 = select_685 = None
        select_686 = torch.ops.aten.select.int(eq, 1, 97)
        bitwise_and_599 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_598, select_686);  bitwise_and_598 = select_686 = None
        select_687 = torch.ops.aten.select.int(eq, 1, 100)
        bitwise_and_600 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_599, select_687);  bitwise_and_599 = select_687 = None
        select_688 = torch.ops.aten.select.int(eq, 1, 101)
        bitwise_and_601 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_600, select_688);  bitwise_and_600 = select_688 = None
        bitwise_or_85 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_84, bitwise_and_601);  bitwise_or_84 = bitwise_and_601 = None
        select_689 = torch.ops.aten.select.int(eq, 1, 86)
        select_690 = torch.ops.aten.select.int(eq, 1, 88)
        bitwise_and_602 = torch.ops.aten.bitwise_and.Tensor(select_689, select_690);  select_689 = select_690 = None
        select_691 = torch.ops.aten.select.int(eq, 1, 91)
        bitwise_and_603 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_602, select_691);  bitwise_and_602 = select_691 = None
        select_692 = torch.ops.aten.select.int(eq, 1, 92)
        bitwise_and_604 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_603, select_692);  bitwise_and_603 = select_692 = None
        select_693 = torch.ops.aten.select.int(eq, 1, 96)
        bitwise_and_605 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_604, select_693);  bitwise_and_604 = select_693 = None
        select_694 = torch.ops.aten.select.int(eq, 1, 98)
        bitwise_and_606 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_605, select_694);  bitwise_and_605 = select_694 = None
        select_695 = torch.ops.aten.select.int(eq, 1, 101)
        bitwise_and_607 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_606, select_695);  bitwise_and_606 = select_695 = None
        select_696 = torch.ops.aten.select.int(eq, 1, 102)
        bitwise_and_608 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_607, select_696);  bitwise_and_607 = select_696 = None
        bitwise_or_86 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_85, bitwise_and_608);  bitwise_or_85 = bitwise_and_608 = None
        select_697 = torch.ops.aten.select.int(eq, 1, 87)
        select_698 = torch.ops.aten.select.int(eq, 1, 89)
        bitwise_and_609 = torch.ops.aten.bitwise_and.Tensor(select_697, select_698);  select_697 = select_698 = None
        select_699 = torch.ops.aten.select.int(eq, 1, 92)
        bitwise_and_610 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_609, select_699);  bitwise_and_609 = select_699 = None
        select_700 = torch.ops.aten.select.int(eq, 1, 93)
        bitwise_and_611 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_610, select_700);  bitwise_and_610 = select_700 = None
        select_701 = torch.ops.aten.select.int(eq, 1, 97)
        bitwise_and_612 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_611, select_701);  bitwise_and_611 = select_701 = None
        select_702 = torch.ops.aten.select.int(eq, 1, 99)
        bitwise_and_613 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_612, select_702);  bitwise_and_612 = select_702 = None
        select_703 = torch.ops.aten.select.int(eq, 1, 102)
        bitwise_and_614 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_613, select_703);  bitwise_and_613 = select_703 = None
        select_704 = torch.ops.aten.select.int(eq, 1, 103)
        bitwise_and_615 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_614, select_704);  bitwise_and_614 = select_704 = None
        bitwise_or_87 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_86, bitwise_and_615);  bitwise_or_86 = bitwise_and_615 = None
        select_705 = torch.ops.aten.select.int(eq, 1, 88)
        select_706 = torch.ops.aten.select.int(eq, 1, 90)
        bitwise_and_616 = torch.ops.aten.bitwise_and.Tensor(select_705, select_706);  select_705 = select_706 = None
        select_707 = torch.ops.aten.select.int(eq, 1, 93)
        bitwise_and_617 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_616, select_707);  bitwise_and_616 = select_707 = None
        select_708 = torch.ops.aten.select.int(eq, 1, 94)
        bitwise_and_618 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_617, select_708);  bitwise_and_617 = select_708 = None
        select_709 = torch.ops.aten.select.int(eq, 1, 98)
        bitwise_and_619 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_618, select_709);  bitwise_and_618 = select_709 = None
        select_710 = torch.ops.aten.select.int(eq, 1, 100)
        bitwise_and_620 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_619, select_710);  bitwise_and_619 = select_710 = None
        select_711 = torch.ops.aten.select.int(eq, 1, 103)
        bitwise_and_621 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_620, select_711);  bitwise_and_620 = select_711 = None
        select_712 = torch.ops.aten.select.int(eq, 1, 104)
        bitwise_and_622 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_621, select_712);  bitwise_and_621 = select_712 = None
        bitwise_or_88 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_87, bitwise_and_622);  bitwise_or_87 = bitwise_and_622 = None
        select_713 = torch.ops.aten.select.int(eq, 1, 89)
        select_714 = torch.ops.aten.select.int(eq, 1, 91)
        bitwise_and_623 = torch.ops.aten.bitwise_and.Tensor(select_713, select_714);  select_713 = select_714 = None
        select_715 = torch.ops.aten.select.int(eq, 1, 94)
        bitwise_and_624 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_623, select_715);  bitwise_and_623 = select_715 = None
        select_716 = torch.ops.aten.select.int(eq, 1, 95)
        bitwise_and_625 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_624, select_716);  bitwise_and_624 = select_716 = None
        select_717 = torch.ops.aten.select.int(eq, 1, 99)
        bitwise_and_626 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_625, select_717);  bitwise_and_625 = select_717 = None
        select_718 = torch.ops.aten.select.int(eq, 1, 101)
        bitwise_and_627 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_626, select_718);  bitwise_and_626 = select_718 = None
        select_719 = torch.ops.aten.select.int(eq, 1, 104)
        bitwise_and_628 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_627, select_719);  bitwise_and_627 = select_719 = None
        select_720 = torch.ops.aten.select.int(eq, 1, 105)
        bitwise_and_629 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_628, select_720);  bitwise_and_628 = select_720 = None
        bitwise_or_89 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_88, bitwise_and_629);  bitwise_or_88 = bitwise_and_629 = None
        select_721 = torch.ops.aten.select.int(eq, 1, 90)
        select_722 = torch.ops.aten.select.int(eq, 1, 92)
        bitwise_and_630 = torch.ops.aten.bitwise_and.Tensor(select_721, select_722);  select_721 = select_722 = None
        select_723 = torch.ops.aten.select.int(eq, 1, 95)
        bitwise_and_631 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_630, select_723);  bitwise_and_630 = select_723 = None
        select_724 = torch.ops.aten.select.int(eq, 1, 96)
        bitwise_and_632 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_631, select_724);  bitwise_and_631 = select_724 = None
        select_725 = torch.ops.aten.select.int(eq, 1, 100)
        bitwise_and_633 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_632, select_725);  bitwise_and_632 = select_725 = None
        select_726 = torch.ops.aten.select.int(eq, 1, 102)
        bitwise_and_634 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_633, select_726);  bitwise_and_633 = select_726 = None
        select_727 = torch.ops.aten.select.int(eq, 1, 105)
        bitwise_and_635 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_634, select_727);  bitwise_and_634 = select_727 = None
        select_728 = torch.ops.aten.select.int(eq, 1, 106)
        bitwise_and_636 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_635, select_728);  bitwise_and_635 = select_728 = None
        bitwise_or_90 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_89, bitwise_and_636);  bitwise_or_89 = bitwise_and_636 = None
        select_729 = torch.ops.aten.select.int(eq, 1, 91)
        select_730 = torch.ops.aten.select.int(eq, 1, 93)
        bitwise_and_637 = torch.ops.aten.bitwise_and.Tensor(select_729, select_730);  select_729 = select_730 = None
        select_731 = torch.ops.aten.select.int(eq, 1, 96)
        bitwise_and_638 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_637, select_731);  bitwise_and_637 = select_731 = None
        select_732 = torch.ops.aten.select.int(eq, 1, 97)
        bitwise_and_639 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_638, select_732);  bitwise_and_638 = select_732 = None
        select_733 = torch.ops.aten.select.int(eq, 1, 101)
        bitwise_and_640 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_639, select_733);  bitwise_and_639 = select_733 = None
        select_734 = torch.ops.aten.select.int(eq, 1, 103)
        bitwise_and_641 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_640, select_734);  bitwise_and_640 = select_734 = None
        select_735 = torch.ops.aten.select.int(eq, 1, 106)
        bitwise_and_642 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_641, select_735);  bitwise_and_641 = select_735 = None
        select_736 = torch.ops.aten.select.int(eq, 1, 107)
        bitwise_and_643 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_642, select_736);  bitwise_and_642 = select_736 = None
        bitwise_or_91 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_90, bitwise_and_643);  bitwise_or_90 = bitwise_and_643 = None
        select_737 = torch.ops.aten.select.int(eq, 1, 92)
        select_738 = torch.ops.aten.select.int(eq, 1, 94)
        bitwise_and_644 = torch.ops.aten.bitwise_and.Tensor(select_737, select_738);  select_737 = select_738 = None
        select_739 = torch.ops.aten.select.int(eq, 1, 97)
        bitwise_and_645 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_644, select_739);  bitwise_and_644 = select_739 = None
        select_740 = torch.ops.aten.select.int(eq, 1, 98)
        bitwise_and_646 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_645, select_740);  bitwise_and_645 = select_740 = None
        select_741 = torch.ops.aten.select.int(eq, 1, 102)
        bitwise_and_647 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_646, select_741);  bitwise_and_646 = select_741 = None
        select_742 = torch.ops.aten.select.int(eq, 1, 104)
        bitwise_and_648 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_647, select_742);  bitwise_and_647 = select_742 = None
        select_743 = torch.ops.aten.select.int(eq, 1, 107)
        bitwise_and_649 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_648, select_743);  bitwise_and_648 = select_743 = None
        select_744 = torch.ops.aten.select.int(eq, 1, 108)
        bitwise_and_650 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_649, select_744);  bitwise_and_649 = select_744 = None
        bitwise_or_92 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_91, bitwise_and_650);  bitwise_or_91 = bitwise_and_650 = None
        select_745 = torch.ops.aten.select.int(eq, 1, 93)
        select_746 = torch.ops.aten.select.int(eq, 1, 95)
        bitwise_and_651 = torch.ops.aten.bitwise_and.Tensor(select_745, select_746);  select_745 = select_746 = None
        select_747 = torch.ops.aten.select.int(eq, 1, 98)
        bitwise_and_652 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_651, select_747);  bitwise_and_651 = select_747 = None
        select_748 = torch.ops.aten.select.int(eq, 1, 99)
        bitwise_and_653 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_652, select_748);  bitwise_and_652 = select_748 = None
        select_749 = torch.ops.aten.select.int(eq, 1, 103)
        bitwise_and_654 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_653, select_749);  bitwise_and_653 = select_749 = None
        select_750 = torch.ops.aten.select.int(eq, 1, 105)
        bitwise_and_655 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_654, select_750);  bitwise_and_654 = select_750 = None
        select_751 = torch.ops.aten.select.int(eq, 1, 108)
        bitwise_and_656 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_655, select_751);  bitwise_and_655 = select_751 = None
        select_752 = torch.ops.aten.select.int(eq, 1, 109)
        bitwise_and_657 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_656, select_752);  bitwise_and_656 = select_752 = None
        bitwise_or_93 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_92, bitwise_and_657);  bitwise_or_92 = bitwise_and_657 = None
        select_753 = torch.ops.aten.select.int(eq, 1, 94)
        select_754 = torch.ops.aten.select.int(eq, 1, 96)
        bitwise_and_658 = torch.ops.aten.bitwise_and.Tensor(select_753, select_754);  select_753 = select_754 = None
        select_755 = torch.ops.aten.select.int(eq, 1, 99)
        bitwise_and_659 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_658, select_755);  bitwise_and_658 = select_755 = None
        select_756 = torch.ops.aten.select.int(eq, 1, 100)
        bitwise_and_660 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_659, select_756);  bitwise_and_659 = select_756 = None
        select_757 = torch.ops.aten.select.int(eq, 1, 104)
        bitwise_and_661 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_660, select_757);  bitwise_and_660 = select_757 = None
        select_758 = torch.ops.aten.select.int(eq, 1, 106)
        bitwise_and_662 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_661, select_758);  bitwise_and_661 = select_758 = None
        select_759 = torch.ops.aten.select.int(eq, 1, 109)
        bitwise_and_663 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_662, select_759);  bitwise_and_662 = select_759 = None
        select_760 = torch.ops.aten.select.int(eq, 1, 110)
        bitwise_and_664 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_663, select_760);  bitwise_and_663 = select_760 = None
        bitwise_or_94 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_93, bitwise_and_664);  bitwise_or_93 = bitwise_and_664 = None
        select_761 = torch.ops.aten.select.int(eq, 1, 95)
        select_762 = torch.ops.aten.select.int(eq, 1, 97)
        bitwise_and_665 = torch.ops.aten.bitwise_and.Tensor(select_761, select_762);  select_761 = select_762 = None
        select_763 = torch.ops.aten.select.int(eq, 1, 100)
        bitwise_and_666 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_665, select_763);  bitwise_and_665 = select_763 = None
        select_764 = torch.ops.aten.select.int(eq, 1, 101)
        bitwise_and_667 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_666, select_764);  bitwise_and_666 = select_764 = None
        select_765 = torch.ops.aten.select.int(eq, 1, 105)
        bitwise_and_668 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_667, select_765);  bitwise_and_667 = select_765 = None
        select_766 = torch.ops.aten.select.int(eq, 1, 107)
        bitwise_and_669 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_668, select_766);  bitwise_and_668 = select_766 = None
        select_767 = torch.ops.aten.select.int(eq, 1, 110)
        bitwise_and_670 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_669, select_767);  bitwise_and_669 = select_767 = None
        select_768 = torch.ops.aten.select.int(eq, 1, 111)
        bitwise_and_671 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_670, select_768);  bitwise_and_670 = select_768 = None
        bitwise_or_95 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_94, bitwise_and_671);  bitwise_or_94 = bitwise_and_671 = None
        select_769 = torch.ops.aten.select.int(eq, 1, 96)
        select_770 = torch.ops.aten.select.int(eq, 1, 98)
        bitwise_and_672 = torch.ops.aten.bitwise_and.Tensor(select_769, select_770);  select_769 = select_770 = None
        select_771 = torch.ops.aten.select.int(eq, 1, 101)
        bitwise_and_673 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_672, select_771);  bitwise_and_672 = select_771 = None
        select_772 = torch.ops.aten.select.int(eq, 1, 102)
        bitwise_and_674 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_673, select_772);  bitwise_and_673 = select_772 = None
        select_773 = torch.ops.aten.select.int(eq, 1, 106)
        bitwise_and_675 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_674, select_773);  bitwise_and_674 = select_773 = None
        select_774 = torch.ops.aten.select.int(eq, 1, 108)
        bitwise_and_676 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_675, select_774);  bitwise_and_675 = select_774 = None
        select_775 = torch.ops.aten.select.int(eq, 1, 111)
        bitwise_and_677 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_676, select_775);  bitwise_and_676 = select_775 = None
        select_776 = torch.ops.aten.select.int(eq, 1, 112)
        bitwise_and_678 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_677, select_776);  bitwise_and_677 = select_776 = None
        bitwise_or_96 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_95, bitwise_and_678);  bitwise_or_95 = bitwise_and_678 = None
        select_777 = torch.ops.aten.select.int(eq, 1, 97)
        select_778 = torch.ops.aten.select.int(eq, 1, 99)
        bitwise_and_679 = torch.ops.aten.bitwise_and.Tensor(select_777, select_778);  select_777 = select_778 = None
        select_779 = torch.ops.aten.select.int(eq, 1, 102)
        bitwise_and_680 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_679, select_779);  bitwise_and_679 = select_779 = None
        select_780 = torch.ops.aten.select.int(eq, 1, 103)
        bitwise_and_681 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_680, select_780);  bitwise_and_680 = select_780 = None
        select_781 = torch.ops.aten.select.int(eq, 1, 107)
        bitwise_and_682 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_681, select_781);  bitwise_and_681 = select_781 = None
        select_782 = torch.ops.aten.select.int(eq, 1, 109)
        bitwise_and_683 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_682, select_782);  bitwise_and_682 = select_782 = None
        select_783 = torch.ops.aten.select.int(eq, 1, 112)
        bitwise_and_684 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_683, select_783);  bitwise_and_683 = select_783 = None
        select_784 = torch.ops.aten.select.int(eq, 1, 113)
        bitwise_and_685 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_684, select_784);  bitwise_and_684 = select_784 = None
        bitwise_or_97 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_96, bitwise_and_685);  bitwise_or_96 = bitwise_and_685 = None
        select_785 = torch.ops.aten.select.int(eq, 1, 98)
        select_786 = torch.ops.aten.select.int(eq, 1, 100)
        bitwise_and_686 = torch.ops.aten.bitwise_and.Tensor(select_785, select_786);  select_785 = select_786 = None
        select_787 = torch.ops.aten.select.int(eq, 1, 103)
        bitwise_and_687 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_686, select_787);  bitwise_and_686 = select_787 = None
        select_788 = torch.ops.aten.select.int(eq, 1, 104)
        bitwise_and_688 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_687, select_788);  bitwise_and_687 = select_788 = None
        select_789 = torch.ops.aten.select.int(eq, 1, 108)
        bitwise_and_689 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_688, select_789);  bitwise_and_688 = select_789 = None
        select_790 = torch.ops.aten.select.int(eq, 1, 110)
        bitwise_and_690 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_689, select_790);  bitwise_and_689 = select_790 = None
        select_791 = torch.ops.aten.select.int(eq, 1, 113)
        bitwise_and_691 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_690, select_791);  bitwise_and_690 = select_791 = None
        select_792 = torch.ops.aten.select.int(eq, 1, 114)
        bitwise_and_692 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_691, select_792);  bitwise_and_691 = select_792 = None
        bitwise_or_98 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_97, bitwise_and_692);  bitwise_or_97 = bitwise_and_692 = None
        select_793 = torch.ops.aten.select.int(eq, 1, 99)
        select_794 = torch.ops.aten.select.int(eq, 1, 101)
        bitwise_and_693 = torch.ops.aten.bitwise_and.Tensor(select_793, select_794);  select_793 = select_794 = None
        select_795 = torch.ops.aten.select.int(eq, 1, 104)
        bitwise_and_694 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_693, select_795);  bitwise_and_693 = select_795 = None
        select_796 = torch.ops.aten.select.int(eq, 1, 105)
        bitwise_and_695 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_694, select_796);  bitwise_and_694 = select_796 = None
        select_797 = torch.ops.aten.select.int(eq, 1, 109)
        bitwise_and_696 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_695, select_797);  bitwise_and_695 = select_797 = None
        select_798 = torch.ops.aten.select.int(eq, 1, 111)
        bitwise_and_697 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_696, select_798);  bitwise_and_696 = select_798 = None
        select_799 = torch.ops.aten.select.int(eq, 1, 114)
        bitwise_and_698 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_697, select_799);  bitwise_and_697 = select_799 = None
        select_800 = torch.ops.aten.select.int(eq, 1, 115);  eq = None
        bitwise_and_699 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_698, select_800);  bitwise_and_698 = select_800 = None
        bitwise_or_99 = torch.ops.aten.bitwise_or.Tensor(bitwise_or_98, bitwise_and_699);  bitwise_or_98 = bitwise_and_699 = None
        bitwise_and_700 = torch.ops.aten.bitwise_and.Tensor(bitwise_or_99, ge);  bitwise_or_99 = ge = None
        full_default_1 = torch.ops.aten.full.default([1, 2], -100.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        _tensor_constant0 = self._tensor_constant0;  _tensor_constant0 = None
        full_default_2 = torch.ops.aten.full.default([], 100.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        select_801 = torch.ops.aten.select.int(full_default_1, 1, 1)
        copy = torch.ops.aten.copy.default(select_801, full_default_2);  select_801 = full_default_2 = None
        select_scatter = torch.ops.aten.select_scatter.default(full_default_1, copy, 1, 1);  full_default_1 = copy = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(bitwise_and_700, -1);  bitwise_and_700 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(unsqueeze_2, torch.float32);  unsqueeze_2 = None
        mul_87 = torch.ops.aten.mul.Tensor(convert_element_type_1, select_scatter);  select_scatter = None
        sub_38 = torch.ops.aten.sub.Tensor(1, convert_element_type_1);  convert_element_type_1 = None
        mul_88 = torch.ops.aten.mul.Tensor(sub_38, addmm_73);  sub_38 = addmm_73 = None
        add_100 = torch.ops.aten.add.Tensor(mul_87, mul_88);  mul_87 = mul_88 = None
        return (add_100,)
        
def load_args(reader):
    buf0 = reader.storage(None, 4096, dtype_hint=torch.int64)
    reader.tensor(buf0, (1, 512), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 4096, dtype_hint=torch.int64)
    reader.tensor(buf1, (1, 512), dtype=torch.int64, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 4096, dtype_hint=torch.int64)
    reader.tensor(buf2, (1, 512), dtype=torch.int64, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 93763584)
    reader.tensor(buf3, (30522, 768), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 4096, dtype_hint=torch.int64)
    reader.tensor(buf4, (1, 512), dtype=torch.int64, is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 6144)
    reader.tensor(buf5, (2, 768), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 1572864)
    reader.tensor(buf6, (512, 768), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 3072)
    reader.tensor(buf7, (768,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 3072)
    reader.tensor(buf8, (768,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 2359296)
    reader.tensor(buf9, (768, 768), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 3072)
    reader.tensor(buf10, (768,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 2359296)
    reader.tensor(buf11, (768, 768), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 3072)
    reader.tensor(buf12, (768,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 2359296)
    reader.tensor(buf13, (768, 768), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 3072)
    reader.tensor(buf14, (768,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 2359296)
    reader.tensor(buf15, (768, 768), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 3072)
    reader.tensor(buf16, (768,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 3072)
    reader.tensor(buf17, (768,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 3072)
    reader.tensor(buf18, (768,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 9437184)
    reader.tensor(buf19, (3072, 768), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 12288)
    reader.tensor(buf20, (3072,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 9437184)
    reader.tensor(buf21, (768, 3072), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 3072)
    reader.tensor(buf22, (768,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 3072)
    reader.tensor(buf23, (768,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 3072)
    reader.tensor(buf24, (768,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 2359296)
    reader.tensor(buf25, (768, 768), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 3072)
    reader.tensor(buf26, (768,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 2359296)
    reader.tensor(buf27, (768, 768), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 3072)
    reader.tensor(buf28, (768,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 2359296)
    reader.tensor(buf29, (768, 768), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 3072)
    reader.tensor(buf30, (768,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 2359296)
    reader.tensor(buf31, (768, 768), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 3072)
    reader.tensor(buf32, (768,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 3072)
    reader.tensor(buf33, (768,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 3072)
    reader.tensor(buf34, (768,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 9437184)
    reader.tensor(buf35, (3072, 768), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 12288)
    reader.tensor(buf36, (3072,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 9437184)
    reader.tensor(buf37, (768, 3072), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 3072)
    reader.tensor(buf38, (768,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 3072)
    reader.tensor(buf39, (768,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 3072)
    reader.tensor(buf40, (768,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 2359296)
    reader.tensor(buf41, (768, 768), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 3072)
    reader.tensor(buf42, (768,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 2359296)
    reader.tensor(buf43, (768, 768), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 3072)
    reader.tensor(buf44, (768,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 2359296)
    reader.tensor(buf45, (768, 768), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 3072)
    reader.tensor(buf46, (768,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 2359296)
    reader.tensor(buf47, (768, 768), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 3072)
    reader.tensor(buf48, (768,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 3072)
    reader.tensor(buf49, (768,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 3072)
    reader.tensor(buf50, (768,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 9437184)
    reader.tensor(buf51, (3072, 768), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 12288)
    reader.tensor(buf52, (3072,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 9437184)
    reader.tensor(buf53, (768, 3072), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 3072)
    reader.tensor(buf54, (768,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 3072)
    reader.tensor(buf55, (768,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 3072)
    reader.tensor(buf56, (768,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 2359296)
    reader.tensor(buf57, (768, 768), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 3072)
    reader.tensor(buf58, (768,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 2359296)
    reader.tensor(buf59, (768, 768), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 3072)
    reader.tensor(buf60, (768,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 2359296)
    reader.tensor(buf61, (768, 768), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 3072)
    reader.tensor(buf62, (768,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 2359296)
    reader.tensor(buf63, (768, 768), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 3072)
    reader.tensor(buf64, (768,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 3072)
    reader.tensor(buf65, (768,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 3072)
    reader.tensor(buf66, (768,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 9437184)
    reader.tensor(buf67, (3072, 768), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 12288)
    reader.tensor(buf68, (3072,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 9437184)
    reader.tensor(buf69, (768, 3072), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 3072)
    reader.tensor(buf70, (768,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 3072)
    reader.tensor(buf71, (768,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 3072)
    reader.tensor(buf72, (768,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 2359296)
    reader.tensor(buf73, (768, 768), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 3072)
    reader.tensor(buf74, (768,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 2359296)
    reader.tensor(buf75, (768, 768), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 3072)
    reader.tensor(buf76, (768,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 2359296)
    reader.tensor(buf77, (768, 768), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 3072)
    reader.tensor(buf78, (768,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 2359296)
    reader.tensor(buf79, (768, 768), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 3072)
    reader.tensor(buf80, (768,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 3072)
    reader.tensor(buf81, (768,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 3072)
    reader.tensor(buf82, (768,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 9437184)
    reader.tensor(buf83, (3072, 768), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 12288)
    reader.tensor(buf84, (3072,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 9437184)
    reader.tensor(buf85, (768, 3072), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 3072)
    reader.tensor(buf86, (768,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 3072)
    reader.tensor(buf87, (768,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 3072)
    reader.tensor(buf88, (768,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 2359296)
    reader.tensor(buf89, (768, 768), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 3072)
    reader.tensor(buf90, (768,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 2359296)
    reader.tensor(buf91, (768, 768), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 3072)
    reader.tensor(buf92, (768,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 2359296)
    reader.tensor(buf93, (768, 768), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 3072)
    reader.tensor(buf94, (768,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 2359296)
    reader.tensor(buf95, (768, 768), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 3072)
    reader.tensor(buf96, (768,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 3072)
    reader.tensor(buf97, (768,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 3072)
    reader.tensor(buf98, (768,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 9437184)
    reader.tensor(buf99, (3072, 768), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 12288)
    reader.tensor(buf100, (3072,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 9437184)
    reader.tensor(buf101, (768, 3072), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 3072)
    reader.tensor(buf102, (768,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 3072)
    reader.tensor(buf103, (768,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 3072)
    reader.tensor(buf104, (768,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 2359296)
    reader.tensor(buf105, (768, 768), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 3072)
    reader.tensor(buf106, (768,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 2359296)
    reader.tensor(buf107, (768, 768), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 3072)
    reader.tensor(buf108, (768,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 2359296)
    reader.tensor(buf109, (768, 768), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 3072)
    reader.tensor(buf110, (768,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 2359296)
    reader.tensor(buf111, (768, 768), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 3072)
    reader.tensor(buf112, (768,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 3072)
    reader.tensor(buf113, (768,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 3072)
    reader.tensor(buf114, (768,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 9437184)
    reader.tensor(buf115, (3072, 768), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 12288)
    reader.tensor(buf116, (3072,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 9437184)
    reader.tensor(buf117, (768, 3072), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 3072)
    reader.tensor(buf118, (768,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 3072)
    reader.tensor(buf119, (768,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 3072)
    reader.tensor(buf120, (768,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 2359296)
    reader.tensor(buf121, (768, 768), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 3072)
    reader.tensor(buf122, (768,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 2359296)
    reader.tensor(buf123, (768, 768), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 3072)
    reader.tensor(buf124, (768,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 2359296)
    reader.tensor(buf125, (768, 768), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 3072)
    reader.tensor(buf126, (768,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 2359296)
    reader.tensor(buf127, (768, 768), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 3072)
    reader.tensor(buf128, (768,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 3072)
    reader.tensor(buf129, (768,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 3072)
    reader.tensor(buf130, (768,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 9437184)
    reader.tensor(buf131, (3072, 768), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 12288)
    reader.tensor(buf132, (3072,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 9437184)
    reader.tensor(buf133, (768, 3072), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 3072)
    reader.tensor(buf134, (768,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 3072)
    reader.tensor(buf135, (768,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 3072)
    reader.tensor(buf136, (768,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 2359296)
    reader.tensor(buf137, (768, 768), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 3072)
    reader.tensor(buf138, (768,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 2359296)
    reader.tensor(buf139, (768, 768), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 3072)
    reader.tensor(buf140, (768,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 2359296)
    reader.tensor(buf141, (768, 768), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 3072)
    reader.tensor(buf142, (768,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 2359296)
    reader.tensor(buf143, (768, 768), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 3072)
    reader.tensor(buf144, (768,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 3072)
    reader.tensor(buf145, (768,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 3072)
    reader.tensor(buf146, (768,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 9437184)
    reader.tensor(buf147, (3072, 768), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 12288)
    reader.tensor(buf148, (3072,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 9437184)
    reader.tensor(buf149, (768, 3072), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 3072)
    reader.tensor(buf150, (768,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 3072)
    reader.tensor(buf151, (768,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 3072)
    reader.tensor(buf152, (768,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 2359296)
    reader.tensor(buf153, (768, 768), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 3072)
    reader.tensor(buf154, (768,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 2359296)
    reader.tensor(buf155, (768, 768), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 3072)
    reader.tensor(buf156, (768,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 2359296)
    reader.tensor(buf157, (768, 768), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 3072)
    reader.tensor(buf158, (768,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 2359296)
    reader.tensor(buf159, (768, 768), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 3072)
    reader.tensor(buf160, (768,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 3072)
    reader.tensor(buf161, (768,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 3072)
    reader.tensor(buf162, (768,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 9437184)
    reader.tensor(buf163, (3072, 768), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 12288)
    reader.tensor(buf164, (3072,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 9437184)
    reader.tensor(buf165, (768, 3072), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 3072)
    reader.tensor(buf166, (768,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 3072)
    reader.tensor(buf167, (768,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 3072)
    reader.tensor(buf168, (768,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 2359296)
    reader.tensor(buf169, (768, 768), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 3072)
    reader.tensor(buf170, (768,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 2359296)
    reader.tensor(buf171, (768, 768), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 3072)
    reader.tensor(buf172, (768,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 2359296)
    reader.tensor(buf173, (768, 768), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 3072)
    reader.tensor(buf174, (768,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 2359296)
    reader.tensor(buf175, (768, 768), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 3072)
    reader.tensor(buf176, (768,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 3072)
    reader.tensor(buf177, (768,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 3072)
    reader.tensor(buf178, (768,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 9437184)
    reader.tensor(buf179, (3072, 768), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 12288)
    reader.tensor(buf180, (3072,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 9437184)
    reader.tensor(buf181, (768, 3072), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 3072)
    reader.tensor(buf182, (768,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 3072)
    reader.tensor(buf183, (768,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 3072)
    reader.tensor(buf184, (768,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 2359296)
    reader.tensor(buf185, (768, 768), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 3072)
    reader.tensor(buf186, (768,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 2359296)
    reader.tensor(buf187, (768, 768), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 3072)
    reader.tensor(buf188, (768,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 2359296)
    reader.tensor(buf189, (768, 768), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 3072)
    reader.tensor(buf190, (768,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 2359296)
    reader.tensor(buf191, (768, 768), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 3072)
    reader.tensor(buf192, (768,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 3072)
    reader.tensor(buf193, (768,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 3072)
    reader.tensor(buf194, (768,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 9437184)
    reader.tensor(buf195, (3072, 768), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 12288)
    reader.tensor(buf196, (3072,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 9437184)
    reader.tensor(buf197, (768, 3072), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 3072)
    reader.tensor(buf198, (768,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 3072)
    reader.tensor(buf199, (768,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 3072)
    reader.tensor(buf200, (768,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 2359296)
    reader.tensor(buf201, (768, 768), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 3072)
    reader.tensor(buf202, (768,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 6144)
    reader.tensor(buf203, (2, 768), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 8)
    reader.tensor(buf204, (2,), is_leaf=True)  # arg204_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)