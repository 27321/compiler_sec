# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


cpp_fused_add_embedding_native_layer_norm_slice_0 = async_compile.cpp_pybinding(['const int64_t*', 'const float*', 'const int64_t*', 'const float*', 'const int64_t*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const int64_t* in_ptr0,
                       const float* in_ptr1,
                       const int64_t* in_ptr2,
                       const float* in_ptr3,
                       const int64_t* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(32)
    {
        int tid = omp_get_thread_num();
        {
            std::unique_ptr<float []> buf_local_buffer_data_0 = std::make_unique<float []>(768L);
            float* local_buffer_data_0 = buf_local_buffer_data_0.get();
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(32L); x0+=static_cast<int64_t>(1L))
            {
                {
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    static WelfordHelper<float, 4096> scalar_welford_helper0(static_cast<int64_t>(768L));
                    static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(48L));
                    static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                    for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(16L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                            {
                                auto tmp0 = in_ptr0[static_cast<int64_t>(x0)];
                                auto tmp10 = in_ptr2[static_cast<int64_t>(x0)];
                                auto tmp21 = in_ptr4[static_cast<int64_t>(x0)];
                                auto tmp1 = 30522L;
                                auto tmp2 = c10::convert<int64_t>(tmp1);
                                auto tmp3 = int64_t(tmp0 + tmp2);
                                auto tmp4 = tmp0 < 0;
                                auto tmp5 = tmp4 ? tmp3 : tmp0;
                                auto tmp6 = tmp5;
                                auto tmp7 = c10::convert<int64_t>(tmp6);
                                TORCH_CHECK((0 <= tmp7) & (tmp7 < 30522L), "index out of bounds: 0 <= tmp7 < 30522L");
                                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*tmp5), static_cast<int64_t>(16));
                                auto tmp11 = 2L;
                                auto tmp12 = c10::convert<int64_t>(tmp11);
                                auto tmp13 = int64_t(tmp10 + tmp12);
                                auto tmp14 = tmp10 < 0;
                                auto tmp15 = tmp14 ? tmp13 : tmp10;
                                auto tmp16 = tmp15;
                                auto tmp17 = c10::convert<int64_t>(tmp16);
                                TORCH_CHECK((0 <= tmp17) & (tmp17 < 2L), "index out of bounds: 0 <= tmp17 < 2L");
                                auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1 + 768L*tmp15), static_cast<int64_t>(16));
                                auto tmp20 = tmp9 + tmp19;
                                auto tmp22 = 512L;
                                auto tmp23 = c10::convert<int64_t>(tmp22);
                                auto tmp24 = int64_t(tmp21 + tmp23);
                                auto tmp25 = tmp21 < 0;
                                auto tmp26 = tmp25 ? tmp24 : tmp21;
                                auto tmp27 = tmp26;
                                auto tmp28 = c10::convert<int64_t>(tmp27);
                                TORCH_CHECK((0 <= tmp28) & (tmp28 < 512L), "index out of bounds: 0 <= tmp28 < 512L");
                                auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x1 + 768L*tmp26), static_cast<int64_t>(16));
                                auto tmp31 = tmp20 + tmp30;
                                tmp31.store(local_buffer_data_0 + static_cast<int64_t>(x1));
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp31, &welford_helper0);
                            }
                        }
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, &scalar_welford_helper0);
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                    masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(16L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(local_buffer_data_0 + static_cast<int64_t>(x1), static_cast<int64_t>(16));
                            auto tmp1 = out_ptr0[static_cast<int64_t>(x0)];
                            auto tmp4 = out_ptr1[static_cast<int64_t>(x0)];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(x1), static_cast<int64_t>(16));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<int64_t>(x1), static_cast<int64_t>(16));
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp5 = static_cast<float>(768.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(1e-12);
                            auto tmp8 = float(tmp6 + tmp7);
                            auto tmp9 = 1 / std::sqrt(tmp8);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp3 * tmp10;
                            auto tmp13 = tmp11 * tmp12;
                            auto tmp15 = tmp13 + tmp14;
                            tmp15.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_1 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(32)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(32L); x0+=static_cast<int64_t>(1L))
            {
                {
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    static WelfordHelper<float, 4096> scalar_welford_helper0(static_cast<int64_t>(768L));
                    static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(48L));
                    static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                    for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(16L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(16));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(16));
                                auto tmp2 = tmp0 + tmp1;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                            }
                        }
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, &scalar_welford_helper0);
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                    masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(16L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(16));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(16));
                            auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                            auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(16));
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(16));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(768.0);
                            auto tmp8 = tmp6 / tmp7;
                            auto tmp9 = static_cast<float>(1e-12);
                            auto tmp10 = float(tmp8 + tmp9);
                            auto tmp11 = 1 / std::sqrt(tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp15 = tmp13 * tmp14;
                            auto tmp17 = tmp15 + tmp16;
                            tmp17.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_2 = async_compile.cpp_pybinding(['float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(32)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(98304L); x0+=static_cast<int64_t>(16L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(98304L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.7071067811865476);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp0 * tmp5;
                        auto tmp7 = tmp6.erf();
                        auto tmp8 = static_cast<float>(1.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 + tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        tmp11.store(in_out_ptr0 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_tanh_3 = async_compile.cpp_pybinding(['float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(768L); x0+=static_cast<int64_t>(16L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(768L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                    auto tmp1 = tmp0.tanh();
                    tmp1.store(in_out_ptr0 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 32), (32, 1))
        assert_size_stride(arg1_1, (1, 512), (512, 1))
        assert_size_stride(arg2_1, (30522, 768), (768, 1))
        assert_size_stride(arg3_1, (1, 512), (512, 1))
        assert_size_stride(arg4_1, (2, 768), (768, 1))
        assert_size_stride(arg5_1, (512, 768), (768, 1))
        assert_size_stride(arg6_1, (768, ), (1, ))
        assert_size_stride(arg7_1, (768, ), (1, ))
        assert_size_stride(arg8_1, (768, 768), (768, 1))
        assert_size_stride(arg9_1, (768, ), (1, ))
        assert_size_stride(arg10_1, (768, 768), (768, 1))
        assert_size_stride(arg11_1, (768, ), (1, ))
        assert_size_stride(arg12_1, (768, 768), (768, 1))
        assert_size_stride(arg13_1, (768, ), (1, ))
        assert_size_stride(arg14_1, (768, 768), (768, 1))
        assert_size_stride(arg15_1, (768, ), (1, ))
        assert_size_stride(arg16_1, (768, ), (1, ))
        assert_size_stride(arg17_1, (768, ), (1, ))
        assert_size_stride(arg18_1, (3072, 768), (768, 1))
        assert_size_stride(arg19_1, (3072, ), (1, ))
        assert_size_stride(arg20_1, (768, 3072), (3072, 1))
        assert_size_stride(arg21_1, (768, ), (1, ))
        assert_size_stride(arg22_1, (768, ), (1, ))
        assert_size_stride(arg23_1, (768, ), (1, ))
        assert_size_stride(arg24_1, (768, 768), (768, 1))
        assert_size_stride(arg25_1, (768, ), (1, ))
        assert_size_stride(arg26_1, (768, 768), (768, 1))
        assert_size_stride(arg27_1, (768, ), (1, ))
        assert_size_stride(arg28_1, (768, 768), (768, 1))
        assert_size_stride(arg29_1, (768, ), (1, ))
        assert_size_stride(arg30_1, (768, 768), (768, 1))
        assert_size_stride(arg31_1, (768, ), (1, ))
        assert_size_stride(arg32_1, (768, ), (1, ))
        assert_size_stride(arg33_1, (768, ), (1, ))
        assert_size_stride(arg34_1, (3072, 768), (768, 1))
        assert_size_stride(arg35_1, (3072, ), (1, ))
        assert_size_stride(arg36_1, (768, 3072), (3072, 1))
        assert_size_stride(arg37_1, (768, ), (1, ))
        assert_size_stride(arg38_1, (768, ), (1, ))
        assert_size_stride(arg39_1, (768, ), (1, ))
        assert_size_stride(arg40_1, (768, 768), (768, 1))
        assert_size_stride(arg41_1, (768, ), (1, ))
        assert_size_stride(arg42_1, (768, 768), (768, 1))
        assert_size_stride(arg43_1, (768, ), (1, ))
        assert_size_stride(arg44_1, (768, 768), (768, 1))
        assert_size_stride(arg45_1, (768, ), (1, ))
        assert_size_stride(arg46_1, (768, 768), (768, 1))
        assert_size_stride(arg47_1, (768, ), (1, ))
        assert_size_stride(arg48_1, (768, ), (1, ))
        assert_size_stride(arg49_1, (768, ), (1, ))
        assert_size_stride(arg50_1, (3072, 768), (768, 1))
        assert_size_stride(arg51_1, (3072, ), (1, ))
        assert_size_stride(arg52_1, (768, 3072), (3072, 1))
        assert_size_stride(arg53_1, (768, ), (1, ))
        assert_size_stride(arg54_1, (768, ), (1, ))
        assert_size_stride(arg55_1, (768, ), (1, ))
        assert_size_stride(arg56_1, (768, 768), (768, 1))
        assert_size_stride(arg57_1, (768, ), (1, ))
        assert_size_stride(arg58_1, (768, 768), (768, 1))
        assert_size_stride(arg59_1, (768, ), (1, ))
        assert_size_stride(arg60_1, (768, 768), (768, 1))
        assert_size_stride(arg61_1, (768, ), (1, ))
        assert_size_stride(arg62_1, (768, 768), (768, 1))
        assert_size_stride(arg63_1, (768, ), (1, ))
        assert_size_stride(arg64_1, (768, ), (1, ))
        assert_size_stride(arg65_1, (768, ), (1, ))
        assert_size_stride(arg66_1, (3072, 768), (768, 1))
        assert_size_stride(arg67_1, (3072, ), (1, ))
        assert_size_stride(arg68_1, (768, 3072), (3072, 1))
        assert_size_stride(arg69_1, (768, ), (1, ))
        assert_size_stride(arg70_1, (768, ), (1, ))
        assert_size_stride(arg71_1, (768, ), (1, ))
        assert_size_stride(arg72_1, (768, 768), (768, 1))
        assert_size_stride(arg73_1, (768, ), (1, ))
        assert_size_stride(arg74_1, (768, 768), (768, 1))
        assert_size_stride(arg75_1, (768, ), (1, ))
        assert_size_stride(arg76_1, (768, 768), (768, 1))
        assert_size_stride(arg77_1, (768, ), (1, ))
        assert_size_stride(arg78_1, (768, 768), (768, 1))
        assert_size_stride(arg79_1, (768, ), (1, ))
        assert_size_stride(arg80_1, (768, ), (1, ))
        assert_size_stride(arg81_1, (768, ), (1, ))
        assert_size_stride(arg82_1, (3072, 768), (768, 1))
        assert_size_stride(arg83_1, (3072, ), (1, ))
        assert_size_stride(arg84_1, (768, 3072), (3072, 1))
        assert_size_stride(arg85_1, (768, ), (1, ))
        assert_size_stride(arg86_1, (768, ), (1, ))
        assert_size_stride(arg87_1, (768, ), (1, ))
        assert_size_stride(arg88_1, (768, 768), (768, 1))
        assert_size_stride(arg89_1, (768, ), (1, ))
        assert_size_stride(arg90_1, (768, 768), (768, 1))
        assert_size_stride(arg91_1, (768, ), (1, ))
        assert_size_stride(arg92_1, (768, 768), (768, 1))
        assert_size_stride(arg93_1, (768, ), (1, ))
        assert_size_stride(arg94_1, (768, 768), (768, 1))
        assert_size_stride(arg95_1, (768, ), (1, ))
        assert_size_stride(arg96_1, (768, ), (1, ))
        assert_size_stride(arg97_1, (768, ), (1, ))
        assert_size_stride(arg98_1, (3072, 768), (768, 1))
        assert_size_stride(arg99_1, (3072, ), (1, ))
        assert_size_stride(arg100_1, (768, 3072), (3072, 1))
        assert_size_stride(arg101_1, (768, ), (1, ))
        assert_size_stride(arg102_1, (768, ), (1, ))
        assert_size_stride(arg103_1, (768, ), (1, ))
        assert_size_stride(arg104_1, (768, 768), (768, 1))
        assert_size_stride(arg105_1, (768, ), (1, ))
        assert_size_stride(arg106_1, (768, 768), (768, 1))
        assert_size_stride(arg107_1, (768, ), (1, ))
        assert_size_stride(arg108_1, (768, 768), (768, 1))
        assert_size_stride(arg109_1, (768, ), (1, ))
        assert_size_stride(arg110_1, (768, 768), (768, 1))
        assert_size_stride(arg111_1, (768, ), (1, ))
        assert_size_stride(arg112_1, (768, ), (1, ))
        assert_size_stride(arg113_1, (768, ), (1, ))
        assert_size_stride(arg114_1, (3072, 768), (768, 1))
        assert_size_stride(arg115_1, (3072, ), (1, ))
        assert_size_stride(arg116_1, (768, 3072), (3072, 1))
        assert_size_stride(arg117_1, (768, ), (1, ))
        assert_size_stride(arg118_1, (768, ), (1, ))
        assert_size_stride(arg119_1, (768, ), (1, ))
        assert_size_stride(arg120_1, (768, 768), (768, 1))
        assert_size_stride(arg121_1, (768, ), (1, ))
        assert_size_stride(arg122_1, (768, 768), (768, 1))
        assert_size_stride(arg123_1, (768, ), (1, ))
        assert_size_stride(arg124_1, (768, 768), (768, 1))
        assert_size_stride(arg125_1, (768, ), (1, ))
        assert_size_stride(arg126_1, (768, 768), (768, 1))
        assert_size_stride(arg127_1, (768, ), (1, ))
        assert_size_stride(arg128_1, (768, ), (1, ))
        assert_size_stride(arg129_1, (768, ), (1, ))
        assert_size_stride(arg130_1, (3072, 768), (768, 1))
        assert_size_stride(arg131_1, (3072, ), (1, ))
        assert_size_stride(arg132_1, (768, 3072), (3072, 1))
        assert_size_stride(arg133_1, (768, ), (1, ))
        assert_size_stride(arg134_1, (768, ), (1, ))
        assert_size_stride(arg135_1, (768, ), (1, ))
        assert_size_stride(arg136_1, (768, 768), (768, 1))
        assert_size_stride(arg137_1, (768, ), (1, ))
        assert_size_stride(arg138_1, (768, 768), (768, 1))
        assert_size_stride(arg139_1, (768, ), (1, ))
        assert_size_stride(arg140_1, (768, 768), (768, 1))
        assert_size_stride(arg141_1, (768, ), (1, ))
        assert_size_stride(arg142_1, (768, 768), (768, 1))
        assert_size_stride(arg143_1, (768, ), (1, ))
        assert_size_stride(arg144_1, (768, ), (1, ))
        assert_size_stride(arg145_1, (768, ), (1, ))
        assert_size_stride(arg146_1, (3072, 768), (768, 1))
        assert_size_stride(arg147_1, (3072, ), (1, ))
        assert_size_stride(arg148_1, (768, 3072), (3072, 1))
        assert_size_stride(arg149_1, (768, ), (1, ))
        assert_size_stride(arg150_1, (768, ), (1, ))
        assert_size_stride(arg151_1, (768, ), (1, ))
        assert_size_stride(arg152_1, (768, 768), (768, 1))
        assert_size_stride(arg153_1, (768, ), (1, ))
        assert_size_stride(arg154_1, (768, 768), (768, 1))
        assert_size_stride(arg155_1, (768, ), (1, ))
        assert_size_stride(arg156_1, (768, 768), (768, 1))
        assert_size_stride(arg157_1, (768, ), (1, ))
        assert_size_stride(arg158_1, (768, 768), (768, 1))
        assert_size_stride(arg159_1, (768, ), (1, ))
        assert_size_stride(arg160_1, (768, ), (1, ))
        assert_size_stride(arg161_1, (768, ), (1, ))
        assert_size_stride(arg162_1, (3072, 768), (768, 1))
        assert_size_stride(arg163_1, (3072, ), (1, ))
        assert_size_stride(arg164_1, (768, 3072), (3072, 1))
        assert_size_stride(arg165_1, (768, ), (1, ))
        assert_size_stride(arg166_1, (768, ), (1, ))
        assert_size_stride(arg167_1, (768, ), (1, ))
        assert_size_stride(arg168_1, (768, 768), (768, 1))
        assert_size_stride(arg169_1, (768, ), (1, ))
        assert_size_stride(arg170_1, (768, 768), (768, 1))
        assert_size_stride(arg171_1, (768, ), (1, ))
        assert_size_stride(arg172_1, (768, 768), (768, 1))
        assert_size_stride(arg173_1, (768, ), (1, ))
        assert_size_stride(arg174_1, (768, 768), (768, 1))
        assert_size_stride(arg175_1, (768, ), (1, ))
        assert_size_stride(arg176_1, (768, ), (1, ))
        assert_size_stride(arg177_1, (768, ), (1, ))
        assert_size_stride(arg178_1, (3072, 768), (768, 1))
        assert_size_stride(arg179_1, (3072, ), (1, ))
        assert_size_stride(arg180_1, (768, 3072), (3072, 1))
        assert_size_stride(arg181_1, (768, ), (1, ))
        assert_size_stride(arg182_1, (768, ), (1, ))
        assert_size_stride(arg183_1, (768, ), (1, ))
        assert_size_stride(arg184_1, (768, 768), (768, 1))
        assert_size_stride(arg185_1, (768, ), (1, ))
        assert_size_stride(arg186_1, (768, 768), (768, 1))
        assert_size_stride(arg187_1, (768, ), (1, ))
        assert_size_stride(arg188_1, (768, 768), (768, 1))
        assert_size_stride(arg189_1, (768, ), (1, ))
        assert_size_stride(arg190_1, (768, 768), (768, 1))
        assert_size_stride(arg191_1, (768, ), (1, ))
        assert_size_stride(arg192_1, (768, ), (1, ))
        assert_size_stride(arg193_1, (768, ), (1, ))
        assert_size_stride(arg194_1, (3072, 768), (768, 1))
        assert_size_stride(arg195_1, (3072, ), (1, ))
        assert_size_stride(arg196_1, (768, 3072), (3072, 1))
        assert_size_stride(arg197_1, (768, ), (1, ))
        assert_size_stride(arg198_1, (768, ), (1, ))
        assert_size_stride(arg199_1, (768, ), (1, ))
        assert_size_stride(arg200_1, (768, 768), (768, 1))
        assert_size_stride(arg201_1, (768, ), (1, ))
        buf1 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf2 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf4 = empty_strided_cpu((1, 32, 768), (24576, 768, 1), torch.float32)
        cpp_fused_add_embedding_native_layer_norm_slice_0(arg0_1, arg2_1, arg1_1, arg4_1, arg3_1, arg5_1, arg6_1, arg7_1, buf1, buf2, buf4)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        buf5 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mixed_query_layer], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg9_1, reinterpret_tensor(buf4, (32, 768), (768, 1), 0), reinterpret_tensor(arg8_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf5)
        del arg8_1
        del arg9_1
        buf6 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg11_1, reinterpret_tensor(buf4, (32, 768), (768, 1), 0), reinterpret_tensor(arg10_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf6)
        del arg10_1
        del arg11_1
        buf7 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg13_1, reinterpret_tensor(buf4, (32, 768), (768, 1), 0), reinterpret_tensor(arg12_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf7)
        del arg12_1
        del arg13_1
        # Topologically Sorted Source Nodes: [mixed_query_layer, x_2, query_layer, linear_1, x, key_layer, linear_2, x_1, value_layer], Original ATen: [aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        buf8 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf5, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf6, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf7, (1, 12, 32, 64), (24576, 64, 768, 1), 0), scale=0.125)
        del buf5
        del buf6
        buf9 = buf8[0]
        assert_size_stride(buf9, (1, 12, 32, 64), (24576, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf9, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf8
        buf11 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [permute_3, context_layer_2, hidden_states], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg15_1, reinterpret_tensor(buf9, (32, 768), (768, 1), 0), reinterpret_tensor(arg14_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf11)
        del arg14_1
        del arg15_1
        del buf9
        buf12 = buf2; del buf2  # reuse
        buf13 = buf1; del buf1  # reuse
        buf15 = reinterpret_tensor(buf11, (1, 32, 768), (24576, 768, 1), 0); del buf11  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf15, buf4, arg16_1, arg17_1, buf12, buf13)
        del arg16_1
        del arg17_1
        del buf12
        del buf13
        del buf4
        buf16 = empty_strided_cpu((32, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_3], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg19_1, reinterpret_tensor(buf15, (32, 768), (768, 1), 0), reinterpret_tensor(arg18_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf16)
        del arg18_1
        del arg19_1
        buf17 = reinterpret_tensor(buf16, (1, 32, 3072), (98304, 3072, 1), 0); del buf16  # reuse
        cpp_fused_gelu_view_2(buf17)
        buf18 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4, hidden_states_5], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg21_1, reinterpret_tensor(buf17, (32, 3072), (3072, 1), 0), reinterpret_tensor(arg20_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf18)
        del arg20_1
        del arg21_1
        del buf17
        buf19 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf20 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf22 = reinterpret_tensor(buf18, (1, 32, 768), (24576, 768, 1), 0); del buf18  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf22, buf15, arg22_1, arg23_1, buf19, buf20)
        del arg22_1
        del arg23_1
        buf23 = reinterpret_tensor(buf15, (32, 768), (768, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_1], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg25_1, reinterpret_tensor(buf22, (32, 768), (768, 1), 0), reinterpret_tensor(arg24_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf23)
        del arg24_1
        del arg25_1
        buf24 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg27_1, reinterpret_tensor(buf22, (32, 768), (768, 1), 0), reinterpret_tensor(arg26_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf24)
        del arg26_1
        del arg27_1
        buf25 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg29_1, reinterpret_tensor(buf22, (32, 768), (768, 1), 0), reinterpret_tensor(arg28_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf25)
        del arg28_1
        del arg29_1
        # Topologically Sorted Source Nodes: [mixed_query_layer_1, x_5, query_layer_1, linear_7, x_3, key_layer_1, linear_8, x_4, value_layer_1], Original ATen: [aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        buf26 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf23, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf24, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf25, (1, 12, 32, 64), (24576, 64, 768, 1), 0), scale=0.125)
        del buf23
        del buf24
        buf27 = buf26[0]
        assert_size_stride(buf27, (1, 12, 32, 64), (24576, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf27, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf26
        buf29 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [permute_7, context_layer_5, hidden_states_8], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg31_1, reinterpret_tensor(buf27, (32, 768), (768, 1), 0), reinterpret_tensor(arg30_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf29)
        del arg30_1
        del arg31_1
        del buf27
        buf30 = buf20; del buf20  # reuse
        buf31 = buf19; del buf19  # reuse
        buf33 = reinterpret_tensor(buf29, (1, 32, 768), (24576, 768, 1), 0); del buf29  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf33, buf22, arg32_1, arg33_1, buf30, buf31)
        del arg32_1
        del arg33_1
        del buf22
        del buf30
        del buf31
        buf34 = empty_strided_cpu((32, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_11], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg35_1, reinterpret_tensor(buf33, (32, 768), (768, 1), 0), reinterpret_tensor(arg34_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf34)
        del arg34_1
        del arg35_1
        buf35 = reinterpret_tensor(buf34, (1, 32, 3072), (98304, 3072, 1), 0); del buf34  # reuse
        cpp_fused_gelu_view_2(buf35)
        buf36 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_11, hidden_states_12, hidden_states_13], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg37_1, reinterpret_tensor(buf35, (32, 3072), (3072, 1), 0), reinterpret_tensor(arg36_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf36)
        del arg36_1
        del arg37_1
        del buf35
        buf37 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf38 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf40 = reinterpret_tensor(buf36, (1, 32, 768), (24576, 768, 1), 0); del buf36  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf40, buf33, arg38_1, arg39_1, buf37, buf38)
        del arg38_1
        del arg39_1
        buf41 = reinterpret_tensor(buf33, (32, 768), (768, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_2], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg41_1, reinterpret_tensor(buf40, (32, 768), (768, 1), 0), reinterpret_tensor(arg40_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf41)
        del arg40_1
        del arg41_1
        buf42 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg43_1, reinterpret_tensor(buf40, (32, 768), (768, 1), 0), reinterpret_tensor(arg42_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf42)
        del arg42_1
        del arg43_1
        buf43 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg45_1, reinterpret_tensor(buf40, (32, 768), (768, 1), 0), reinterpret_tensor(arg44_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf43)
        del arg44_1
        del arg45_1
        # Topologically Sorted Source Nodes: [mixed_query_layer_2, x_8, query_layer_2, linear_13, x_6, key_layer_2, linear_14, x_7, value_layer_2], Original ATen: [aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        buf44 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf41, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf42, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf43, (1, 12, 32, 64), (24576, 64, 768, 1), 0), scale=0.125)
        del buf41
        del buf42
        buf45 = buf44[0]
        assert_size_stride(buf45, (1, 12, 32, 64), (24576, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf45, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf44
        buf47 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [permute_11, context_layer_8, hidden_states_16], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg47_1, reinterpret_tensor(buf45, (32, 768), (768, 1), 0), reinterpret_tensor(arg46_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf47)
        del arg46_1
        del arg47_1
        del buf45
        buf48 = buf38; del buf38  # reuse
        buf49 = buf37; del buf37  # reuse
        buf51 = reinterpret_tensor(buf47, (1, 32, 768), (24576, 768, 1), 0); del buf47  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf51, buf40, arg48_1, arg49_1, buf48, buf49)
        del arg48_1
        del arg49_1
        del buf40
        del buf48
        del buf49
        buf52 = empty_strided_cpu((32, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_19], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg51_1, reinterpret_tensor(buf51, (32, 768), (768, 1), 0), reinterpret_tensor(arg50_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf52)
        del arg50_1
        del arg51_1
        buf53 = reinterpret_tensor(buf52, (1, 32, 3072), (98304, 3072, 1), 0); del buf52  # reuse
        cpp_fused_gelu_view_2(buf53)
        buf54 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20, hidden_states_21], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg53_1, reinterpret_tensor(buf53, (32, 3072), (3072, 1), 0), reinterpret_tensor(arg52_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf54)
        del arg52_1
        del arg53_1
        del buf53
        buf55 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf56 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf58 = reinterpret_tensor(buf54, (1, 32, 768), (24576, 768, 1), 0); del buf54  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf58, buf51, arg54_1, arg55_1, buf55, buf56)
        del arg54_1
        del arg55_1
        buf59 = reinterpret_tensor(buf51, (32, 768), (768, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_3], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg57_1, reinterpret_tensor(buf58, (32, 768), (768, 1), 0), reinterpret_tensor(arg56_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf59)
        del arg56_1
        del arg57_1
        buf60 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg59_1, reinterpret_tensor(buf58, (32, 768), (768, 1), 0), reinterpret_tensor(arg58_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf60)
        del arg58_1
        del arg59_1
        buf61 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg61_1, reinterpret_tensor(buf58, (32, 768), (768, 1), 0), reinterpret_tensor(arg60_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf61)
        del arg60_1
        del arg61_1
        # Topologically Sorted Source Nodes: [mixed_query_layer_3, x_11, query_layer_3, linear_19, x_9, key_layer_3, linear_20, x_10, value_layer_3], Original ATen: [aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        buf62 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf59, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf60, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf61, (1, 12, 32, 64), (24576, 64, 768, 1), 0), scale=0.125)
        del buf59
        del buf60
        buf63 = buf62[0]
        assert_size_stride(buf63, (1, 12, 32, 64), (24576, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf63, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf62
        buf65 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [permute_15, context_layer_11, hidden_states_24], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg63_1, reinterpret_tensor(buf63, (32, 768), (768, 1), 0), reinterpret_tensor(arg62_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf65)
        del arg62_1
        del arg63_1
        del buf63
        buf66 = buf56; del buf56  # reuse
        buf67 = buf55; del buf55  # reuse
        buf69 = reinterpret_tensor(buf65, (1, 32, 768), (24576, 768, 1), 0); del buf65  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf69, buf58, arg64_1, arg65_1, buf66, buf67)
        del arg64_1
        del arg65_1
        del buf58
        del buf66
        del buf67
        buf70 = empty_strided_cpu((32, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_27], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg67_1, reinterpret_tensor(buf69, (32, 768), (768, 1), 0), reinterpret_tensor(arg66_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf70)
        del arg66_1
        del arg67_1
        buf71 = reinterpret_tensor(buf70, (1, 32, 3072), (98304, 3072, 1), 0); del buf70  # reuse
        cpp_fused_gelu_view_2(buf71)
        buf72 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_27, hidden_states_28, hidden_states_29], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg69_1, reinterpret_tensor(buf71, (32, 3072), (3072, 1), 0), reinterpret_tensor(arg68_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf72)
        del arg68_1
        del arg69_1
        del buf71
        buf73 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf74 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf76 = reinterpret_tensor(buf72, (1, 32, 768), (24576, 768, 1), 0); del buf72  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf76, buf69, arg70_1, arg71_1, buf73, buf74)
        del arg70_1
        del arg71_1
        buf77 = reinterpret_tensor(buf69, (32, 768), (768, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_4], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg73_1, reinterpret_tensor(buf76, (32, 768), (768, 1), 0), reinterpret_tensor(arg72_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf77)
        del arg72_1
        del arg73_1
        buf78 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg75_1, reinterpret_tensor(buf76, (32, 768), (768, 1), 0), reinterpret_tensor(arg74_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf78)
        del arg74_1
        del arg75_1
        buf79 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg77_1, reinterpret_tensor(buf76, (32, 768), (768, 1), 0), reinterpret_tensor(arg76_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf79)
        del arg76_1
        del arg77_1
        # Topologically Sorted Source Nodes: [mixed_query_layer_4, x_14, query_layer_4, linear_25, x_12, key_layer_4, linear_26, x_13, value_layer_4], Original ATen: [aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        buf80 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf77, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf78, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf79, (1, 12, 32, 64), (24576, 64, 768, 1), 0), scale=0.125)
        del buf77
        del buf78
        buf81 = buf80[0]
        assert_size_stride(buf81, (1, 12, 32, 64), (24576, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf81, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf80
        buf83 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [permute_19, context_layer_14, hidden_states_32], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg79_1, reinterpret_tensor(buf81, (32, 768), (768, 1), 0), reinterpret_tensor(arg78_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf83)
        del arg78_1
        del arg79_1
        del buf81
        buf84 = buf74; del buf74  # reuse
        buf85 = buf73; del buf73  # reuse
        buf87 = reinterpret_tensor(buf83, (1, 32, 768), (24576, 768, 1), 0); del buf83  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf87, buf76, arg80_1, arg81_1, buf84, buf85)
        del arg80_1
        del arg81_1
        del buf76
        del buf84
        del buf85
        buf88 = empty_strided_cpu((32, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_35], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg83_1, reinterpret_tensor(buf87, (32, 768), (768, 1), 0), reinterpret_tensor(arg82_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf88)
        del arg82_1
        del arg83_1
        buf89 = reinterpret_tensor(buf88, (1, 32, 3072), (98304, 3072, 1), 0); del buf88  # reuse
        cpp_fused_gelu_view_2(buf89)
        buf90 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_36, hidden_states_37], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg85_1, reinterpret_tensor(buf89, (32, 3072), (3072, 1), 0), reinterpret_tensor(arg84_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf90)
        del arg84_1
        del arg85_1
        del buf89
        buf91 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf92 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf94 = reinterpret_tensor(buf90, (1, 32, 768), (24576, 768, 1), 0); del buf90  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf94, buf87, arg86_1, arg87_1, buf91, buf92)
        del arg86_1
        del arg87_1
        buf95 = reinterpret_tensor(buf87, (32, 768), (768, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_5], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg89_1, reinterpret_tensor(buf94, (32, 768), (768, 1), 0), reinterpret_tensor(arg88_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf95)
        del arg88_1
        del arg89_1
        buf96 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg91_1, reinterpret_tensor(buf94, (32, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf96)
        del arg90_1
        del arg91_1
        buf97 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg93_1, reinterpret_tensor(buf94, (32, 768), (768, 1), 0), reinterpret_tensor(arg92_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf97)
        del arg92_1
        del arg93_1
        # Topologically Sorted Source Nodes: [mixed_query_layer_5, x_17, query_layer_5, linear_31, x_15, key_layer_5, linear_32, x_16, value_layer_5], Original ATen: [aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        buf98 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf95, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf96, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf97, (1, 12, 32, 64), (24576, 64, 768, 1), 0), scale=0.125)
        del buf95
        del buf96
        buf99 = buf98[0]
        assert_size_stride(buf99, (1, 12, 32, 64), (24576, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf99, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf98
        buf101 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [permute_23, context_layer_17, hidden_states_40], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg95_1, reinterpret_tensor(buf99, (32, 768), (768, 1), 0), reinterpret_tensor(arg94_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf101)
        del arg94_1
        del arg95_1
        del buf99
        buf102 = buf92; del buf92  # reuse
        buf103 = buf91; del buf91  # reuse
        buf105 = reinterpret_tensor(buf101, (1, 32, 768), (24576, 768, 1), 0); del buf101  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf105, buf94, arg96_1, arg97_1, buf102, buf103)
        del arg96_1
        del arg97_1
        del buf102
        del buf103
        del buf94
        buf106 = empty_strided_cpu((32, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_43], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg99_1, reinterpret_tensor(buf105, (32, 768), (768, 1), 0), reinterpret_tensor(arg98_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf106)
        del arg98_1
        del arg99_1
        buf107 = reinterpret_tensor(buf106, (1, 32, 3072), (98304, 3072, 1), 0); del buf106  # reuse
        cpp_fused_gelu_view_2(buf107)
        buf108 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_43, hidden_states_44, hidden_states_45], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg101_1, reinterpret_tensor(buf107, (32, 3072), (3072, 1), 0), reinterpret_tensor(arg100_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf108)
        del arg100_1
        del arg101_1
        del buf107
        buf109 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf110 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf112 = reinterpret_tensor(buf108, (1, 32, 768), (24576, 768, 1), 0); del buf108  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf112, buf105, arg102_1, arg103_1, buf109, buf110)
        del arg102_1
        del arg103_1
        buf113 = reinterpret_tensor(buf105, (32, 768), (768, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_6], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg105_1, reinterpret_tensor(buf112, (32, 768), (768, 1), 0), reinterpret_tensor(arg104_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf113)
        del arg104_1
        del arg105_1
        buf114 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg107_1, reinterpret_tensor(buf112, (32, 768), (768, 1), 0), reinterpret_tensor(arg106_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf114)
        del arg106_1
        del arg107_1
        buf115 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg109_1, reinterpret_tensor(buf112, (32, 768), (768, 1), 0), reinterpret_tensor(arg108_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf115)
        del arg108_1
        del arg109_1
        # Topologically Sorted Source Nodes: [mixed_query_layer_6, x_20, query_layer_6, linear_37, x_18, key_layer_6, linear_38, x_19, value_layer_6], Original ATen: [aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        buf116 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf113, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf114, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf115, (1, 12, 32, 64), (24576, 64, 768, 1), 0), scale=0.125)
        del buf113
        del buf114
        buf117 = buf116[0]
        assert_size_stride(buf117, (1, 12, 32, 64), (24576, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf117, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf116
        buf119 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [permute_27, context_layer_20, hidden_states_48], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg111_1, reinterpret_tensor(buf117, (32, 768), (768, 1), 0), reinterpret_tensor(arg110_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf119)
        del arg110_1
        del arg111_1
        del buf117
        buf120 = buf110; del buf110  # reuse
        buf121 = buf109; del buf109  # reuse
        buf123 = reinterpret_tensor(buf119, (1, 32, 768), (24576, 768, 1), 0); del buf119  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf123, buf112, arg112_1, arg113_1, buf120, buf121)
        del arg112_1
        del arg113_1
        del buf112
        del buf120
        del buf121
        buf124 = empty_strided_cpu((32, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_51], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg115_1, reinterpret_tensor(buf123, (32, 768), (768, 1), 0), reinterpret_tensor(arg114_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf124)
        del arg114_1
        del arg115_1
        buf125 = reinterpret_tensor(buf124, (1, 32, 3072), (98304, 3072, 1), 0); del buf124  # reuse
        cpp_fused_gelu_view_2(buf125)
        buf126 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_51, hidden_states_52, hidden_states_53], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg117_1, reinterpret_tensor(buf125, (32, 3072), (3072, 1), 0), reinterpret_tensor(arg116_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf126)
        del arg116_1
        del arg117_1
        del buf125
        buf127 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf128 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf130 = reinterpret_tensor(buf126, (1, 32, 768), (24576, 768, 1), 0); del buf126  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf130, buf123, arg118_1, arg119_1, buf127, buf128)
        del arg118_1
        del arg119_1
        buf131 = reinterpret_tensor(buf123, (32, 768), (768, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_7], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg121_1, reinterpret_tensor(buf130, (32, 768), (768, 1), 0), reinterpret_tensor(arg120_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf131)
        del arg120_1
        del arg121_1
        buf132 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg123_1, reinterpret_tensor(buf130, (32, 768), (768, 1), 0), reinterpret_tensor(arg122_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf132)
        del arg122_1
        del arg123_1
        buf133 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg125_1, reinterpret_tensor(buf130, (32, 768), (768, 1), 0), reinterpret_tensor(arg124_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf133)
        del arg124_1
        del arg125_1
        # Topologically Sorted Source Nodes: [mixed_query_layer_7, x_23, query_layer_7, linear_43, x_21, key_layer_7, linear_44, x_22, value_layer_7], Original ATen: [aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        buf134 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf131, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf132, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf133, (1, 12, 32, 64), (24576, 64, 768, 1), 0), scale=0.125)
        del buf131
        del buf132
        buf135 = buf134[0]
        assert_size_stride(buf135, (1, 12, 32, 64), (24576, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf135, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf134
        buf137 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [permute_31, context_layer_23, hidden_states_56], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg127_1, reinterpret_tensor(buf135, (32, 768), (768, 1), 0), reinterpret_tensor(arg126_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf137)
        del arg126_1
        del arg127_1
        del buf135
        buf138 = buf128; del buf128  # reuse
        buf139 = buf127; del buf127  # reuse
        buf141 = reinterpret_tensor(buf137, (1, 32, 768), (24576, 768, 1), 0); del buf137  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf141, buf130, arg128_1, arg129_1, buf138, buf139)
        del arg128_1
        del arg129_1
        del buf130
        del buf138
        del buf139
        buf142 = empty_strided_cpu((32, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_59], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg131_1, reinterpret_tensor(buf141, (32, 768), (768, 1), 0), reinterpret_tensor(arg130_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf142)
        del arg130_1
        del arg131_1
        buf143 = reinterpret_tensor(buf142, (1, 32, 3072), (98304, 3072, 1), 0); del buf142  # reuse
        cpp_fused_gelu_view_2(buf143)
        buf144 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_59, hidden_states_60, hidden_states_61], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg133_1, reinterpret_tensor(buf143, (32, 3072), (3072, 1), 0), reinterpret_tensor(arg132_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf144)
        del arg132_1
        del arg133_1
        del buf143
        buf145 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf146 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf148 = reinterpret_tensor(buf144, (1, 32, 768), (24576, 768, 1), 0); del buf144  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf148, buf141, arg134_1, arg135_1, buf145, buf146)
        del arg134_1
        del arg135_1
        buf149 = reinterpret_tensor(buf141, (32, 768), (768, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_8], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg137_1, reinterpret_tensor(buf148, (32, 768), (768, 1), 0), reinterpret_tensor(arg136_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf149)
        del arg136_1
        del arg137_1
        buf150 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg139_1, reinterpret_tensor(buf148, (32, 768), (768, 1), 0), reinterpret_tensor(arg138_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf150)
        del arg138_1
        del arg139_1
        buf151 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg141_1, reinterpret_tensor(buf148, (32, 768), (768, 1), 0), reinterpret_tensor(arg140_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf151)
        del arg140_1
        del arg141_1
        # Topologically Sorted Source Nodes: [mixed_query_layer_8, x_26, query_layer_8, linear_49, x_24, key_layer_8, linear_50, x_25, value_layer_8], Original ATen: [aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        buf152 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf149, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf150, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf151, (1, 12, 32, 64), (24576, 64, 768, 1), 0), scale=0.125)
        del buf149
        del buf150
        buf153 = buf152[0]
        assert_size_stride(buf153, (1, 12, 32, 64), (24576, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf153, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf152
        buf155 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [permute_35, context_layer_26, hidden_states_64], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg143_1, reinterpret_tensor(buf153, (32, 768), (768, 1), 0), reinterpret_tensor(arg142_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf155)
        del arg142_1
        del arg143_1
        del buf153
        buf156 = buf146; del buf146  # reuse
        buf157 = buf145; del buf145  # reuse
        buf159 = reinterpret_tensor(buf155, (1, 32, 768), (24576, 768, 1), 0); del buf155  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf159, buf148, arg144_1, arg145_1, buf156, buf157)
        del arg144_1
        del arg145_1
        del buf148
        del buf156
        del buf157
        buf160 = empty_strided_cpu((32, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_67], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg147_1, reinterpret_tensor(buf159, (32, 768), (768, 1), 0), reinterpret_tensor(arg146_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf160)
        del arg146_1
        del arg147_1
        buf161 = reinterpret_tensor(buf160, (1, 32, 3072), (98304, 3072, 1), 0); del buf160  # reuse
        cpp_fused_gelu_view_2(buf161)
        buf162 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_67, hidden_states_68, hidden_states_69], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg149_1, reinterpret_tensor(buf161, (32, 3072), (3072, 1), 0), reinterpret_tensor(arg148_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf162)
        del arg148_1
        del arg149_1
        del buf161
        buf163 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf164 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf166 = reinterpret_tensor(buf162, (1, 32, 768), (24576, 768, 1), 0); del buf162  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf166, buf159, arg150_1, arg151_1, buf163, buf164)
        del arg150_1
        del arg151_1
        buf167 = reinterpret_tensor(buf159, (32, 768), (768, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_9], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg153_1, reinterpret_tensor(buf166, (32, 768), (768, 1), 0), reinterpret_tensor(arg152_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf167)
        del arg152_1
        del arg153_1
        buf168 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_55], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg155_1, reinterpret_tensor(buf166, (32, 768), (768, 1), 0), reinterpret_tensor(arg154_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf168)
        del arg154_1
        del arg155_1
        buf169 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg157_1, reinterpret_tensor(buf166, (32, 768), (768, 1), 0), reinterpret_tensor(arg156_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf169)
        del arg156_1
        del arg157_1
        # Topologically Sorted Source Nodes: [mixed_query_layer_9, x_29, query_layer_9, linear_55, x_27, key_layer_9, linear_56, x_28, value_layer_9], Original ATen: [aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        buf170 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf167, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf168, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf169, (1, 12, 32, 64), (24576, 64, 768, 1), 0), scale=0.125)
        del buf167
        del buf168
        buf171 = buf170[0]
        assert_size_stride(buf171, (1, 12, 32, 64), (24576, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf171, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf170
        buf173 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [permute_39, context_layer_29, hidden_states_72], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg159_1, reinterpret_tensor(buf171, (32, 768), (768, 1), 0), reinterpret_tensor(arg158_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf173)
        del arg158_1
        del arg159_1
        del buf171
        buf174 = buf164; del buf164  # reuse
        buf175 = buf163; del buf163  # reuse
        buf177 = reinterpret_tensor(buf173, (1, 32, 768), (24576, 768, 1), 0); del buf173  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf177, buf166, arg160_1, arg161_1, buf174, buf175)
        del arg160_1
        del arg161_1
        del buf166
        del buf174
        del buf175
        buf178 = empty_strided_cpu((32, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_75], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg163_1, reinterpret_tensor(buf177, (32, 768), (768, 1), 0), reinterpret_tensor(arg162_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf178)
        del arg162_1
        del arg163_1
        buf179 = reinterpret_tensor(buf178, (1, 32, 3072), (98304, 3072, 1), 0); del buf178  # reuse
        cpp_fused_gelu_view_2(buf179)
        buf180 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_75, hidden_states_76, hidden_states_77], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg165_1, reinterpret_tensor(buf179, (32, 3072), (3072, 1), 0), reinterpret_tensor(arg164_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf180)
        del arg164_1
        del arg165_1
        del buf179
        buf181 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf182 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf184 = reinterpret_tensor(buf180, (1, 32, 768), (24576, 768, 1), 0); del buf180  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf184, buf177, arg166_1, arg167_1, buf181, buf182)
        del arg166_1
        del arg167_1
        buf185 = reinterpret_tensor(buf177, (32, 768), (768, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_10], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg169_1, reinterpret_tensor(buf184, (32, 768), (768, 1), 0), reinterpret_tensor(arg168_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf185)
        del arg168_1
        del arg169_1
        buf186 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg171_1, reinterpret_tensor(buf184, (32, 768), (768, 1), 0), reinterpret_tensor(arg170_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf186)
        del arg170_1
        del arg171_1
        buf187 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_62], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg173_1, reinterpret_tensor(buf184, (32, 768), (768, 1), 0), reinterpret_tensor(arg172_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf187)
        del arg172_1
        del arg173_1
        # Topologically Sorted Source Nodes: [mixed_query_layer_10, x_32, query_layer_10, linear_61, x_30, key_layer_10, linear_62, x_31, value_layer_10], Original ATen: [aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        buf188 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf185, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf186, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf187, (1, 12, 32, 64), (24576, 64, 768, 1), 0), scale=0.125)
        del buf185
        del buf186
        buf189 = buf188[0]
        assert_size_stride(buf189, (1, 12, 32, 64), (24576, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf189, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf188
        buf191 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [permute_43, context_layer_32, hidden_states_80], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg175_1, reinterpret_tensor(buf189, (32, 768), (768, 1), 0), reinterpret_tensor(arg174_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf191)
        del arg174_1
        del arg175_1
        del buf189
        buf192 = buf182; del buf182  # reuse
        buf193 = buf181; del buf181  # reuse
        buf195 = reinterpret_tensor(buf191, (1, 32, 768), (24576, 768, 1), 0); del buf191  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf195, buf184, arg176_1, arg177_1, buf192, buf193)
        del arg176_1
        del arg177_1
        del buf184
        del buf192
        del buf193
        buf196 = empty_strided_cpu((32, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_83], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg179_1, reinterpret_tensor(buf195, (32, 768), (768, 1), 0), reinterpret_tensor(arg178_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf196)
        del arg178_1
        del arg179_1
        buf197 = reinterpret_tensor(buf196, (1, 32, 3072), (98304, 3072, 1), 0); del buf196  # reuse
        cpp_fused_gelu_view_2(buf197)
        buf198 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_83, hidden_states_84, hidden_states_85], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg181_1, reinterpret_tensor(buf197, (32, 3072), (3072, 1), 0), reinterpret_tensor(arg180_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf198)
        del arg180_1
        del arg181_1
        del buf197
        buf199 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf200 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf202 = reinterpret_tensor(buf198, (1, 32, 768), (24576, 768, 1), 0); del buf198  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf202, buf195, arg182_1, arg183_1, buf199, buf200)
        del arg182_1
        del arg183_1
        buf203 = reinterpret_tensor(buf195, (32, 768), (768, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_11], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg185_1, reinterpret_tensor(buf202, (32, 768), (768, 1), 0), reinterpret_tensor(arg184_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf203)
        del arg184_1
        del arg185_1
        buf204 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_67], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg187_1, reinterpret_tensor(buf202, (32, 768), (768, 1), 0), reinterpret_tensor(arg186_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf204)
        del arg186_1
        del arg187_1
        buf205 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg189_1, reinterpret_tensor(buf202, (32, 768), (768, 1), 0), reinterpret_tensor(arg188_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf205)
        del arg188_1
        del arg189_1
        # Topologically Sorted Source Nodes: [mixed_query_layer_11, x_35, query_layer_11, linear_67, x_33, key_layer_11, linear_68, x_34, value_layer_11], Original ATen: [aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        buf206 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf203, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf204, (1, 12, 32, 64), (24576, 64, 768, 1), 0), reinterpret_tensor(buf205, (1, 12, 32, 64), (24576, 64, 768, 1), 0), scale=0.125)
        del buf203
        del buf204
        buf207 = buf206[0]
        assert_size_stride(buf207, (1, 12, 32, 64), (24576, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf207, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf206
        buf209 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [permute_47, context_layer_35, hidden_states_88], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg191_1, reinterpret_tensor(buf207, (32, 768), (768, 1), 0), reinterpret_tensor(arg190_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf209)
        del arg190_1
        del arg191_1
        del buf207
        buf210 = buf200; del buf200  # reuse
        buf211 = buf199; del buf199  # reuse
        buf213 = reinterpret_tensor(buf209, (1, 32, 768), (24576, 768, 1), 0); del buf209  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf213, buf202, arg192_1, arg193_1, buf210, buf211)
        del arg192_1
        del arg193_1
        del buf202
        del buf210
        del buf211
        buf214 = empty_strided_cpu((32, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_91], Original ATen: [aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(arg195_1, reinterpret_tensor(buf213, (32, 768), (768, 1), 0), reinterpret_tensor(arg194_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf214)
        del arg194_1
        del arg195_1
        buf215 = reinterpret_tensor(buf214, (1, 32, 3072), (98304, 3072, 1), 0); del buf214  # reuse
        cpp_fused_gelu_view_2(buf215)
        buf216 = empty_strided_cpu((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_91, hidden_states_92, hidden_states_93], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(arg197_1, reinterpret_tensor(buf215, (32, 3072), (3072, 1), 0), reinterpret_tensor(arg196_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf216)
        del arg196_1
        del arg197_1
        del buf215
        buf217 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf218 = empty_strided_cpu((1, 32, 1), (32, 1, 32), torch.float32)
        buf220 = reinterpret_tensor(buf216, (1, 32, 768), (24576, 768, 1), 0); del buf216  # reuse
        cpp_fused_add_native_layer_norm_view_1(buf220, buf213, arg198_1, arg199_1, buf217, buf218)
        del arg198_1
        del arg199_1
        del buf213
        del buf217
        del buf218
        buf221 = empty_strided_cpu((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [first_token_tensor, pooled_output], Original ATen: [aten.select, aten.t, aten.addmm]
        extern_kernels.addmm(arg201_1, reinterpret_tensor(buf220, (1, 768), (768, 1), 0), reinterpret_tensor(arg200_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf221)
        del arg200_1
        del arg201_1
        buf222 = buf221; del buf221  # reuse
        cpp_fused_tanh_3(buf222)
        return (buf220, buf222, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 32), (32, 1), device='cpu', dtype=torch.int64)
    arg1_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg2_1 = rand_strided((30522, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg4_1 = rand_strided((2, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
