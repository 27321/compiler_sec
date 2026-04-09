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


cpp_fused_add_embedding_native_layer_norm_0 = async_compile.cpp_pybinding(['const int64_t*', 'const float*', 'const int64_t*', 'const float*', 'const int64_t*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*'], r'''
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
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(1L))
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


cpp_fused__scaled_dot_product_flash_attention_for_cpu__to_copy_mul_permute_rsub_unsqueeze_view_1 = async_compile.cpp_pybinding(['const int64_t*', 'float*', 'float*', 'float*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const int64_t* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(16L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(512L)))
                {
                    auto tmp0 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                    auto tmp1 = at::vec::convert<float,1,int64_t,2>(tmp0);
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp3 - tmp1;
                    auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    tmp7.store(out_ptr0 + static_cast<int64_t>(x0));
                    tmp7.store(out_ptr1 + static_cast<int64_t>(x0));
                    tmp7.store(out_ptr2 + static_cast<int64_t>(x0));
                    tmp7.store(out_ptr3 + static_cast<int64_t>(x0));
                    tmp7.store(out_ptr4 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_2 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*'], r'''
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
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(1L))
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


cpp_fused_gelu_view_3 = async_compile.cpp_pybinding(['float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(32)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(1572864L); x0+=static_cast<int64_t>(16L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(1572864L)))
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


cpp_fused__scaled_dot_product_flash_attention_for_cpu__to_copy_mul_permute_rsub_unsqueeze_view_4 = async_compile.cpp_pybinding(['const int64_t*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const int64_t* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(16L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(512L)))
                {
                    auto tmp0 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                    auto tmp1 = at::vec::convert<float,1,int64_t,2>(tmp0);
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp3 - tmp1;
                    auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    tmp7.store(out_ptr0 + static_cast<int64_t>(x0));
                    tmp7.store(out_ptr1 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_add_bitwise_and_bitwise_or_eq_ge_native_layer_norm_select_sum_unsqueeze_view_zeros_5 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'const int64_t*', 'float*', 'float*', 'int64_t*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const int64_t* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       int64_t* out_ptr2,
                       float* out_ptr4)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(32)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(1L))
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
        #pragma omp single
        {
            {
                {
                    int64_t tmp_acc0 = 0;
                    at::vec::VectorizedN<int64_t,2> tmp_acc0_vec = at::vec::VectorizedN<int64_t,2>(0);
                    for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(16L))
                    {
                        {
                            if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(512L)))
                            {
                                auto tmp0 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr4 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                                auto tmp1 = static_cast<int64_t>(1998);
                                auto tmp2 = at::vec::VectorizedN<int64_t,2>(tmp1);
                                auto tmp3 = at::vec::VecMask<int64_t,2>(tmp0 == tmp2);
                                auto tmp4 = tmp3.to<int64_t,2>();
                                tmp_acc0_vec = tmp_acc0_vec + tmp4;
                            }
                        }
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<int64_t, 2>([](at::vec::Vectorized<int64_t>& x, at::vec::Vectorized<int64_t>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<int64_t>(0L)] = static_cast<int64_t>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                {
                    {
                        auto tmp0 = in_ptr4[static_cast<int64_t>(0L)];
                        auto tmp3 = in_ptr4[static_cast<int64_t>(2L)];
                        auto tmp6 = in_ptr4[static_cast<int64_t>(5L)];
                        auto tmp9 = in_ptr4[static_cast<int64_t>(6L)];
                        auto tmp12 = in_ptr4[static_cast<int64_t>(10L)];
                        auto tmp15 = in_ptr4[static_cast<int64_t>(12L)];
                        auto tmp18 = in_ptr4[static_cast<int64_t>(15L)];
                        auto tmp21 = in_ptr4[static_cast<int64_t>(16L)];
                        auto tmp26 = in_ptr4[static_cast<int64_t>(1L)];
                        auto tmp28 = in_ptr4[static_cast<int64_t>(3L)];
                        auto tmp32 = in_ptr4[static_cast<int64_t>(7L)];
                        auto tmp35 = in_ptr4[static_cast<int64_t>(11L)];
                        auto tmp38 = in_ptr4[static_cast<int64_t>(13L)];
                        auto tmp42 = in_ptr4[static_cast<int64_t>(17L)];
                        auto tmp46 = in_ptr4[static_cast<int64_t>(4L)];
                        auto tmp50 = in_ptr4[static_cast<int64_t>(8L)];
                        auto tmp54 = in_ptr4[static_cast<int64_t>(14L)];
                        auto tmp58 = in_ptr4[static_cast<int64_t>(18L)];
                        auto tmp64 = in_ptr4[static_cast<int64_t>(9L)];
                        auto tmp70 = in_ptr4[static_cast<int64_t>(19L)];
                        auto tmp80 = in_ptr4[static_cast<int64_t>(20L)];
                        auto tmp90 = in_ptr4[static_cast<int64_t>(21L)];
                        auto tmp100 = in_ptr4[static_cast<int64_t>(22L)];
                        auto tmp110 = in_ptr4[static_cast<int64_t>(23L)];
                        auto tmp120 = in_ptr4[static_cast<int64_t>(24L)];
                        auto tmp130 = in_ptr4[static_cast<int64_t>(25L)];
                        auto tmp140 = in_ptr4[static_cast<int64_t>(26L)];
                        auto tmp150 = in_ptr4[static_cast<int64_t>(27L)];
                        auto tmp160 = in_ptr4[static_cast<int64_t>(28L)];
                        auto tmp170 = in_ptr4[static_cast<int64_t>(29L)];
                        auto tmp180 = in_ptr4[static_cast<int64_t>(30L)];
                        auto tmp190 = in_ptr4[static_cast<int64_t>(31L)];
                        auto tmp200 = in_ptr4[static_cast<int64_t>(32L)];
                        auto tmp210 = in_ptr4[static_cast<int64_t>(33L)];
                        auto tmp220 = in_ptr4[static_cast<int64_t>(34L)];
                        auto tmp230 = in_ptr4[static_cast<int64_t>(35L)];
                        auto tmp240 = in_ptr4[static_cast<int64_t>(36L)];
                        auto tmp250 = in_ptr4[static_cast<int64_t>(37L)];
                        auto tmp260 = in_ptr4[static_cast<int64_t>(38L)];
                        auto tmp270 = in_ptr4[static_cast<int64_t>(39L)];
                        auto tmp280 = in_ptr4[static_cast<int64_t>(40L)];
                        auto tmp290 = in_ptr4[static_cast<int64_t>(41L)];
                        auto tmp300 = in_ptr4[static_cast<int64_t>(42L)];
                        auto tmp310 = in_ptr4[static_cast<int64_t>(43L)];
                        auto tmp320 = in_ptr4[static_cast<int64_t>(44L)];
                        auto tmp330 = in_ptr4[static_cast<int64_t>(45L)];
                        auto tmp340 = in_ptr4[static_cast<int64_t>(46L)];
                        auto tmp350 = in_ptr4[static_cast<int64_t>(47L)];
                        auto tmp360 = in_ptr4[static_cast<int64_t>(48L)];
                        auto tmp370 = in_ptr4[static_cast<int64_t>(49L)];
                        auto tmp380 = in_ptr4[static_cast<int64_t>(50L)];
                        auto tmp390 = in_ptr4[static_cast<int64_t>(51L)];
                        auto tmp400 = in_ptr4[static_cast<int64_t>(52L)];
                        auto tmp410 = in_ptr4[static_cast<int64_t>(53L)];
                        auto tmp420 = in_ptr4[static_cast<int64_t>(54L)];
                        auto tmp430 = in_ptr4[static_cast<int64_t>(55L)];
                        auto tmp440 = in_ptr4[static_cast<int64_t>(56L)];
                        auto tmp450 = in_ptr4[static_cast<int64_t>(57L)];
                        auto tmp460 = in_ptr4[static_cast<int64_t>(58L)];
                        auto tmp470 = in_ptr4[static_cast<int64_t>(59L)];
                        auto tmp480 = in_ptr4[static_cast<int64_t>(60L)];
                        auto tmp490 = in_ptr4[static_cast<int64_t>(61L)];
                        auto tmp500 = in_ptr4[static_cast<int64_t>(62L)];
                        auto tmp510 = in_ptr4[static_cast<int64_t>(63L)];
                        auto tmp520 = in_ptr4[static_cast<int64_t>(64L)];
                        auto tmp530 = in_ptr4[static_cast<int64_t>(65L)];
                        auto tmp540 = in_ptr4[static_cast<int64_t>(66L)];
                        auto tmp550 = in_ptr4[static_cast<int64_t>(67L)];
                        auto tmp560 = in_ptr4[static_cast<int64_t>(68L)];
                        auto tmp570 = in_ptr4[static_cast<int64_t>(69L)];
                        auto tmp580 = in_ptr4[static_cast<int64_t>(70L)];
                        auto tmp590 = in_ptr4[static_cast<int64_t>(71L)];
                        auto tmp600 = in_ptr4[static_cast<int64_t>(72L)];
                        auto tmp610 = in_ptr4[static_cast<int64_t>(73L)];
                        auto tmp620 = in_ptr4[static_cast<int64_t>(74L)];
                        auto tmp630 = in_ptr4[static_cast<int64_t>(75L)];
                        auto tmp640 = in_ptr4[static_cast<int64_t>(76L)];
                        auto tmp650 = in_ptr4[static_cast<int64_t>(77L)];
                        auto tmp660 = in_ptr4[static_cast<int64_t>(78L)];
                        auto tmp670 = in_ptr4[static_cast<int64_t>(79L)];
                        auto tmp680 = in_ptr4[static_cast<int64_t>(80L)];
                        auto tmp690 = in_ptr4[static_cast<int64_t>(81L)];
                        auto tmp700 = in_ptr4[static_cast<int64_t>(82L)];
                        auto tmp710 = in_ptr4[static_cast<int64_t>(83L)];
                        auto tmp720 = in_ptr4[static_cast<int64_t>(84L)];
                        auto tmp730 = in_ptr4[static_cast<int64_t>(85L)];
                        auto tmp740 = in_ptr4[static_cast<int64_t>(86L)];
                        auto tmp750 = in_ptr4[static_cast<int64_t>(87L)];
                        auto tmp760 = in_ptr4[static_cast<int64_t>(88L)];
                        auto tmp770 = in_ptr4[static_cast<int64_t>(89L)];
                        auto tmp780 = in_ptr4[static_cast<int64_t>(90L)];
                        auto tmp790 = in_ptr4[static_cast<int64_t>(91L)];
                        auto tmp800 = in_ptr4[static_cast<int64_t>(92L)];
                        auto tmp810 = in_ptr4[static_cast<int64_t>(93L)];
                        auto tmp820 = in_ptr4[static_cast<int64_t>(94L)];
                        auto tmp830 = in_ptr4[static_cast<int64_t>(95L)];
                        auto tmp840 = in_ptr4[static_cast<int64_t>(96L)];
                        auto tmp850 = in_ptr4[static_cast<int64_t>(97L)];
                        auto tmp860 = in_ptr4[static_cast<int64_t>(98L)];
                        auto tmp870 = in_ptr4[static_cast<int64_t>(99L)];
                        auto tmp880 = in_ptr4[static_cast<int64_t>(100L)];
                        auto tmp890 = in_ptr4[static_cast<int64_t>(101L)];
                        auto tmp900 = in_ptr4[static_cast<int64_t>(102L)];
                        auto tmp910 = in_ptr4[static_cast<int64_t>(103L)];
                        auto tmp920 = in_ptr4[static_cast<int64_t>(104L)];
                        auto tmp930 = in_ptr4[static_cast<int64_t>(105L)];
                        auto tmp940 = in_ptr4[static_cast<int64_t>(106L)];
                        auto tmp950 = in_ptr4[static_cast<int64_t>(107L)];
                        auto tmp960 = in_ptr4[static_cast<int64_t>(108L)];
                        auto tmp970 = in_ptr4[static_cast<int64_t>(109L)];
                        auto tmp980 = in_ptr4[static_cast<int64_t>(110L)];
                        auto tmp990 = in_ptr4[static_cast<int64_t>(111L)];
                        auto tmp1000 = in_ptr4[static_cast<int64_t>(112L)];
                        auto tmp1010 = in_ptr4[static_cast<int64_t>(113L)];
                        auto tmp1020 = in_ptr4[static_cast<int64_t>(114L)];
                        auto tmp1030 = in_ptr4[static_cast<int64_t>(115L)];
                        auto tmp1034 = out_ptr2[static_cast<int64_t>(0L)];
                        auto tmp1 = static_cast<int64_t>(1998);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp4 = tmp3 == tmp1;
                        auto tmp5 = decltype(tmp2)(tmp2 & tmp4);
                        auto tmp7 = tmp6 == tmp1;
                        auto tmp8 = decltype(tmp5)(tmp5 & tmp7);
                        auto tmp10 = tmp9 == tmp1;
                        auto tmp11 = decltype(tmp8)(tmp8 & tmp10);
                        auto tmp13 = tmp12 == tmp1;
                        auto tmp14 = decltype(tmp11)(tmp11 & tmp13);
                        auto tmp16 = tmp15 == tmp1;
                        auto tmp17 = decltype(tmp14)(tmp14 & tmp16);
                        auto tmp19 = tmp18 == tmp1;
                        auto tmp20 = decltype(tmp17)(tmp17 & tmp19);
                        auto tmp22 = tmp21 == tmp1;
                        auto tmp23 = decltype(tmp20)(tmp20 & tmp22);
                        auto tmp24 = static_cast<bool>(false);
                        auto tmp25 = decltype(tmp24)(tmp24 | tmp23);
                        auto tmp27 = tmp26 == tmp1;
                        auto tmp29 = tmp28 == tmp1;
                        auto tmp30 = decltype(tmp27)(tmp27 & tmp29);
                        auto tmp31 = decltype(tmp30)(tmp30 & tmp10);
                        auto tmp33 = tmp32 == tmp1;
                        auto tmp34 = decltype(tmp31)(tmp31 & tmp33);
                        auto tmp36 = tmp35 == tmp1;
                        auto tmp37 = decltype(tmp34)(tmp34 & tmp36);
                        auto tmp39 = tmp38 == tmp1;
                        auto tmp40 = decltype(tmp37)(tmp37 & tmp39);
                        auto tmp41 = decltype(tmp40)(tmp40 & tmp22);
                        auto tmp43 = tmp42 == tmp1;
                        auto tmp44 = decltype(tmp41)(tmp41 & tmp43);
                        auto tmp45 = decltype(tmp25)(tmp25 | tmp44);
                        auto tmp47 = tmp46 == tmp1;
                        auto tmp48 = decltype(tmp4)(tmp4 & tmp47);
                        auto tmp49 = decltype(tmp48)(tmp48 & tmp33);
                        auto tmp51 = tmp50 == tmp1;
                        auto tmp52 = decltype(tmp49)(tmp49 & tmp51);
                        auto tmp53 = decltype(tmp52)(tmp52 & tmp16);
                        auto tmp55 = tmp54 == tmp1;
                        auto tmp56 = decltype(tmp53)(tmp53 & tmp55);
                        auto tmp57 = decltype(tmp56)(tmp56 & tmp43);
                        auto tmp59 = tmp58 == tmp1;
                        auto tmp60 = decltype(tmp57)(tmp57 & tmp59);
                        auto tmp61 = decltype(tmp45)(tmp45 | tmp60);
                        auto tmp62 = decltype(tmp29)(tmp29 & tmp7);
                        auto tmp63 = decltype(tmp62)(tmp62 & tmp51);
                        auto tmp65 = tmp64 == tmp1;
                        auto tmp66 = decltype(tmp63)(tmp63 & tmp65);
                        auto tmp67 = decltype(tmp66)(tmp66 & tmp39);
                        auto tmp68 = decltype(tmp67)(tmp67 & tmp19);
                        auto tmp69 = decltype(tmp68)(tmp68 & tmp59);
                        auto tmp71 = tmp70 == tmp1;
                        auto tmp72 = decltype(tmp69)(tmp69 & tmp71);
                        auto tmp73 = decltype(tmp61)(tmp61 | tmp72);
                        auto tmp74 = decltype(tmp47)(tmp47 & tmp10);
                        auto tmp75 = decltype(tmp74)(tmp74 & tmp65);
                        auto tmp76 = decltype(tmp75)(tmp75 & tmp13);
                        auto tmp77 = decltype(tmp76)(tmp76 & tmp55);
                        auto tmp78 = decltype(tmp77)(tmp77 & tmp22);
                        auto tmp79 = decltype(tmp78)(tmp78 & tmp71);
                        auto tmp81 = tmp80 == tmp1;
                        auto tmp82 = decltype(tmp79)(tmp79 & tmp81);
                        auto tmp83 = decltype(tmp73)(tmp73 | tmp82);
                        auto tmp84 = decltype(tmp7)(tmp7 & tmp33);
                        auto tmp85 = decltype(tmp84)(tmp84 & tmp13);
                        auto tmp86 = decltype(tmp85)(tmp85 & tmp36);
                        auto tmp87 = decltype(tmp86)(tmp86 & tmp19);
                        auto tmp88 = decltype(tmp87)(tmp87 & tmp43);
                        auto tmp89 = decltype(tmp88)(tmp88 & tmp81);
                        auto tmp91 = tmp90 == tmp1;
                        auto tmp92 = decltype(tmp89)(tmp89 & tmp91);
                        auto tmp93 = decltype(tmp83)(tmp83 | tmp92);
                        auto tmp94 = decltype(tmp10)(tmp10 & tmp51);
                        auto tmp95 = decltype(tmp94)(tmp94 & tmp36);
                        auto tmp96 = decltype(tmp95)(tmp95 & tmp16);
                        auto tmp97 = decltype(tmp96)(tmp96 & tmp22);
                        auto tmp98 = decltype(tmp97)(tmp97 & tmp59);
                        auto tmp99 = decltype(tmp98)(tmp98 & tmp91);
                        auto tmp101 = tmp100 == tmp1;
                        auto tmp102 = decltype(tmp99)(tmp99 & tmp101);
                        auto tmp103 = decltype(tmp93)(tmp93 | tmp102);
                        auto tmp104 = decltype(tmp33)(tmp33 & tmp65);
                        auto tmp105 = decltype(tmp104)(tmp104 & tmp16);
                        auto tmp106 = decltype(tmp105)(tmp105 & tmp39);
                        auto tmp107 = decltype(tmp106)(tmp106 & tmp43);
                        auto tmp108 = decltype(tmp107)(tmp107 & tmp71);
                        auto tmp109 = decltype(tmp108)(tmp108 & tmp101);
                        auto tmp111 = tmp110 == tmp1;
                        auto tmp112 = decltype(tmp109)(tmp109 & tmp111);
                        auto tmp113 = decltype(tmp103)(tmp103 | tmp112);
                        auto tmp114 = decltype(tmp51)(tmp51 & tmp13);
                        auto tmp115 = decltype(tmp114)(tmp114 & tmp39);
                        auto tmp116 = decltype(tmp115)(tmp115 & tmp55);
                        auto tmp117 = decltype(tmp116)(tmp116 & tmp59);
                        auto tmp118 = decltype(tmp117)(tmp117 & tmp81);
                        auto tmp119 = decltype(tmp118)(tmp118 & tmp111);
                        auto tmp121 = tmp120 == tmp1;
                        auto tmp122 = decltype(tmp119)(tmp119 & tmp121);
                        auto tmp123 = decltype(tmp113)(tmp113 | tmp122);
                        auto tmp124 = decltype(tmp65)(tmp65 & tmp36);
                        auto tmp125 = decltype(tmp124)(tmp124 & tmp55);
                        auto tmp126 = decltype(tmp125)(tmp125 & tmp19);
                        auto tmp127 = decltype(tmp126)(tmp126 & tmp71);
                        auto tmp128 = decltype(tmp127)(tmp127 & tmp91);
                        auto tmp129 = decltype(tmp128)(tmp128 & tmp121);
                        auto tmp131 = tmp130 == tmp1;
                        auto tmp132 = decltype(tmp129)(tmp129 & tmp131);
                        auto tmp133 = decltype(tmp123)(tmp123 | tmp132);
                        auto tmp134 = decltype(tmp13)(tmp13 & tmp16);
                        auto tmp135 = decltype(tmp134)(tmp134 & tmp19);
                        auto tmp136 = decltype(tmp135)(tmp135 & tmp22);
                        auto tmp137 = decltype(tmp136)(tmp136 & tmp81);
                        auto tmp138 = decltype(tmp137)(tmp137 & tmp101);
                        auto tmp139 = decltype(tmp138)(tmp138 & tmp131);
                        auto tmp141 = tmp140 == tmp1;
                        auto tmp142 = decltype(tmp139)(tmp139 & tmp141);
                        auto tmp143 = decltype(tmp133)(tmp133 | tmp142);
                        auto tmp144 = decltype(tmp36)(tmp36 & tmp39);
                        auto tmp145 = decltype(tmp144)(tmp144 & tmp22);
                        auto tmp146 = decltype(tmp145)(tmp145 & tmp43);
                        auto tmp147 = decltype(tmp146)(tmp146 & tmp91);
                        auto tmp148 = decltype(tmp147)(tmp147 & tmp111);
                        auto tmp149 = decltype(tmp148)(tmp148 & tmp141);
                        auto tmp151 = tmp150 == tmp1;
                        auto tmp152 = decltype(tmp149)(tmp149 & tmp151);
                        auto tmp153 = decltype(tmp143)(tmp143 | tmp152);
                        auto tmp154 = decltype(tmp16)(tmp16 & tmp55);
                        auto tmp155 = decltype(tmp154)(tmp154 & tmp43);
                        auto tmp156 = decltype(tmp155)(tmp155 & tmp59);
                        auto tmp157 = decltype(tmp156)(tmp156 & tmp101);
                        auto tmp158 = decltype(tmp157)(tmp157 & tmp121);
                        auto tmp159 = decltype(tmp158)(tmp158 & tmp151);
                        auto tmp161 = tmp160 == tmp1;
                        auto tmp162 = decltype(tmp159)(tmp159 & tmp161);
                        auto tmp163 = decltype(tmp153)(tmp153 | tmp162);
                        auto tmp164 = decltype(tmp39)(tmp39 & tmp19);
                        auto tmp165 = decltype(tmp164)(tmp164 & tmp59);
                        auto tmp166 = decltype(tmp165)(tmp165 & tmp71);
                        auto tmp167 = decltype(tmp166)(tmp166 & tmp111);
                        auto tmp168 = decltype(tmp167)(tmp167 & tmp131);
                        auto tmp169 = decltype(tmp168)(tmp168 & tmp161);
                        auto tmp171 = tmp170 == tmp1;
                        auto tmp172 = decltype(tmp169)(tmp169 & tmp171);
                        auto tmp173 = decltype(tmp163)(tmp163 | tmp172);
                        auto tmp174 = decltype(tmp55)(tmp55 & tmp22);
                        auto tmp175 = decltype(tmp174)(tmp174 & tmp71);
                        auto tmp176 = decltype(tmp175)(tmp175 & tmp81);
                        auto tmp177 = decltype(tmp176)(tmp176 & tmp121);
                        auto tmp178 = decltype(tmp177)(tmp177 & tmp141);
                        auto tmp179 = decltype(tmp178)(tmp178 & tmp171);
                        auto tmp181 = tmp180 == tmp1;
                        auto tmp182 = decltype(tmp179)(tmp179 & tmp181);
                        auto tmp183 = decltype(tmp173)(tmp173 | tmp182);
                        auto tmp184 = decltype(tmp19)(tmp19 & tmp43);
                        auto tmp185 = decltype(tmp184)(tmp184 & tmp81);
                        auto tmp186 = decltype(tmp185)(tmp185 & tmp91);
                        auto tmp187 = decltype(tmp186)(tmp186 & tmp131);
                        auto tmp188 = decltype(tmp187)(tmp187 & tmp151);
                        auto tmp189 = decltype(tmp188)(tmp188 & tmp181);
                        auto tmp191 = tmp190 == tmp1;
                        auto tmp192 = decltype(tmp189)(tmp189 & tmp191);
                        auto tmp193 = decltype(tmp183)(tmp183 | tmp192);
                        auto tmp194 = decltype(tmp22)(tmp22 & tmp59);
                        auto tmp195 = decltype(tmp194)(tmp194 & tmp91);
                        auto tmp196 = decltype(tmp195)(tmp195 & tmp101);
                        auto tmp197 = decltype(tmp196)(tmp196 & tmp141);
                        auto tmp198 = decltype(tmp197)(tmp197 & tmp161);
                        auto tmp199 = decltype(tmp198)(tmp198 & tmp191);
                        auto tmp201 = tmp200 == tmp1;
                        auto tmp202 = decltype(tmp199)(tmp199 & tmp201);
                        auto tmp203 = decltype(tmp193)(tmp193 | tmp202);
                        auto tmp204 = decltype(tmp43)(tmp43 & tmp71);
                        auto tmp205 = decltype(tmp204)(tmp204 & tmp101);
                        auto tmp206 = decltype(tmp205)(tmp205 & tmp111);
                        auto tmp207 = decltype(tmp206)(tmp206 & tmp151);
                        auto tmp208 = decltype(tmp207)(tmp207 & tmp171);
                        auto tmp209 = decltype(tmp208)(tmp208 & tmp201);
                        auto tmp211 = tmp210 == tmp1;
                        auto tmp212 = decltype(tmp209)(tmp209 & tmp211);
                        auto tmp213 = decltype(tmp203)(tmp203 | tmp212);
                        auto tmp214 = decltype(tmp59)(tmp59 & tmp81);
                        auto tmp215 = decltype(tmp214)(tmp214 & tmp111);
                        auto tmp216 = decltype(tmp215)(tmp215 & tmp121);
                        auto tmp217 = decltype(tmp216)(tmp216 & tmp161);
                        auto tmp218 = decltype(tmp217)(tmp217 & tmp181);
                        auto tmp219 = decltype(tmp218)(tmp218 & tmp211);
                        auto tmp221 = tmp220 == tmp1;
                        auto tmp222 = decltype(tmp219)(tmp219 & tmp221);
                        auto tmp223 = decltype(tmp213)(tmp213 | tmp222);
                        auto tmp224 = decltype(tmp71)(tmp71 & tmp91);
                        auto tmp225 = decltype(tmp224)(tmp224 & tmp121);
                        auto tmp226 = decltype(tmp225)(tmp225 & tmp131);
                        auto tmp227 = decltype(tmp226)(tmp226 & tmp171);
                        auto tmp228 = decltype(tmp227)(tmp227 & tmp191);
                        auto tmp229 = decltype(tmp228)(tmp228 & tmp221);
                        auto tmp231 = tmp230 == tmp1;
                        auto tmp232 = decltype(tmp229)(tmp229 & tmp231);
                        auto tmp233 = decltype(tmp223)(tmp223 | tmp232);
                        auto tmp234 = decltype(tmp81)(tmp81 & tmp101);
                        auto tmp235 = decltype(tmp234)(tmp234 & tmp131);
                        auto tmp236 = decltype(tmp235)(tmp235 & tmp141);
                        auto tmp237 = decltype(tmp236)(tmp236 & tmp181);
                        auto tmp238 = decltype(tmp237)(tmp237 & tmp201);
                        auto tmp239 = decltype(tmp238)(tmp238 & tmp231);
                        auto tmp241 = tmp240 == tmp1;
                        auto tmp242 = decltype(tmp239)(tmp239 & tmp241);
                        auto tmp243 = decltype(tmp233)(tmp233 | tmp242);
                        auto tmp244 = decltype(tmp91)(tmp91 & tmp111);
                        auto tmp245 = decltype(tmp244)(tmp244 & tmp141);
                        auto tmp246 = decltype(tmp245)(tmp245 & tmp151);
                        auto tmp247 = decltype(tmp246)(tmp246 & tmp191);
                        auto tmp248 = decltype(tmp247)(tmp247 & tmp211);
                        auto tmp249 = decltype(tmp248)(tmp248 & tmp241);
                        auto tmp251 = tmp250 == tmp1;
                        auto tmp252 = decltype(tmp249)(tmp249 & tmp251);
                        auto tmp253 = decltype(tmp243)(tmp243 | tmp252);
                        auto tmp254 = decltype(tmp101)(tmp101 & tmp121);
                        auto tmp255 = decltype(tmp254)(tmp254 & tmp151);
                        auto tmp256 = decltype(tmp255)(tmp255 & tmp161);
                        auto tmp257 = decltype(tmp256)(tmp256 & tmp201);
                        auto tmp258 = decltype(tmp257)(tmp257 & tmp221);
                        auto tmp259 = decltype(tmp258)(tmp258 & tmp251);
                        auto tmp261 = tmp260 == tmp1;
                        auto tmp262 = decltype(tmp259)(tmp259 & tmp261);
                        auto tmp263 = decltype(tmp253)(tmp253 | tmp262);
                        auto tmp264 = decltype(tmp111)(tmp111 & tmp131);
                        auto tmp265 = decltype(tmp264)(tmp264 & tmp161);
                        auto tmp266 = decltype(tmp265)(tmp265 & tmp171);
                        auto tmp267 = decltype(tmp266)(tmp266 & tmp211);
                        auto tmp268 = decltype(tmp267)(tmp267 & tmp231);
                        auto tmp269 = decltype(tmp268)(tmp268 & tmp261);
                        auto tmp271 = tmp270 == tmp1;
                        auto tmp272 = decltype(tmp269)(tmp269 & tmp271);
                        auto tmp273 = decltype(tmp263)(tmp263 | tmp272);
                        auto tmp274 = decltype(tmp121)(tmp121 & tmp141);
                        auto tmp275 = decltype(tmp274)(tmp274 & tmp171);
                        auto tmp276 = decltype(tmp275)(tmp275 & tmp181);
                        auto tmp277 = decltype(tmp276)(tmp276 & tmp221);
                        auto tmp278 = decltype(tmp277)(tmp277 & tmp241);
                        auto tmp279 = decltype(tmp278)(tmp278 & tmp271);
                        auto tmp281 = tmp280 == tmp1;
                        auto tmp282 = decltype(tmp279)(tmp279 & tmp281);
                        auto tmp283 = decltype(tmp273)(tmp273 | tmp282);
                        auto tmp284 = decltype(tmp131)(tmp131 & tmp151);
                        auto tmp285 = decltype(tmp284)(tmp284 & tmp181);
                        auto tmp286 = decltype(tmp285)(tmp285 & tmp191);
                        auto tmp287 = decltype(tmp286)(tmp286 & tmp231);
                        auto tmp288 = decltype(tmp287)(tmp287 & tmp251);
                        auto tmp289 = decltype(tmp288)(tmp288 & tmp281);
                        auto tmp291 = tmp290 == tmp1;
                        auto tmp292 = decltype(tmp289)(tmp289 & tmp291);
                        auto tmp293 = decltype(tmp283)(tmp283 | tmp292);
                        auto tmp294 = decltype(tmp141)(tmp141 & tmp161);
                        auto tmp295 = decltype(tmp294)(tmp294 & tmp191);
                        auto tmp296 = decltype(tmp295)(tmp295 & tmp201);
                        auto tmp297 = decltype(tmp296)(tmp296 & tmp241);
                        auto tmp298 = decltype(tmp297)(tmp297 & tmp261);
                        auto tmp299 = decltype(tmp298)(tmp298 & tmp291);
                        auto tmp301 = tmp300 == tmp1;
                        auto tmp302 = decltype(tmp299)(tmp299 & tmp301);
                        auto tmp303 = decltype(tmp293)(tmp293 | tmp302);
                        auto tmp304 = decltype(tmp151)(tmp151 & tmp171);
                        auto tmp305 = decltype(tmp304)(tmp304 & tmp201);
                        auto tmp306 = decltype(tmp305)(tmp305 & tmp211);
                        auto tmp307 = decltype(tmp306)(tmp306 & tmp251);
                        auto tmp308 = decltype(tmp307)(tmp307 & tmp271);
                        auto tmp309 = decltype(tmp308)(tmp308 & tmp301);
                        auto tmp311 = tmp310 == tmp1;
                        auto tmp312 = decltype(tmp309)(tmp309 & tmp311);
                        auto tmp313 = decltype(tmp303)(tmp303 | tmp312);
                        auto tmp314 = decltype(tmp161)(tmp161 & tmp181);
                        auto tmp315 = decltype(tmp314)(tmp314 & tmp211);
                        auto tmp316 = decltype(tmp315)(tmp315 & tmp221);
                        auto tmp317 = decltype(tmp316)(tmp316 & tmp261);
                        auto tmp318 = decltype(tmp317)(tmp317 & tmp281);
                        auto tmp319 = decltype(tmp318)(tmp318 & tmp311);
                        auto tmp321 = tmp320 == tmp1;
                        auto tmp322 = decltype(tmp319)(tmp319 & tmp321);
                        auto tmp323 = decltype(tmp313)(tmp313 | tmp322);
                        auto tmp324 = decltype(tmp171)(tmp171 & tmp191);
                        auto tmp325 = decltype(tmp324)(tmp324 & tmp221);
                        auto tmp326 = decltype(tmp325)(tmp325 & tmp231);
                        auto tmp327 = decltype(tmp326)(tmp326 & tmp271);
                        auto tmp328 = decltype(tmp327)(tmp327 & tmp291);
                        auto tmp329 = decltype(tmp328)(tmp328 & tmp321);
                        auto tmp331 = tmp330 == tmp1;
                        auto tmp332 = decltype(tmp329)(tmp329 & tmp331);
                        auto tmp333 = decltype(tmp323)(tmp323 | tmp332);
                        auto tmp334 = decltype(tmp181)(tmp181 & tmp201);
                        auto tmp335 = decltype(tmp334)(tmp334 & tmp231);
                        auto tmp336 = decltype(tmp335)(tmp335 & tmp241);
                        auto tmp337 = decltype(tmp336)(tmp336 & tmp281);
                        auto tmp338 = decltype(tmp337)(tmp337 & tmp301);
                        auto tmp339 = decltype(tmp338)(tmp338 & tmp331);
                        auto tmp341 = tmp340 == tmp1;
                        auto tmp342 = decltype(tmp339)(tmp339 & tmp341);
                        auto tmp343 = decltype(tmp333)(tmp333 | tmp342);
                        auto tmp344 = decltype(tmp191)(tmp191 & tmp211);
                        auto tmp345 = decltype(tmp344)(tmp344 & tmp241);
                        auto tmp346 = decltype(tmp345)(tmp345 & tmp251);
                        auto tmp347 = decltype(tmp346)(tmp346 & tmp291);
                        auto tmp348 = decltype(tmp347)(tmp347 & tmp311);
                        auto tmp349 = decltype(tmp348)(tmp348 & tmp341);
                        auto tmp351 = tmp350 == tmp1;
                        auto tmp352 = decltype(tmp349)(tmp349 & tmp351);
                        auto tmp353 = decltype(tmp343)(tmp343 | tmp352);
                        auto tmp354 = decltype(tmp201)(tmp201 & tmp221);
                        auto tmp355 = decltype(tmp354)(tmp354 & tmp251);
                        auto tmp356 = decltype(tmp355)(tmp355 & tmp261);
                        auto tmp357 = decltype(tmp356)(tmp356 & tmp301);
                        auto tmp358 = decltype(tmp357)(tmp357 & tmp321);
                        auto tmp359 = decltype(tmp358)(tmp358 & tmp351);
                        auto tmp361 = tmp360 == tmp1;
                        auto tmp362 = decltype(tmp359)(tmp359 & tmp361);
                        auto tmp363 = decltype(tmp353)(tmp353 | tmp362);
                        auto tmp364 = decltype(tmp211)(tmp211 & tmp231);
                        auto tmp365 = decltype(tmp364)(tmp364 & tmp261);
                        auto tmp366 = decltype(tmp365)(tmp365 & tmp271);
                        auto tmp367 = decltype(tmp366)(tmp366 & tmp311);
                        auto tmp368 = decltype(tmp367)(tmp367 & tmp331);
                        auto tmp369 = decltype(tmp368)(tmp368 & tmp361);
                        auto tmp371 = tmp370 == tmp1;
                        auto tmp372 = decltype(tmp369)(tmp369 & tmp371);
                        auto tmp373 = decltype(tmp363)(tmp363 | tmp372);
                        auto tmp374 = decltype(tmp221)(tmp221 & tmp241);
                        auto tmp375 = decltype(tmp374)(tmp374 & tmp271);
                        auto tmp376 = decltype(tmp375)(tmp375 & tmp281);
                        auto tmp377 = decltype(tmp376)(tmp376 & tmp321);
                        auto tmp378 = decltype(tmp377)(tmp377 & tmp341);
                        auto tmp379 = decltype(tmp378)(tmp378 & tmp371);
                        auto tmp381 = tmp380 == tmp1;
                        auto tmp382 = decltype(tmp379)(tmp379 & tmp381);
                        auto tmp383 = decltype(tmp373)(tmp373 | tmp382);
                        auto tmp384 = decltype(tmp231)(tmp231 & tmp251);
                        auto tmp385 = decltype(tmp384)(tmp384 & tmp281);
                        auto tmp386 = decltype(tmp385)(tmp385 & tmp291);
                        auto tmp387 = decltype(tmp386)(tmp386 & tmp331);
                        auto tmp388 = decltype(tmp387)(tmp387 & tmp351);
                        auto tmp389 = decltype(tmp388)(tmp388 & tmp381);
                        auto tmp391 = tmp390 == tmp1;
                        auto tmp392 = decltype(tmp389)(tmp389 & tmp391);
                        auto tmp393 = decltype(tmp383)(tmp383 | tmp392);
                        auto tmp394 = decltype(tmp241)(tmp241 & tmp261);
                        auto tmp395 = decltype(tmp394)(tmp394 & tmp291);
                        auto tmp396 = decltype(tmp395)(tmp395 & tmp301);
                        auto tmp397 = decltype(tmp396)(tmp396 & tmp341);
                        auto tmp398 = decltype(tmp397)(tmp397 & tmp361);
                        auto tmp399 = decltype(tmp398)(tmp398 & tmp391);
                        auto tmp401 = tmp400 == tmp1;
                        auto tmp402 = decltype(tmp399)(tmp399 & tmp401);
                        auto tmp403 = decltype(tmp393)(tmp393 | tmp402);
                        auto tmp404 = decltype(tmp251)(tmp251 & tmp271);
                        auto tmp405 = decltype(tmp404)(tmp404 & tmp301);
                        auto tmp406 = decltype(tmp405)(tmp405 & tmp311);
                        auto tmp407 = decltype(tmp406)(tmp406 & tmp351);
                        auto tmp408 = decltype(tmp407)(tmp407 & tmp371);
                        auto tmp409 = decltype(tmp408)(tmp408 & tmp401);
                        auto tmp411 = tmp410 == tmp1;
                        auto tmp412 = decltype(tmp409)(tmp409 & tmp411);
                        auto tmp413 = decltype(tmp403)(tmp403 | tmp412);
                        auto tmp414 = decltype(tmp261)(tmp261 & tmp281);
                        auto tmp415 = decltype(tmp414)(tmp414 & tmp311);
                        auto tmp416 = decltype(tmp415)(tmp415 & tmp321);
                        auto tmp417 = decltype(tmp416)(tmp416 & tmp361);
                        auto tmp418 = decltype(tmp417)(tmp417 & tmp381);
                        auto tmp419 = decltype(tmp418)(tmp418 & tmp411);
                        auto tmp421 = tmp420 == tmp1;
                        auto tmp422 = decltype(tmp419)(tmp419 & tmp421);
                        auto tmp423 = decltype(tmp413)(tmp413 | tmp422);
                        auto tmp424 = decltype(tmp271)(tmp271 & tmp291);
                        auto tmp425 = decltype(tmp424)(tmp424 & tmp321);
                        auto tmp426 = decltype(tmp425)(tmp425 & tmp331);
                        auto tmp427 = decltype(tmp426)(tmp426 & tmp371);
                        auto tmp428 = decltype(tmp427)(tmp427 & tmp391);
                        auto tmp429 = decltype(tmp428)(tmp428 & tmp421);
                        auto tmp431 = tmp430 == tmp1;
                        auto tmp432 = decltype(tmp429)(tmp429 & tmp431);
                        auto tmp433 = decltype(tmp423)(tmp423 | tmp432);
                        auto tmp434 = decltype(tmp281)(tmp281 & tmp301);
                        auto tmp435 = decltype(tmp434)(tmp434 & tmp331);
                        auto tmp436 = decltype(tmp435)(tmp435 & tmp341);
                        auto tmp437 = decltype(tmp436)(tmp436 & tmp381);
                        auto tmp438 = decltype(tmp437)(tmp437 & tmp401);
                        auto tmp439 = decltype(tmp438)(tmp438 & tmp431);
                        auto tmp441 = tmp440 == tmp1;
                        auto tmp442 = decltype(tmp439)(tmp439 & tmp441);
                        auto tmp443 = decltype(tmp433)(tmp433 | tmp442);
                        auto tmp444 = decltype(tmp291)(tmp291 & tmp311);
                        auto tmp445 = decltype(tmp444)(tmp444 & tmp341);
                        auto tmp446 = decltype(tmp445)(tmp445 & tmp351);
                        auto tmp447 = decltype(tmp446)(tmp446 & tmp391);
                        auto tmp448 = decltype(tmp447)(tmp447 & tmp411);
                        auto tmp449 = decltype(tmp448)(tmp448 & tmp441);
                        auto tmp451 = tmp450 == tmp1;
                        auto tmp452 = decltype(tmp449)(tmp449 & tmp451);
                        auto tmp453 = decltype(tmp443)(tmp443 | tmp452);
                        auto tmp454 = decltype(tmp301)(tmp301 & tmp321);
                        auto tmp455 = decltype(tmp454)(tmp454 & tmp351);
                        auto tmp456 = decltype(tmp455)(tmp455 & tmp361);
                        auto tmp457 = decltype(tmp456)(tmp456 & tmp401);
                        auto tmp458 = decltype(tmp457)(tmp457 & tmp421);
                        auto tmp459 = decltype(tmp458)(tmp458 & tmp451);
                        auto tmp461 = tmp460 == tmp1;
                        auto tmp462 = decltype(tmp459)(tmp459 & tmp461);
                        auto tmp463 = decltype(tmp453)(tmp453 | tmp462);
                        auto tmp464 = decltype(tmp311)(tmp311 & tmp331);
                        auto tmp465 = decltype(tmp464)(tmp464 & tmp361);
                        auto tmp466 = decltype(tmp465)(tmp465 & tmp371);
                        auto tmp467 = decltype(tmp466)(tmp466 & tmp411);
                        auto tmp468 = decltype(tmp467)(tmp467 & tmp431);
                        auto tmp469 = decltype(tmp468)(tmp468 & tmp461);
                        auto tmp471 = tmp470 == tmp1;
                        auto tmp472 = decltype(tmp469)(tmp469 & tmp471);
                        auto tmp473 = decltype(tmp463)(tmp463 | tmp472);
                        auto tmp474 = decltype(tmp321)(tmp321 & tmp341);
                        auto tmp475 = decltype(tmp474)(tmp474 & tmp371);
                        auto tmp476 = decltype(tmp475)(tmp475 & tmp381);
                        auto tmp477 = decltype(tmp476)(tmp476 & tmp421);
                        auto tmp478 = decltype(tmp477)(tmp477 & tmp441);
                        auto tmp479 = decltype(tmp478)(tmp478 & tmp471);
                        auto tmp481 = tmp480 == tmp1;
                        auto tmp482 = decltype(tmp479)(tmp479 & tmp481);
                        auto tmp483 = decltype(tmp473)(tmp473 | tmp482);
                        auto tmp484 = decltype(tmp331)(tmp331 & tmp351);
                        auto tmp485 = decltype(tmp484)(tmp484 & tmp381);
                        auto tmp486 = decltype(tmp485)(tmp485 & tmp391);
                        auto tmp487 = decltype(tmp486)(tmp486 & tmp431);
                        auto tmp488 = decltype(tmp487)(tmp487 & tmp451);
                        auto tmp489 = decltype(tmp488)(tmp488 & tmp481);
                        auto tmp491 = tmp490 == tmp1;
                        auto tmp492 = decltype(tmp489)(tmp489 & tmp491);
                        auto tmp493 = decltype(tmp483)(tmp483 | tmp492);
                        auto tmp494 = decltype(tmp341)(tmp341 & tmp361);
                        auto tmp495 = decltype(tmp494)(tmp494 & tmp391);
                        auto tmp496 = decltype(tmp495)(tmp495 & tmp401);
                        auto tmp497 = decltype(tmp496)(tmp496 & tmp441);
                        auto tmp498 = decltype(tmp497)(tmp497 & tmp461);
                        auto tmp499 = decltype(tmp498)(tmp498 & tmp491);
                        auto tmp501 = tmp500 == tmp1;
                        auto tmp502 = decltype(tmp499)(tmp499 & tmp501);
                        auto tmp503 = decltype(tmp493)(tmp493 | tmp502);
                        auto tmp504 = decltype(tmp351)(tmp351 & tmp371);
                        auto tmp505 = decltype(tmp504)(tmp504 & tmp401);
                        auto tmp506 = decltype(tmp505)(tmp505 & tmp411);
                        auto tmp507 = decltype(tmp506)(tmp506 & tmp451);
                        auto tmp508 = decltype(tmp507)(tmp507 & tmp471);
                        auto tmp509 = decltype(tmp508)(tmp508 & tmp501);
                        auto tmp511 = tmp510 == tmp1;
                        auto tmp512 = decltype(tmp509)(tmp509 & tmp511);
                        auto tmp513 = decltype(tmp503)(tmp503 | tmp512);
                        auto tmp514 = decltype(tmp361)(tmp361 & tmp381);
                        auto tmp515 = decltype(tmp514)(tmp514 & tmp411);
                        auto tmp516 = decltype(tmp515)(tmp515 & tmp421);
                        auto tmp517 = decltype(tmp516)(tmp516 & tmp461);
                        auto tmp518 = decltype(tmp517)(tmp517 & tmp481);
                        auto tmp519 = decltype(tmp518)(tmp518 & tmp511);
                        auto tmp521 = tmp520 == tmp1;
                        auto tmp522 = decltype(tmp519)(tmp519 & tmp521);
                        auto tmp523 = decltype(tmp513)(tmp513 | tmp522);
                        auto tmp524 = decltype(tmp371)(tmp371 & tmp391);
                        auto tmp525 = decltype(tmp524)(tmp524 & tmp421);
                        auto tmp526 = decltype(tmp525)(tmp525 & tmp431);
                        auto tmp527 = decltype(tmp526)(tmp526 & tmp471);
                        auto tmp528 = decltype(tmp527)(tmp527 & tmp491);
                        auto tmp529 = decltype(tmp528)(tmp528 & tmp521);
                        auto tmp531 = tmp530 == tmp1;
                        auto tmp532 = decltype(tmp529)(tmp529 & tmp531);
                        auto tmp533 = decltype(tmp523)(tmp523 | tmp532);
                        auto tmp534 = decltype(tmp381)(tmp381 & tmp401);
                        auto tmp535 = decltype(tmp534)(tmp534 & tmp431);
                        auto tmp536 = decltype(tmp535)(tmp535 & tmp441);
                        auto tmp537 = decltype(tmp536)(tmp536 & tmp481);
                        auto tmp538 = decltype(tmp537)(tmp537 & tmp501);
                        auto tmp539 = decltype(tmp538)(tmp538 & tmp531);
                        auto tmp541 = tmp540 == tmp1;
                        auto tmp542 = decltype(tmp539)(tmp539 & tmp541);
                        auto tmp543 = decltype(tmp533)(tmp533 | tmp542);
                        auto tmp544 = decltype(tmp391)(tmp391 & tmp411);
                        auto tmp545 = decltype(tmp544)(tmp544 & tmp441);
                        auto tmp546 = decltype(tmp545)(tmp545 & tmp451);
                        auto tmp547 = decltype(tmp546)(tmp546 & tmp491);
                        auto tmp548 = decltype(tmp547)(tmp547 & tmp511);
                        auto tmp549 = decltype(tmp548)(tmp548 & tmp541);
                        auto tmp551 = tmp550 == tmp1;
                        auto tmp552 = decltype(tmp549)(tmp549 & tmp551);
                        auto tmp553 = decltype(tmp543)(tmp543 | tmp552);
                        auto tmp554 = decltype(tmp401)(tmp401 & tmp421);
                        auto tmp555 = decltype(tmp554)(tmp554 & tmp451);
                        auto tmp556 = decltype(tmp555)(tmp555 & tmp461);
                        auto tmp557 = decltype(tmp556)(tmp556 & tmp501);
                        auto tmp558 = decltype(tmp557)(tmp557 & tmp521);
                        auto tmp559 = decltype(tmp558)(tmp558 & tmp551);
                        auto tmp561 = tmp560 == tmp1;
                        auto tmp562 = decltype(tmp559)(tmp559 & tmp561);
                        auto tmp563 = decltype(tmp553)(tmp553 | tmp562);
                        auto tmp564 = decltype(tmp411)(tmp411 & tmp431);
                        auto tmp565 = decltype(tmp564)(tmp564 & tmp461);
                        auto tmp566 = decltype(tmp565)(tmp565 & tmp471);
                        auto tmp567 = decltype(tmp566)(tmp566 & tmp511);
                        auto tmp568 = decltype(tmp567)(tmp567 & tmp531);
                        auto tmp569 = decltype(tmp568)(tmp568 & tmp561);
                        auto tmp571 = tmp570 == tmp1;
                        auto tmp572 = decltype(tmp569)(tmp569 & tmp571);
                        auto tmp573 = decltype(tmp563)(tmp563 | tmp572);
                        auto tmp574 = decltype(tmp421)(tmp421 & tmp441);
                        auto tmp575 = decltype(tmp574)(tmp574 & tmp471);
                        auto tmp576 = decltype(tmp575)(tmp575 & tmp481);
                        auto tmp577 = decltype(tmp576)(tmp576 & tmp521);
                        auto tmp578 = decltype(tmp577)(tmp577 & tmp541);
                        auto tmp579 = decltype(tmp578)(tmp578 & tmp571);
                        auto tmp581 = tmp580 == tmp1;
                        auto tmp582 = decltype(tmp579)(tmp579 & tmp581);
                        auto tmp583 = decltype(tmp573)(tmp573 | tmp582);
                        auto tmp584 = decltype(tmp431)(tmp431 & tmp451);
                        auto tmp585 = decltype(tmp584)(tmp584 & tmp481);
                        auto tmp586 = decltype(tmp585)(tmp585 & tmp491);
                        auto tmp587 = decltype(tmp586)(tmp586 & tmp531);
                        auto tmp588 = decltype(tmp587)(tmp587 & tmp551);
                        auto tmp589 = decltype(tmp588)(tmp588 & tmp581);
                        auto tmp591 = tmp590 == tmp1;
                        auto tmp592 = decltype(tmp589)(tmp589 & tmp591);
                        auto tmp593 = decltype(tmp583)(tmp583 | tmp592);
                        auto tmp594 = decltype(tmp441)(tmp441 & tmp461);
                        auto tmp595 = decltype(tmp594)(tmp594 & tmp491);
                        auto tmp596 = decltype(tmp595)(tmp595 & tmp501);
                        auto tmp597 = decltype(tmp596)(tmp596 & tmp541);
                        auto tmp598 = decltype(tmp597)(tmp597 & tmp561);
                        auto tmp599 = decltype(tmp598)(tmp598 & tmp591);
                        auto tmp601 = tmp600 == tmp1;
                        auto tmp602 = decltype(tmp599)(tmp599 & tmp601);
                        auto tmp603 = decltype(tmp593)(tmp593 | tmp602);
                        auto tmp604 = decltype(tmp451)(tmp451 & tmp471);
                        auto tmp605 = decltype(tmp604)(tmp604 & tmp501);
                        auto tmp606 = decltype(tmp605)(tmp605 & tmp511);
                        auto tmp607 = decltype(tmp606)(tmp606 & tmp551);
                        auto tmp608 = decltype(tmp607)(tmp607 & tmp571);
                        auto tmp609 = decltype(tmp608)(tmp608 & tmp601);
                        auto tmp611 = tmp610 == tmp1;
                        auto tmp612 = decltype(tmp609)(tmp609 & tmp611);
                        auto tmp613 = decltype(tmp603)(tmp603 | tmp612);
                        auto tmp614 = decltype(tmp461)(tmp461 & tmp481);
                        auto tmp615 = decltype(tmp614)(tmp614 & tmp511);
                        auto tmp616 = decltype(tmp615)(tmp615 & tmp521);
                        auto tmp617 = decltype(tmp616)(tmp616 & tmp561);
                        auto tmp618 = decltype(tmp617)(tmp617 & tmp581);
                        auto tmp619 = decltype(tmp618)(tmp618 & tmp611);
                        auto tmp621 = tmp620 == tmp1;
                        auto tmp622 = decltype(tmp619)(tmp619 & tmp621);
                        auto tmp623 = decltype(tmp613)(tmp613 | tmp622);
                        auto tmp624 = decltype(tmp471)(tmp471 & tmp491);
                        auto tmp625 = decltype(tmp624)(tmp624 & tmp521);
                        auto tmp626 = decltype(tmp625)(tmp625 & tmp531);
                        auto tmp627 = decltype(tmp626)(tmp626 & tmp571);
                        auto tmp628 = decltype(tmp627)(tmp627 & tmp591);
                        auto tmp629 = decltype(tmp628)(tmp628 & tmp621);
                        auto tmp631 = tmp630 == tmp1;
                        auto tmp632 = decltype(tmp629)(tmp629 & tmp631);
                        auto tmp633 = decltype(tmp623)(tmp623 | tmp632);
                        auto tmp634 = decltype(tmp481)(tmp481 & tmp501);
                        auto tmp635 = decltype(tmp634)(tmp634 & tmp531);
                        auto tmp636 = decltype(tmp635)(tmp635 & tmp541);
                        auto tmp637 = decltype(tmp636)(tmp636 & tmp581);
                        auto tmp638 = decltype(tmp637)(tmp637 & tmp601);
                        auto tmp639 = decltype(tmp638)(tmp638 & tmp631);
                        auto tmp641 = tmp640 == tmp1;
                        auto tmp642 = decltype(tmp639)(tmp639 & tmp641);
                        auto tmp643 = decltype(tmp633)(tmp633 | tmp642);
                        auto tmp644 = decltype(tmp491)(tmp491 & tmp511);
                        auto tmp645 = decltype(tmp644)(tmp644 & tmp541);
                        auto tmp646 = decltype(tmp645)(tmp645 & tmp551);
                        auto tmp647 = decltype(tmp646)(tmp646 & tmp591);
                        auto tmp648 = decltype(tmp647)(tmp647 & tmp611);
                        auto tmp649 = decltype(tmp648)(tmp648 & tmp641);
                        auto tmp651 = tmp650 == tmp1;
                        auto tmp652 = decltype(tmp649)(tmp649 & tmp651);
                        auto tmp653 = decltype(tmp643)(tmp643 | tmp652);
                        auto tmp654 = decltype(tmp501)(tmp501 & tmp521);
                        auto tmp655 = decltype(tmp654)(tmp654 & tmp551);
                        auto tmp656 = decltype(tmp655)(tmp655 & tmp561);
                        auto tmp657 = decltype(tmp656)(tmp656 & tmp601);
                        auto tmp658 = decltype(tmp657)(tmp657 & tmp621);
                        auto tmp659 = decltype(tmp658)(tmp658 & tmp651);
                        auto tmp661 = tmp660 == tmp1;
                        auto tmp662 = decltype(tmp659)(tmp659 & tmp661);
                        auto tmp663 = decltype(tmp653)(tmp653 | tmp662);
                        auto tmp664 = decltype(tmp511)(tmp511 & tmp531);
                        auto tmp665 = decltype(tmp664)(tmp664 & tmp561);
                        auto tmp666 = decltype(tmp665)(tmp665 & tmp571);
                        auto tmp667 = decltype(tmp666)(tmp666 & tmp611);
                        auto tmp668 = decltype(tmp667)(tmp667 & tmp631);
                        auto tmp669 = decltype(tmp668)(tmp668 & tmp661);
                        auto tmp671 = tmp670 == tmp1;
                        auto tmp672 = decltype(tmp669)(tmp669 & tmp671);
                        auto tmp673 = decltype(tmp663)(tmp663 | tmp672);
                        auto tmp674 = decltype(tmp521)(tmp521 & tmp541);
                        auto tmp675 = decltype(tmp674)(tmp674 & tmp571);
                        auto tmp676 = decltype(tmp675)(tmp675 & tmp581);
                        auto tmp677 = decltype(tmp676)(tmp676 & tmp621);
                        auto tmp678 = decltype(tmp677)(tmp677 & tmp641);
                        auto tmp679 = decltype(tmp678)(tmp678 & tmp671);
                        auto tmp681 = tmp680 == tmp1;
                        auto tmp682 = decltype(tmp679)(tmp679 & tmp681);
                        auto tmp683 = decltype(tmp673)(tmp673 | tmp682);
                        auto tmp684 = decltype(tmp531)(tmp531 & tmp551);
                        auto tmp685 = decltype(tmp684)(tmp684 & tmp581);
                        auto tmp686 = decltype(tmp685)(tmp685 & tmp591);
                        auto tmp687 = decltype(tmp686)(tmp686 & tmp631);
                        auto tmp688 = decltype(tmp687)(tmp687 & tmp651);
                        auto tmp689 = decltype(tmp688)(tmp688 & tmp681);
                        auto tmp691 = tmp690 == tmp1;
                        auto tmp692 = decltype(tmp689)(tmp689 & tmp691);
                        auto tmp693 = decltype(tmp683)(tmp683 | tmp692);
                        auto tmp694 = decltype(tmp541)(tmp541 & tmp561);
                        auto tmp695 = decltype(tmp694)(tmp694 & tmp591);
                        auto tmp696 = decltype(tmp695)(tmp695 & tmp601);
                        auto tmp697 = decltype(tmp696)(tmp696 & tmp641);
                        auto tmp698 = decltype(tmp697)(tmp697 & tmp661);
                        auto tmp699 = decltype(tmp698)(tmp698 & tmp691);
                        auto tmp701 = tmp700 == tmp1;
                        auto tmp702 = decltype(tmp699)(tmp699 & tmp701);
                        auto tmp703 = decltype(tmp693)(tmp693 | tmp702);
                        auto tmp704 = decltype(tmp551)(tmp551 & tmp571);
                        auto tmp705 = decltype(tmp704)(tmp704 & tmp601);
                        auto tmp706 = decltype(tmp705)(tmp705 & tmp611);
                        auto tmp707 = decltype(tmp706)(tmp706 & tmp651);
                        auto tmp708 = decltype(tmp707)(tmp707 & tmp671);
                        auto tmp709 = decltype(tmp708)(tmp708 & tmp701);
                        auto tmp711 = tmp710 == tmp1;
                        auto tmp712 = decltype(tmp709)(tmp709 & tmp711);
                        auto tmp713 = decltype(tmp703)(tmp703 | tmp712);
                        auto tmp714 = decltype(tmp561)(tmp561 & tmp581);
                        auto tmp715 = decltype(tmp714)(tmp714 & tmp611);
                        auto tmp716 = decltype(tmp715)(tmp715 & tmp621);
                        auto tmp717 = decltype(tmp716)(tmp716 & tmp661);
                        auto tmp718 = decltype(tmp717)(tmp717 & tmp681);
                        auto tmp719 = decltype(tmp718)(tmp718 & tmp711);
                        auto tmp721 = tmp720 == tmp1;
                        auto tmp722 = decltype(tmp719)(tmp719 & tmp721);
                        auto tmp723 = decltype(tmp713)(tmp713 | tmp722);
                        auto tmp724 = decltype(tmp571)(tmp571 & tmp591);
                        auto tmp725 = decltype(tmp724)(tmp724 & tmp621);
                        auto tmp726 = decltype(tmp725)(tmp725 & tmp631);
                        auto tmp727 = decltype(tmp726)(tmp726 & tmp671);
                        auto tmp728 = decltype(tmp727)(tmp727 & tmp691);
                        auto tmp729 = decltype(tmp728)(tmp728 & tmp721);
                        auto tmp731 = tmp730 == tmp1;
                        auto tmp732 = decltype(tmp729)(tmp729 & tmp731);
                        auto tmp733 = decltype(tmp723)(tmp723 | tmp732);
                        auto tmp734 = decltype(tmp581)(tmp581 & tmp601);
                        auto tmp735 = decltype(tmp734)(tmp734 & tmp631);
                        auto tmp736 = decltype(tmp735)(tmp735 & tmp641);
                        auto tmp737 = decltype(tmp736)(tmp736 & tmp681);
                        auto tmp738 = decltype(tmp737)(tmp737 & tmp701);
                        auto tmp739 = decltype(tmp738)(tmp738 & tmp731);
                        auto tmp741 = tmp740 == tmp1;
                        auto tmp742 = decltype(tmp739)(tmp739 & tmp741);
                        auto tmp743 = decltype(tmp733)(tmp733 | tmp742);
                        auto tmp744 = decltype(tmp591)(tmp591 & tmp611);
                        auto tmp745 = decltype(tmp744)(tmp744 & tmp641);
                        auto tmp746 = decltype(tmp745)(tmp745 & tmp651);
                        auto tmp747 = decltype(tmp746)(tmp746 & tmp691);
                        auto tmp748 = decltype(tmp747)(tmp747 & tmp711);
                        auto tmp749 = decltype(tmp748)(tmp748 & tmp741);
                        auto tmp751 = tmp750 == tmp1;
                        auto tmp752 = decltype(tmp749)(tmp749 & tmp751);
                        auto tmp753 = decltype(tmp743)(tmp743 | tmp752);
                        auto tmp754 = decltype(tmp601)(tmp601 & tmp621);
                        auto tmp755 = decltype(tmp754)(tmp754 & tmp651);
                        auto tmp756 = decltype(tmp755)(tmp755 & tmp661);
                        auto tmp757 = decltype(tmp756)(tmp756 & tmp701);
                        auto tmp758 = decltype(tmp757)(tmp757 & tmp721);
                        auto tmp759 = decltype(tmp758)(tmp758 & tmp751);
                        auto tmp761 = tmp760 == tmp1;
                        auto tmp762 = decltype(tmp759)(tmp759 & tmp761);
                        auto tmp763 = decltype(tmp753)(tmp753 | tmp762);
                        auto tmp764 = decltype(tmp611)(tmp611 & tmp631);
                        auto tmp765 = decltype(tmp764)(tmp764 & tmp661);
                        auto tmp766 = decltype(tmp765)(tmp765 & tmp671);
                        auto tmp767 = decltype(tmp766)(tmp766 & tmp711);
                        auto tmp768 = decltype(tmp767)(tmp767 & tmp731);
                        auto tmp769 = decltype(tmp768)(tmp768 & tmp761);
                        auto tmp771 = tmp770 == tmp1;
                        auto tmp772 = decltype(tmp769)(tmp769 & tmp771);
                        auto tmp773 = decltype(tmp763)(tmp763 | tmp772);
                        auto tmp774 = decltype(tmp621)(tmp621 & tmp641);
                        auto tmp775 = decltype(tmp774)(tmp774 & tmp671);
                        auto tmp776 = decltype(tmp775)(tmp775 & tmp681);
                        auto tmp777 = decltype(tmp776)(tmp776 & tmp721);
                        auto tmp778 = decltype(tmp777)(tmp777 & tmp741);
                        auto tmp779 = decltype(tmp778)(tmp778 & tmp771);
                        auto tmp781 = tmp780 == tmp1;
                        auto tmp782 = decltype(tmp779)(tmp779 & tmp781);
                        auto tmp783 = decltype(tmp773)(tmp773 | tmp782);
                        auto tmp784 = decltype(tmp631)(tmp631 & tmp651);
                        auto tmp785 = decltype(tmp784)(tmp784 & tmp681);
                        auto tmp786 = decltype(tmp785)(tmp785 & tmp691);
                        auto tmp787 = decltype(tmp786)(tmp786 & tmp731);
                        auto tmp788 = decltype(tmp787)(tmp787 & tmp751);
                        auto tmp789 = decltype(tmp788)(tmp788 & tmp781);
                        auto tmp791 = tmp790 == tmp1;
                        auto tmp792 = decltype(tmp789)(tmp789 & tmp791);
                        auto tmp793 = decltype(tmp783)(tmp783 | tmp792);
                        auto tmp794 = decltype(tmp641)(tmp641 & tmp661);
                        auto tmp795 = decltype(tmp794)(tmp794 & tmp691);
                        auto tmp796 = decltype(tmp795)(tmp795 & tmp701);
                        auto tmp797 = decltype(tmp796)(tmp796 & tmp741);
                        auto tmp798 = decltype(tmp797)(tmp797 & tmp761);
                        auto tmp799 = decltype(tmp798)(tmp798 & tmp791);
                        auto tmp801 = tmp800 == tmp1;
                        auto tmp802 = decltype(tmp799)(tmp799 & tmp801);
                        auto tmp803 = decltype(tmp793)(tmp793 | tmp802);
                        auto tmp804 = decltype(tmp651)(tmp651 & tmp671);
                        auto tmp805 = decltype(tmp804)(tmp804 & tmp701);
                        auto tmp806 = decltype(tmp805)(tmp805 & tmp711);
                        auto tmp807 = decltype(tmp806)(tmp806 & tmp751);
                        auto tmp808 = decltype(tmp807)(tmp807 & tmp771);
                        auto tmp809 = decltype(tmp808)(tmp808 & tmp801);
                        auto tmp811 = tmp810 == tmp1;
                        auto tmp812 = decltype(tmp809)(tmp809 & tmp811);
                        auto tmp813 = decltype(tmp803)(tmp803 | tmp812);
                        auto tmp814 = decltype(tmp661)(tmp661 & tmp681);
                        auto tmp815 = decltype(tmp814)(tmp814 & tmp711);
                        auto tmp816 = decltype(tmp815)(tmp815 & tmp721);
                        auto tmp817 = decltype(tmp816)(tmp816 & tmp761);
                        auto tmp818 = decltype(tmp817)(tmp817 & tmp781);
                        auto tmp819 = decltype(tmp818)(tmp818 & tmp811);
                        auto tmp821 = tmp820 == tmp1;
                        auto tmp822 = decltype(tmp819)(tmp819 & tmp821);
                        auto tmp823 = decltype(tmp813)(tmp813 | tmp822);
                        auto tmp824 = decltype(tmp671)(tmp671 & tmp691);
                        auto tmp825 = decltype(tmp824)(tmp824 & tmp721);
                        auto tmp826 = decltype(tmp825)(tmp825 & tmp731);
                        auto tmp827 = decltype(tmp826)(tmp826 & tmp771);
                        auto tmp828 = decltype(tmp827)(tmp827 & tmp791);
                        auto tmp829 = decltype(tmp828)(tmp828 & tmp821);
                        auto tmp831 = tmp830 == tmp1;
                        auto tmp832 = decltype(tmp829)(tmp829 & tmp831);
                        auto tmp833 = decltype(tmp823)(tmp823 | tmp832);
                        auto tmp834 = decltype(tmp681)(tmp681 & tmp701);
                        auto tmp835 = decltype(tmp834)(tmp834 & tmp731);
                        auto tmp836 = decltype(tmp835)(tmp835 & tmp741);
                        auto tmp837 = decltype(tmp836)(tmp836 & tmp781);
                        auto tmp838 = decltype(tmp837)(tmp837 & tmp801);
                        auto tmp839 = decltype(tmp838)(tmp838 & tmp831);
                        auto tmp841 = tmp840 == tmp1;
                        auto tmp842 = decltype(tmp839)(tmp839 & tmp841);
                        auto tmp843 = decltype(tmp833)(tmp833 | tmp842);
                        auto tmp844 = decltype(tmp691)(tmp691 & tmp711);
                        auto tmp845 = decltype(tmp844)(tmp844 & tmp741);
                        auto tmp846 = decltype(tmp845)(tmp845 & tmp751);
                        auto tmp847 = decltype(tmp846)(tmp846 & tmp791);
                        auto tmp848 = decltype(tmp847)(tmp847 & tmp811);
                        auto tmp849 = decltype(tmp848)(tmp848 & tmp841);
                        auto tmp851 = tmp850 == tmp1;
                        auto tmp852 = decltype(tmp849)(tmp849 & tmp851);
                        auto tmp853 = decltype(tmp843)(tmp843 | tmp852);
                        auto tmp854 = decltype(tmp701)(tmp701 & tmp721);
                        auto tmp855 = decltype(tmp854)(tmp854 & tmp751);
                        auto tmp856 = decltype(tmp855)(tmp855 & tmp761);
                        auto tmp857 = decltype(tmp856)(tmp856 & tmp801);
                        auto tmp858 = decltype(tmp857)(tmp857 & tmp821);
                        auto tmp859 = decltype(tmp858)(tmp858 & tmp851);
                        auto tmp861 = tmp860 == tmp1;
                        auto tmp862 = decltype(tmp859)(tmp859 & tmp861);
                        auto tmp863 = decltype(tmp853)(tmp853 | tmp862);
                        auto tmp864 = decltype(tmp711)(tmp711 & tmp731);
                        auto tmp865 = decltype(tmp864)(tmp864 & tmp761);
                        auto tmp866 = decltype(tmp865)(tmp865 & tmp771);
                        auto tmp867 = decltype(tmp866)(tmp866 & tmp811);
                        auto tmp868 = decltype(tmp867)(tmp867 & tmp831);
                        auto tmp869 = decltype(tmp868)(tmp868 & tmp861);
                        auto tmp871 = tmp870 == tmp1;
                        auto tmp872 = decltype(tmp869)(tmp869 & tmp871);
                        auto tmp873 = decltype(tmp863)(tmp863 | tmp872);
                        auto tmp874 = decltype(tmp721)(tmp721 & tmp741);
                        auto tmp875 = decltype(tmp874)(tmp874 & tmp771);
                        auto tmp876 = decltype(tmp875)(tmp875 & tmp781);
                        auto tmp877 = decltype(tmp876)(tmp876 & tmp821);
                        auto tmp878 = decltype(tmp877)(tmp877 & tmp841);
                        auto tmp879 = decltype(tmp878)(tmp878 & tmp871);
                        auto tmp881 = tmp880 == tmp1;
                        auto tmp882 = decltype(tmp879)(tmp879 & tmp881);
                        auto tmp883 = decltype(tmp873)(tmp873 | tmp882);
                        auto tmp884 = decltype(tmp731)(tmp731 & tmp751);
                        auto tmp885 = decltype(tmp884)(tmp884 & tmp781);
                        auto tmp886 = decltype(tmp885)(tmp885 & tmp791);
                        auto tmp887 = decltype(tmp886)(tmp886 & tmp831);
                        auto tmp888 = decltype(tmp887)(tmp887 & tmp851);
                        auto tmp889 = decltype(tmp888)(tmp888 & tmp881);
                        auto tmp891 = tmp890 == tmp1;
                        auto tmp892 = decltype(tmp889)(tmp889 & tmp891);
                        auto tmp893 = decltype(tmp883)(tmp883 | tmp892);
                        auto tmp894 = decltype(tmp741)(tmp741 & tmp761);
                        auto tmp895 = decltype(tmp894)(tmp894 & tmp791);
                        auto tmp896 = decltype(tmp895)(tmp895 & tmp801);
                        auto tmp897 = decltype(tmp896)(tmp896 & tmp841);
                        auto tmp898 = decltype(tmp897)(tmp897 & tmp861);
                        auto tmp899 = decltype(tmp898)(tmp898 & tmp891);
                        auto tmp901 = tmp900 == tmp1;
                        auto tmp902 = decltype(tmp899)(tmp899 & tmp901);
                        auto tmp903 = decltype(tmp893)(tmp893 | tmp902);
                        auto tmp904 = decltype(tmp751)(tmp751 & tmp771);
                        auto tmp905 = decltype(tmp904)(tmp904 & tmp801);
                        auto tmp906 = decltype(tmp905)(tmp905 & tmp811);
                        auto tmp907 = decltype(tmp906)(tmp906 & tmp851);
                        auto tmp908 = decltype(tmp907)(tmp907 & tmp871);
                        auto tmp909 = decltype(tmp908)(tmp908 & tmp901);
                        auto tmp911 = tmp910 == tmp1;
                        auto tmp912 = decltype(tmp909)(tmp909 & tmp911);
                        auto tmp913 = decltype(tmp903)(tmp903 | tmp912);
                        auto tmp914 = decltype(tmp761)(tmp761 & tmp781);
                        auto tmp915 = decltype(tmp914)(tmp914 & tmp811);
                        auto tmp916 = decltype(tmp915)(tmp915 & tmp821);
                        auto tmp917 = decltype(tmp916)(tmp916 & tmp861);
                        auto tmp918 = decltype(tmp917)(tmp917 & tmp881);
                        auto tmp919 = decltype(tmp918)(tmp918 & tmp911);
                        auto tmp921 = tmp920 == tmp1;
                        auto tmp922 = decltype(tmp919)(tmp919 & tmp921);
                        auto tmp923 = decltype(tmp913)(tmp913 | tmp922);
                        auto tmp924 = decltype(tmp771)(tmp771 & tmp791);
                        auto tmp925 = decltype(tmp924)(tmp924 & tmp821);
                        auto tmp926 = decltype(tmp925)(tmp925 & tmp831);
                        auto tmp927 = decltype(tmp926)(tmp926 & tmp871);
                        auto tmp928 = decltype(tmp927)(tmp927 & tmp891);
                        auto tmp929 = decltype(tmp928)(tmp928 & tmp921);
                        auto tmp931 = tmp930 == tmp1;
                        auto tmp932 = decltype(tmp929)(tmp929 & tmp931);
                        auto tmp933 = decltype(tmp923)(tmp923 | tmp932);
                        auto tmp934 = decltype(tmp781)(tmp781 & tmp801);
                        auto tmp935 = decltype(tmp934)(tmp934 & tmp831);
                        auto tmp936 = decltype(tmp935)(tmp935 & tmp841);
                        auto tmp937 = decltype(tmp936)(tmp936 & tmp881);
                        auto tmp938 = decltype(tmp937)(tmp937 & tmp901);
                        auto tmp939 = decltype(tmp938)(tmp938 & tmp931);
                        auto tmp941 = tmp940 == tmp1;
                        auto tmp942 = decltype(tmp939)(tmp939 & tmp941);
                        auto tmp943 = decltype(tmp933)(tmp933 | tmp942);
                        auto tmp944 = decltype(tmp791)(tmp791 & tmp811);
                        auto tmp945 = decltype(tmp944)(tmp944 & tmp841);
                        auto tmp946 = decltype(tmp945)(tmp945 & tmp851);
                        auto tmp947 = decltype(tmp946)(tmp946 & tmp891);
                        auto tmp948 = decltype(tmp947)(tmp947 & tmp911);
                        auto tmp949 = decltype(tmp948)(tmp948 & tmp941);
                        auto tmp951 = tmp950 == tmp1;
                        auto tmp952 = decltype(tmp949)(tmp949 & tmp951);
                        auto tmp953 = decltype(tmp943)(tmp943 | tmp952);
                        auto tmp954 = decltype(tmp801)(tmp801 & tmp821);
                        auto tmp955 = decltype(tmp954)(tmp954 & tmp851);
                        auto tmp956 = decltype(tmp955)(tmp955 & tmp861);
                        auto tmp957 = decltype(tmp956)(tmp956 & tmp901);
                        auto tmp958 = decltype(tmp957)(tmp957 & tmp921);
                        auto tmp959 = decltype(tmp958)(tmp958 & tmp951);
                        auto tmp961 = tmp960 == tmp1;
                        auto tmp962 = decltype(tmp959)(tmp959 & tmp961);
                        auto tmp963 = decltype(tmp953)(tmp953 | tmp962);
                        auto tmp964 = decltype(tmp811)(tmp811 & tmp831);
                        auto tmp965 = decltype(tmp964)(tmp964 & tmp861);
                        auto tmp966 = decltype(tmp965)(tmp965 & tmp871);
                        auto tmp967 = decltype(tmp966)(tmp966 & tmp911);
                        auto tmp968 = decltype(tmp967)(tmp967 & tmp931);
                        auto tmp969 = decltype(tmp968)(tmp968 & tmp961);
                        auto tmp971 = tmp970 == tmp1;
                        auto tmp972 = decltype(tmp969)(tmp969 & tmp971);
                        auto tmp973 = decltype(tmp963)(tmp963 | tmp972);
                        auto tmp974 = decltype(tmp821)(tmp821 & tmp841);
                        auto tmp975 = decltype(tmp974)(tmp974 & tmp871);
                        auto tmp976 = decltype(tmp975)(tmp975 & tmp881);
                        auto tmp977 = decltype(tmp976)(tmp976 & tmp921);
                        auto tmp978 = decltype(tmp977)(tmp977 & tmp941);
                        auto tmp979 = decltype(tmp978)(tmp978 & tmp971);
                        auto tmp981 = tmp980 == tmp1;
                        auto tmp982 = decltype(tmp979)(tmp979 & tmp981);
                        auto tmp983 = decltype(tmp973)(tmp973 | tmp982);
                        auto tmp984 = decltype(tmp831)(tmp831 & tmp851);
                        auto tmp985 = decltype(tmp984)(tmp984 & tmp881);
                        auto tmp986 = decltype(tmp985)(tmp985 & tmp891);
                        auto tmp987 = decltype(tmp986)(tmp986 & tmp931);
                        auto tmp988 = decltype(tmp987)(tmp987 & tmp951);
                        auto tmp989 = decltype(tmp988)(tmp988 & tmp981);
                        auto tmp991 = tmp990 == tmp1;
                        auto tmp992 = decltype(tmp989)(tmp989 & tmp991);
                        auto tmp993 = decltype(tmp983)(tmp983 | tmp992);
                        auto tmp994 = decltype(tmp841)(tmp841 & tmp861);
                        auto tmp995 = decltype(tmp994)(tmp994 & tmp891);
                        auto tmp996 = decltype(tmp995)(tmp995 & tmp901);
                        auto tmp997 = decltype(tmp996)(tmp996 & tmp941);
                        auto tmp998 = decltype(tmp997)(tmp997 & tmp961);
                        auto tmp999 = decltype(tmp998)(tmp998 & tmp991);
                        auto tmp1001 = tmp1000 == tmp1;
                        auto tmp1002 = decltype(tmp999)(tmp999 & tmp1001);
                        auto tmp1003 = decltype(tmp993)(tmp993 | tmp1002);
                        auto tmp1004 = decltype(tmp851)(tmp851 & tmp871);
                        auto tmp1005 = decltype(tmp1004)(tmp1004 & tmp901);
                        auto tmp1006 = decltype(tmp1005)(tmp1005 & tmp911);
                        auto tmp1007 = decltype(tmp1006)(tmp1006 & tmp951);
                        auto tmp1008 = decltype(tmp1007)(tmp1007 & tmp971);
                        auto tmp1009 = decltype(tmp1008)(tmp1008 & tmp1001);
                        auto tmp1011 = tmp1010 == tmp1;
                        auto tmp1012 = decltype(tmp1009)(tmp1009 & tmp1011);
                        auto tmp1013 = decltype(tmp1003)(tmp1003 | tmp1012);
                        auto tmp1014 = decltype(tmp861)(tmp861 & tmp881);
                        auto tmp1015 = decltype(tmp1014)(tmp1014 & tmp911);
                        auto tmp1016 = decltype(tmp1015)(tmp1015 & tmp921);
                        auto tmp1017 = decltype(tmp1016)(tmp1016 & tmp961);
                        auto tmp1018 = decltype(tmp1017)(tmp1017 & tmp981);
                        auto tmp1019 = decltype(tmp1018)(tmp1018 & tmp1011);
                        auto tmp1021 = tmp1020 == tmp1;
                        auto tmp1022 = decltype(tmp1019)(tmp1019 & tmp1021);
                        auto tmp1023 = decltype(tmp1013)(tmp1013 | tmp1022);
                        auto tmp1024 = decltype(tmp871)(tmp871 & tmp891);
                        auto tmp1025 = decltype(tmp1024)(tmp1024 & tmp921);
                        auto tmp1026 = decltype(tmp1025)(tmp1025 & tmp931);
                        auto tmp1027 = decltype(tmp1026)(tmp1026 & tmp971);
                        auto tmp1028 = decltype(tmp1027)(tmp1027 & tmp991);
                        auto tmp1029 = decltype(tmp1028)(tmp1028 & tmp1021);
                        auto tmp1031 = tmp1030 == tmp1;
                        auto tmp1032 = decltype(tmp1029)(tmp1029 & tmp1031);
                        auto tmp1033 = decltype(tmp1023)(tmp1023 | tmp1032);
                        auto tmp1035 = static_cast<int64_t>(8);
                        auto tmp1036 = tmp1034 >= tmp1035;
                        auto tmp1037 = decltype(tmp1033)(tmp1033 & tmp1036);
                        auto tmp1038 = c10::convert<float>(tmp1037);
                        out_ptr4[static_cast<int64_t>(0L)] = tmp1038;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_tanh_6 = async_compile.cpp_pybinding(['float*'], r'''
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


cpp_fused_add_fill_full_like_lift_fresh_mul_rsub_select_7 = async_compile.cpp_pybinding(['float*', 'const float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(16L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0L) && x0 < static_cast<int64_t>(2L)))
                {
                    for (int64_t x0_tail = static_cast<int64_t>(0L);x0_tail < static_cast<int64_t>(2L); x0_tail++)
                    {
                        auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                        auto tmp11 = in_out_ptr0[static_cast<int64_t>(x0_tail)];
                        auto tmp1 = x0_tail;
                        auto tmp2 = c10::convert<int32_t>(tmp1);
                        auto tmp3 = static_cast<int32_t>(1);
                        auto tmp4 = tmp2 == tmp3;
                        auto tmp5 = static_cast<float>(100.0);
                        auto tmp6 = static_cast<float>(-100.0);
                        auto tmp7 = tmp4 ? tmp5 : tmp6;
                        auto tmp8 = float(tmp0 * tmp7);
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = float(tmp9 - tmp0);
                        auto tmp12 = float(tmp10 * tmp11);
                        auto tmp13 = float(tmp8 + tmp12);
                        in_out_ptr0[static_cast<int64_t>(x0_tail)] = tmp13;
                    }
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 512), (512, 1))
        assert_size_stride(arg1_1, (1, 512), (512, 1))
        assert_size_stride(arg2_1, (1, 512), (512, 1))
        assert_size_stride(arg3_1, (30522, 768), (768, 1))
        assert_size_stride(arg4_1, (1, 512), (512, 1))
        assert_size_stride(arg5_1, (2, 768), (768, 1))
        assert_size_stride(arg6_1, (512, 768), (768, 1))
        assert_size_stride(arg7_1, (768, ), (1, ))
        assert_size_stride(arg8_1, (768, ), (1, ))
        assert_size_stride(arg9_1, (768, 768), (768, 1))
        assert_size_stride(arg10_1, (768, ), (1, ))
        assert_size_stride(arg11_1, (768, 768), (768, 1))
        assert_size_stride(arg12_1, (768, ), (1, ))
        assert_size_stride(arg13_1, (768, 768), (768, 1))
        assert_size_stride(arg14_1, (768, ), (1, ))
        assert_size_stride(arg15_1, (768, 768), (768, 1))
        assert_size_stride(arg16_1, (768, ), (1, ))
        assert_size_stride(arg17_1, (768, ), (1, ))
        assert_size_stride(arg18_1, (768, ), (1, ))
        assert_size_stride(arg19_1, (3072, 768), (768, 1))
        assert_size_stride(arg20_1, (3072, ), (1, ))
        assert_size_stride(arg21_1, (768, 3072), (3072, 1))
        assert_size_stride(arg22_1, (768, ), (1, ))
        assert_size_stride(arg23_1, (768, ), (1, ))
        assert_size_stride(arg24_1, (768, ), (1, ))
        assert_size_stride(arg25_1, (768, 768), (768, 1))
        assert_size_stride(arg26_1, (768, ), (1, ))
        assert_size_stride(arg27_1, (768, 768), (768, 1))
        assert_size_stride(arg28_1, (768, ), (1, ))
        assert_size_stride(arg29_1, (768, 768), (768, 1))
        assert_size_stride(arg30_1, (768, ), (1, ))
        assert_size_stride(arg31_1, (768, 768), (768, 1))
        assert_size_stride(arg32_1, (768, ), (1, ))
        assert_size_stride(arg33_1, (768, ), (1, ))
        assert_size_stride(arg34_1, (768, ), (1, ))
        assert_size_stride(arg35_1, (3072, 768), (768, 1))
        assert_size_stride(arg36_1, (3072, ), (1, ))
        assert_size_stride(arg37_1, (768, 3072), (3072, 1))
        assert_size_stride(arg38_1, (768, ), (1, ))
        assert_size_stride(arg39_1, (768, ), (1, ))
        assert_size_stride(arg40_1, (768, ), (1, ))
        assert_size_stride(arg41_1, (768, 768), (768, 1))
        assert_size_stride(arg42_1, (768, ), (1, ))
        assert_size_stride(arg43_1, (768, 768), (768, 1))
        assert_size_stride(arg44_1, (768, ), (1, ))
        assert_size_stride(arg45_1, (768, 768), (768, 1))
        assert_size_stride(arg46_1, (768, ), (1, ))
        assert_size_stride(arg47_1, (768, 768), (768, 1))
        assert_size_stride(arg48_1, (768, ), (1, ))
        assert_size_stride(arg49_1, (768, ), (1, ))
        assert_size_stride(arg50_1, (768, ), (1, ))
        assert_size_stride(arg51_1, (3072, 768), (768, 1))
        assert_size_stride(arg52_1, (3072, ), (1, ))
        assert_size_stride(arg53_1, (768, 3072), (3072, 1))
        assert_size_stride(arg54_1, (768, ), (1, ))
        assert_size_stride(arg55_1, (768, ), (1, ))
        assert_size_stride(arg56_1, (768, ), (1, ))
        assert_size_stride(arg57_1, (768, 768), (768, 1))
        assert_size_stride(arg58_1, (768, ), (1, ))
        assert_size_stride(arg59_1, (768, 768), (768, 1))
        assert_size_stride(arg60_1, (768, ), (1, ))
        assert_size_stride(arg61_1, (768, 768), (768, 1))
        assert_size_stride(arg62_1, (768, ), (1, ))
        assert_size_stride(arg63_1, (768, 768), (768, 1))
        assert_size_stride(arg64_1, (768, ), (1, ))
        assert_size_stride(arg65_1, (768, ), (1, ))
        assert_size_stride(arg66_1, (768, ), (1, ))
        assert_size_stride(arg67_1, (3072, 768), (768, 1))
        assert_size_stride(arg68_1, (3072, ), (1, ))
        assert_size_stride(arg69_1, (768, 3072), (3072, 1))
        assert_size_stride(arg70_1, (768, ), (1, ))
        assert_size_stride(arg71_1, (768, ), (1, ))
        assert_size_stride(arg72_1, (768, ), (1, ))
        assert_size_stride(arg73_1, (768, 768), (768, 1))
        assert_size_stride(arg74_1, (768, ), (1, ))
        assert_size_stride(arg75_1, (768, 768), (768, 1))
        assert_size_stride(arg76_1, (768, ), (1, ))
        assert_size_stride(arg77_1, (768, 768), (768, 1))
        assert_size_stride(arg78_1, (768, ), (1, ))
        assert_size_stride(arg79_1, (768, 768), (768, 1))
        assert_size_stride(arg80_1, (768, ), (1, ))
        assert_size_stride(arg81_1, (768, ), (1, ))
        assert_size_stride(arg82_1, (768, ), (1, ))
        assert_size_stride(arg83_1, (3072, 768), (768, 1))
        assert_size_stride(arg84_1, (3072, ), (1, ))
        assert_size_stride(arg85_1, (768, 3072), (3072, 1))
        assert_size_stride(arg86_1, (768, ), (1, ))
        assert_size_stride(arg87_1, (768, ), (1, ))
        assert_size_stride(arg88_1, (768, ), (1, ))
        assert_size_stride(arg89_1, (768, 768), (768, 1))
        assert_size_stride(arg90_1, (768, ), (1, ))
        assert_size_stride(arg91_1, (768, 768), (768, 1))
        assert_size_stride(arg92_1, (768, ), (1, ))
        assert_size_stride(arg93_1, (768, 768), (768, 1))
        assert_size_stride(arg94_1, (768, ), (1, ))
        assert_size_stride(arg95_1, (768, 768), (768, 1))
        assert_size_stride(arg96_1, (768, ), (1, ))
        assert_size_stride(arg97_1, (768, ), (1, ))
        assert_size_stride(arg98_1, (768, ), (1, ))
        assert_size_stride(arg99_1, (3072, 768), (768, 1))
        assert_size_stride(arg100_1, (3072, ), (1, ))
        assert_size_stride(arg101_1, (768, 3072), (3072, 1))
        assert_size_stride(arg102_1, (768, ), (1, ))
        assert_size_stride(arg103_1, (768, ), (1, ))
        assert_size_stride(arg104_1, (768, ), (1, ))
        assert_size_stride(arg105_1, (768, 768), (768, 1))
        assert_size_stride(arg106_1, (768, ), (1, ))
        assert_size_stride(arg107_1, (768, 768), (768, 1))
        assert_size_stride(arg108_1, (768, ), (1, ))
        assert_size_stride(arg109_1, (768, 768), (768, 1))
        assert_size_stride(arg110_1, (768, ), (1, ))
        assert_size_stride(arg111_1, (768, 768), (768, 1))
        assert_size_stride(arg112_1, (768, ), (1, ))
        assert_size_stride(arg113_1, (768, ), (1, ))
        assert_size_stride(arg114_1, (768, ), (1, ))
        assert_size_stride(arg115_1, (3072, 768), (768, 1))
        assert_size_stride(arg116_1, (3072, ), (1, ))
        assert_size_stride(arg117_1, (768, 3072), (3072, 1))
        assert_size_stride(arg118_1, (768, ), (1, ))
        assert_size_stride(arg119_1, (768, ), (1, ))
        assert_size_stride(arg120_1, (768, ), (1, ))
        assert_size_stride(arg121_1, (768, 768), (768, 1))
        assert_size_stride(arg122_1, (768, ), (1, ))
        assert_size_stride(arg123_1, (768, 768), (768, 1))
        assert_size_stride(arg124_1, (768, ), (1, ))
        assert_size_stride(arg125_1, (768, 768), (768, 1))
        assert_size_stride(arg126_1, (768, ), (1, ))
        assert_size_stride(arg127_1, (768, 768), (768, 1))
        assert_size_stride(arg128_1, (768, ), (1, ))
        assert_size_stride(arg129_1, (768, ), (1, ))
        assert_size_stride(arg130_1, (768, ), (1, ))
        assert_size_stride(arg131_1, (3072, 768), (768, 1))
        assert_size_stride(arg132_1, (3072, ), (1, ))
        assert_size_stride(arg133_1, (768, 3072), (3072, 1))
        assert_size_stride(arg134_1, (768, ), (1, ))
        assert_size_stride(arg135_1, (768, ), (1, ))
        assert_size_stride(arg136_1, (768, ), (1, ))
        assert_size_stride(arg137_1, (768, 768), (768, 1))
        assert_size_stride(arg138_1, (768, ), (1, ))
        assert_size_stride(arg139_1, (768, 768), (768, 1))
        assert_size_stride(arg140_1, (768, ), (1, ))
        assert_size_stride(arg141_1, (768, 768), (768, 1))
        assert_size_stride(arg142_1, (768, ), (1, ))
        assert_size_stride(arg143_1, (768, 768), (768, 1))
        assert_size_stride(arg144_1, (768, ), (1, ))
        assert_size_stride(arg145_1, (768, ), (1, ))
        assert_size_stride(arg146_1, (768, ), (1, ))
        assert_size_stride(arg147_1, (3072, 768), (768, 1))
        assert_size_stride(arg148_1, (3072, ), (1, ))
        assert_size_stride(arg149_1, (768, 3072), (3072, 1))
        assert_size_stride(arg150_1, (768, ), (1, ))
        assert_size_stride(arg151_1, (768, ), (1, ))
        assert_size_stride(arg152_1, (768, ), (1, ))
        assert_size_stride(arg153_1, (768, 768), (768, 1))
        assert_size_stride(arg154_1, (768, ), (1, ))
        assert_size_stride(arg155_1, (768, 768), (768, 1))
        assert_size_stride(arg156_1, (768, ), (1, ))
        assert_size_stride(arg157_1, (768, 768), (768, 1))
        assert_size_stride(arg158_1, (768, ), (1, ))
        assert_size_stride(arg159_1, (768, 768), (768, 1))
        assert_size_stride(arg160_1, (768, ), (1, ))
        assert_size_stride(arg161_1, (768, ), (1, ))
        assert_size_stride(arg162_1, (768, ), (1, ))
        assert_size_stride(arg163_1, (3072, 768), (768, 1))
        assert_size_stride(arg164_1, (3072, ), (1, ))
        assert_size_stride(arg165_1, (768, 3072), (3072, 1))
        assert_size_stride(arg166_1, (768, ), (1, ))
        assert_size_stride(arg167_1, (768, ), (1, ))
        assert_size_stride(arg168_1, (768, ), (1, ))
        assert_size_stride(arg169_1, (768, 768), (768, 1))
        assert_size_stride(arg170_1, (768, ), (1, ))
        assert_size_stride(arg171_1, (768, 768), (768, 1))
        assert_size_stride(arg172_1, (768, ), (1, ))
        assert_size_stride(arg173_1, (768, 768), (768, 1))
        assert_size_stride(arg174_1, (768, ), (1, ))
        assert_size_stride(arg175_1, (768, 768), (768, 1))
        assert_size_stride(arg176_1, (768, ), (1, ))
        assert_size_stride(arg177_1, (768, ), (1, ))
        assert_size_stride(arg178_1, (768, ), (1, ))
        assert_size_stride(arg179_1, (3072, 768), (768, 1))
        assert_size_stride(arg180_1, (3072, ), (1, ))
        assert_size_stride(arg181_1, (768, 3072), (3072, 1))
        assert_size_stride(arg182_1, (768, ), (1, ))
        assert_size_stride(arg183_1, (768, ), (1, ))
        assert_size_stride(arg184_1, (768, ), (1, ))
        assert_size_stride(arg185_1, (768, 768), (768, 1))
        assert_size_stride(arg186_1, (768, ), (1, ))
        assert_size_stride(arg187_1, (768, 768), (768, 1))
        assert_size_stride(arg188_1, (768, ), (1, ))
        assert_size_stride(arg189_1, (768, 768), (768, 1))
        assert_size_stride(arg190_1, (768, ), (1, ))
        assert_size_stride(arg191_1, (768, 768), (768, 1))
        assert_size_stride(arg192_1, (768, ), (1, ))
        assert_size_stride(arg193_1, (768, ), (1, ))
        assert_size_stride(arg194_1, (768, ), (1, ))
        assert_size_stride(arg195_1, (3072, 768), (768, 1))
        assert_size_stride(arg196_1, (3072, ), (1, ))
        assert_size_stride(arg197_1, (768, 3072), (3072, 1))
        assert_size_stride(arg198_1, (768, ), (1, ))
        assert_size_stride(arg199_1, (768, ), (1, ))
        assert_size_stride(arg200_1, (768, ), (1, ))
        assert_size_stride(arg201_1, (768, 768), (768, 1))
        assert_size_stride(arg202_1, (768, ), (1, ))
        assert_size_stride(arg203_1, (2, 768), (768, 1))
        assert_size_stride(arg204_1, (2, ), (1, ))
        buf1 = empty_strided_cpu((1, 512, 1), (512, 1, 512), torch.float32)
        buf2 = empty_strided_cpu((1, 512, 1), (512, 1, 512), torch.float32)
        buf4 = empty_strided_cpu((1, 512, 768), (393216, 768, 1), torch.float32)
        # [Provenance debug handles] cpp_fused_add_embedding_native_layer_norm_0:1
        cpp_fused_add_embedding_native_layer_norm_0(arg0_1, arg3_1, arg2_1, arg5_1, arg4_1, arg6_1, arg7_1, arg8_1, buf1, buf2, buf4)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del arg8_1
        buf5 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mixed_query_layer], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:2
        extern_kernels.addmm(arg10_1, reinterpret_tensor(buf4, (512, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf5)
        del arg10_1
        del arg9_1
        buf6 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:3
        extern_kernels.addmm(arg12_1, reinterpret_tensor(buf4, (512, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf6)
        del arg11_1
        del arg12_1
        buf7 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:4
        extern_kernels.addmm(arg14_1, reinterpret_tensor(buf4, (512, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf7)
        del arg13_1
        del arg14_1
        buf8 = reinterpret_tensor(buf2, (1, 1, 1, 512), (512, 512, 512, 1), 0); del buf2  # reuse
        buf27 = reinterpret_tensor(buf1, (1, 1, 1, 512), (512, 512, 512, 1), 0); del buf1  # reuse
        buf46 = empty_strided_cpu((1, 1, 1, 512), (512, 512, 512, 1), torch.float32)
        buf65 = empty_strided_cpu((1, 1, 1, 512), (512, 512, 512, 1), torch.float32)
        buf84 = empty_strided_cpu((1, 1, 1, 512), (512, 512, 512, 1), torch.float32)
        # [Provenance debug handles] cpp_fused__scaled_dot_product_flash_attention_for_cpu__to_copy_mul_permute_rsub_unsqueeze_view_1:5
        cpp_fused__scaled_dot_product_flash_attention_for_cpu__to_copy_mul_permute_rsub_unsqueeze_view_1(arg1_1, buf8, buf27, buf46, buf65, buf84)
        # Topologically Sorted Source Nodes: [mixed_query_layer, x_2, query_layer, linear_1, x, key_layer, linear_2, x_1, value_layer, extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, ], Original ATen: [aten.view, aten.permute, aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten._scaled_dot_product_flash_attention_for_cpu]
        # [Provenance debug handles] torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default:6
        buf9 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf5, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf6, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf7, (1, 12, 512, 64), (393216, 64, 768, 1), 0), attn_mask=buf8)
        del buf5
        del buf6
        buf10 = buf9[0]
        assert_size_stride(buf10, (1, 12, 512, 64), (393216, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf10, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf9
        buf12 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [permute_3, context_layer_2, hidden_states], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:7
        extern_kernels.addmm(arg16_1, reinterpret_tensor(buf10, (512, 768), (768, 1), 0), reinterpret_tensor(arg15_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf12)
        del arg15_1
        del arg16_1
        del buf10
        buf13 = reinterpret_tensor(buf8, (1, 512, 1), (512, 1, 512), 0); del buf8  # reuse
        buf14 = empty_strided_cpu((1, 512, 1), (512, 1, 512), torch.float32)
        buf16 = reinterpret_tensor(buf12, (1, 512, 768), (393216, 768, 1), 0); del buf12  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:8
        cpp_fused_add_native_layer_norm_view_2(buf16, buf4, arg17_1, arg18_1, buf13, buf14)
        del arg17_1
        del arg18_1
        del buf13
        del buf14
        del buf4
        buf17 = empty_strided_cpu((512, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_3], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:9
        extern_kernels.addmm(arg20_1, reinterpret_tensor(buf16, (512, 768), (768, 1), 0), reinterpret_tensor(arg19_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf17)
        del arg19_1
        del arg20_1
        buf18 = reinterpret_tensor(buf17, (1, 512, 3072), (1572864, 3072, 1), 0); del buf17  # reuse
        # [Provenance debug handles] cpp_fused_gelu_view_3:10
        cpp_fused_gelu_view_3(buf18)
        buf19 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4, hidden_states_5], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:11
        extern_kernels.addmm(arg22_1, reinterpret_tensor(buf18, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg21_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf19)
        del arg21_1
        del arg22_1
        del buf18
        buf20 = empty_strided_cpu((1, 512, 1), (512, 1, 512), torch.float32)
        buf21 = empty_strided_cpu((1, 512, 1), (512, 1, 512), torch.float32)
        buf23 = reinterpret_tensor(buf19, (1, 512, 768), (393216, 768, 1), 0); del buf19  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:12
        cpp_fused_add_native_layer_norm_view_2(buf23, buf16, arg23_1, arg24_1, buf20, buf21)
        del arg23_1
        del arg24_1
        del buf20
        buf24 = reinterpret_tensor(buf16, (512, 768), (768, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_1], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:13
        extern_kernels.addmm(arg26_1, reinterpret_tensor(buf23, (512, 768), (768, 1), 0), reinterpret_tensor(arg25_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf24)
        del arg25_1
        del arg26_1
        buf25 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:14
        extern_kernels.addmm(arg28_1, reinterpret_tensor(buf23, (512, 768), (768, 1), 0), reinterpret_tensor(arg27_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf25)
        del arg27_1
        del arg28_1
        buf26 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:15
        extern_kernels.addmm(arg30_1, reinterpret_tensor(buf23, (512, 768), (768, 1), 0), reinterpret_tensor(arg29_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf26)
        del arg29_1
        del arg30_1
        # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_1, x_5, query_layer_1, linear_7, x_3, key_layer_1, linear_8, x_4, value_layer_1, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        # [Provenance debug handles] torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default:16
        buf28 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf24, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf25, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf26, (1, 12, 512, 64), (393216, 64, 768, 1), 0), attn_mask=buf27)
        del buf24
        del buf25
        buf29 = buf28[0]
        assert_size_stride(buf29, (1, 12, 512, 64), (393216, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf29, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf28
        buf31 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [permute_7, context_layer_5, hidden_states_8], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:17
        extern_kernels.addmm(arg32_1, reinterpret_tensor(buf29, (512, 768), (768, 1), 0), reinterpret_tensor(arg31_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf31)
        del arg31_1
        del arg32_1
        del buf29
        buf32 = reinterpret_tensor(buf27, (1, 512, 1), (512, 1, 512), 0); del buf27  # reuse
        buf33 = buf21; del buf21  # reuse
        buf35 = reinterpret_tensor(buf31, (1, 512, 768), (393216, 768, 1), 0); del buf31  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:18
        cpp_fused_add_native_layer_norm_view_2(buf35, buf23, arg33_1, arg34_1, buf32, buf33)
        del arg33_1
        del arg34_1
        del buf23
        del buf32
        buf36 = empty_strided_cpu((512, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_11], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:19
        extern_kernels.addmm(arg36_1, reinterpret_tensor(buf35, (512, 768), (768, 1), 0), reinterpret_tensor(arg35_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf36)
        del arg35_1
        del arg36_1
        buf37 = reinterpret_tensor(buf36, (1, 512, 3072), (1572864, 3072, 1), 0); del buf36  # reuse
        # [Provenance debug handles] cpp_fused_gelu_view_3:20
        cpp_fused_gelu_view_3(buf37)
        buf38 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_11, hidden_states_12, hidden_states_13], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:21
        extern_kernels.addmm(arg38_1, reinterpret_tensor(buf37, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg37_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf38)
        del arg37_1
        del arg38_1
        buf39 = buf33; del buf33  # reuse
        buf40 = empty_strided_cpu((1, 512, 1), (512, 1, 512), torch.float32)
        buf42 = reinterpret_tensor(buf38, (1, 512, 768), (393216, 768, 1), 0); del buf38  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:22
        cpp_fused_add_native_layer_norm_view_2(buf42, buf35, arg39_1, arg40_1, buf39, buf40)
        del arg39_1
        del arg40_1
        del buf39
        buf43 = reinterpret_tensor(buf35, (512, 768), (768, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_2], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:23
        extern_kernels.addmm(arg42_1, reinterpret_tensor(buf42, (512, 768), (768, 1), 0), reinterpret_tensor(arg41_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf43)
        del arg41_1
        del arg42_1
        buf44 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:24
        extern_kernels.addmm(arg44_1, reinterpret_tensor(buf42, (512, 768), (768, 1), 0), reinterpret_tensor(arg43_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf44)
        del arg43_1
        del arg44_1
        buf45 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:25
        extern_kernels.addmm(arg46_1, reinterpret_tensor(buf42, (512, 768), (768, 1), 0), reinterpret_tensor(arg45_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf45)
        del arg45_1
        del arg46_1
        # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_2, x_8, query_layer_2, linear_13, x_6, key_layer_2, linear_14, x_7, value_layer_2, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        # [Provenance debug handles] torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default:26
        buf47 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf43, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf44, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf45, (1, 12, 512, 64), (393216, 64, 768, 1), 0), attn_mask=buf46)
        del buf43
        del buf44
        buf48 = buf47[0]
        assert_size_stride(buf48, (1, 12, 512, 64), (393216, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf48, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf47
        buf50 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [permute_11, context_layer_8, hidden_states_16], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:27
        extern_kernels.addmm(arg48_1, reinterpret_tensor(buf48, (512, 768), (768, 1), 0), reinterpret_tensor(arg47_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf50)
        del arg47_1
        del arg48_1
        del buf48
        buf51 = reinterpret_tensor(buf46, (1, 512, 1), (512, 1, 512), 0); del buf46  # reuse
        buf52 = buf40; del buf40  # reuse
        buf54 = reinterpret_tensor(buf50, (1, 512, 768), (393216, 768, 1), 0); del buf50  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:28
        cpp_fused_add_native_layer_norm_view_2(buf54, buf42, arg49_1, arg50_1, buf51, buf52)
        del arg49_1
        del arg50_1
        del buf42
        buf55 = reinterpret_tensor(buf37, (512, 3072), (3072, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_19], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:29
        extern_kernels.addmm(arg52_1, reinterpret_tensor(buf54, (512, 768), (768, 1), 0), reinterpret_tensor(arg51_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf55)
        del arg51_1
        del arg52_1
        buf56 = reinterpret_tensor(buf55, (1, 512, 3072), (1572864, 3072, 1), 0); del buf55  # reuse
        # [Provenance debug handles] cpp_fused_gelu_view_3:30
        cpp_fused_gelu_view_3(buf56)
        buf57 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20, hidden_states_21], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:31
        extern_kernels.addmm(arg54_1, reinterpret_tensor(buf56, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg53_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf57)
        del arg53_1
        del arg54_1
        buf58 = buf52; del buf52  # reuse
        buf59 = buf51; del buf51  # reuse
        buf61 = reinterpret_tensor(buf57, (1, 512, 768), (393216, 768, 1), 0); del buf57  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:32
        cpp_fused_add_native_layer_norm_view_2(buf61, buf54, arg55_1, arg56_1, buf58, buf59)
        del arg55_1
        del arg56_1
        buf62 = reinterpret_tensor(buf54, (512, 768), (768, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_3], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:33
        extern_kernels.addmm(arg58_1, reinterpret_tensor(buf61, (512, 768), (768, 1), 0), reinterpret_tensor(arg57_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf62)
        del arg57_1
        del arg58_1
        buf63 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:34
        extern_kernels.addmm(arg60_1, reinterpret_tensor(buf61, (512, 768), (768, 1), 0), reinterpret_tensor(arg59_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf63)
        del arg59_1
        del arg60_1
        buf64 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:35
        extern_kernels.addmm(arg62_1, reinterpret_tensor(buf61, (512, 768), (768, 1), 0), reinterpret_tensor(arg61_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf64)
        del arg61_1
        del arg62_1
        # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_3, x_11, query_layer_3, linear_19, x_9, key_layer_3, linear_20, x_10, value_layer_3, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        # [Provenance debug handles] torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default:36
        buf66 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf62, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf63, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf64, (1, 12, 512, 64), (393216, 64, 768, 1), 0), attn_mask=buf65)
        del buf62
        del buf63
        buf67 = buf66[0]
        assert_size_stride(buf67, (1, 12, 512, 64), (393216, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf67, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf66
        buf69 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [permute_15, context_layer_11, hidden_states_24], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:37
        extern_kernels.addmm(arg64_1, reinterpret_tensor(buf67, (512, 768), (768, 1), 0), reinterpret_tensor(arg63_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf69)
        del arg63_1
        del arg64_1
        del buf67
        buf70 = reinterpret_tensor(buf65, (1, 512, 1), (512, 1, 512), 0); del buf65  # reuse
        buf71 = buf59; del buf59  # reuse
        buf73 = reinterpret_tensor(buf69, (1, 512, 768), (393216, 768, 1), 0); del buf69  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:38
        cpp_fused_add_native_layer_norm_view_2(buf73, buf61, arg65_1, arg66_1, buf70, buf71)
        del arg65_1
        del arg66_1
        del buf61
        buf74 = reinterpret_tensor(buf56, (512, 3072), (3072, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_27], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:39
        extern_kernels.addmm(arg68_1, reinterpret_tensor(buf73, (512, 768), (768, 1), 0), reinterpret_tensor(arg67_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf74)
        del arg67_1
        del arg68_1
        buf75 = reinterpret_tensor(buf74, (1, 512, 3072), (1572864, 3072, 1), 0); del buf74  # reuse
        # [Provenance debug handles] cpp_fused_gelu_view_3:40
        cpp_fused_gelu_view_3(buf75)
        buf76 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_27, hidden_states_28, hidden_states_29], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:41
        extern_kernels.addmm(arg70_1, reinterpret_tensor(buf75, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg69_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf76)
        del arg69_1
        del arg70_1
        buf77 = buf71; del buf71  # reuse
        buf78 = buf70; del buf70  # reuse
        buf80 = reinterpret_tensor(buf76, (1, 512, 768), (393216, 768, 1), 0); del buf76  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:42
        cpp_fused_add_native_layer_norm_view_2(buf80, buf73, arg71_1, arg72_1, buf77, buf78)
        del arg71_1
        del arg72_1
        buf81 = reinterpret_tensor(buf73, (512, 768), (768, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_4], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:43
        extern_kernels.addmm(arg74_1, reinterpret_tensor(buf80, (512, 768), (768, 1), 0), reinterpret_tensor(arg73_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf81)
        del arg73_1
        del arg74_1
        buf82 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:44
        extern_kernels.addmm(arg76_1, reinterpret_tensor(buf80, (512, 768), (768, 1), 0), reinterpret_tensor(arg75_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf82)
        del arg75_1
        del arg76_1
        buf83 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:45
        extern_kernels.addmm(arg78_1, reinterpret_tensor(buf80, (512, 768), (768, 1), 0), reinterpret_tensor(arg77_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf83)
        del arg77_1
        del arg78_1
        # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_4, x_14, query_layer_4, linear_25, x_12, key_layer_4, linear_26, x_13, value_layer_4, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        # [Provenance debug handles] torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default:46
        buf85 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf81, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf82, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf83, (1, 12, 512, 64), (393216, 64, 768, 1), 0), attn_mask=buf84)
        del buf81
        del buf82
        buf86 = buf85[0]
        assert_size_stride(buf86, (1, 12, 512, 64), (393216, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf86, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf85
        buf88 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [permute_19, context_layer_14, hidden_states_32], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:47
        extern_kernels.addmm(arg80_1, reinterpret_tensor(buf86, (512, 768), (768, 1), 0), reinterpret_tensor(arg79_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf88)
        del arg79_1
        del arg80_1
        del buf86
        buf89 = reinterpret_tensor(buf84, (1, 512, 1), (512, 1, 512), 0); del buf84  # reuse
        buf90 = buf78; del buf78  # reuse
        buf92 = reinterpret_tensor(buf88, (1, 512, 768), (393216, 768, 1), 0); del buf88  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:48
        cpp_fused_add_native_layer_norm_view_2(buf92, buf80, arg81_1, arg82_1, buf89, buf90)
        del arg81_1
        del arg82_1
        del buf80
        buf93 = reinterpret_tensor(buf75, (512, 3072), (3072, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:49
        extern_kernels.addmm(arg84_1, reinterpret_tensor(buf92, (512, 768), (768, 1), 0), reinterpret_tensor(arg83_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf93)
        del arg83_1
        del arg84_1
        buf94 = reinterpret_tensor(buf93, (1, 512, 3072), (1572864, 3072, 1), 0); del buf93  # reuse
        # [Provenance debug handles] cpp_fused_gelu_view_3:50
        cpp_fused_gelu_view_3(buf94)
        buf95 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_36, hidden_states_37], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:51
        extern_kernels.addmm(arg86_1, reinterpret_tensor(buf94, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg85_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf95)
        del arg85_1
        del arg86_1
        del buf94
        buf96 = buf90; del buf90  # reuse
        buf97 = buf89; del buf89  # reuse
        buf99 = reinterpret_tensor(buf95, (1, 512, 768), (393216, 768, 1), 0); del buf95  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:52
        cpp_fused_add_native_layer_norm_view_2(buf99, buf92, arg87_1, arg88_1, buf96, buf97)
        del arg87_1
        del arg88_1
        buf100 = reinterpret_tensor(buf92, (512, 768), (768, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_5], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:53
        extern_kernels.addmm(arg90_1, reinterpret_tensor(buf99, (512, 768), (768, 1), 0), reinterpret_tensor(arg89_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf100)
        del arg89_1
        del arg90_1
        buf101 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:54
        extern_kernels.addmm(arg92_1, reinterpret_tensor(buf99, (512, 768), (768, 1), 0), reinterpret_tensor(arg91_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf101)
        del arg91_1
        del arg92_1
        buf102 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:55
        extern_kernels.addmm(arg94_1, reinterpret_tensor(buf99, (512, 768), (768, 1), 0), reinterpret_tensor(arg93_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf102)
        del arg93_1
        del arg94_1
        buf103 = reinterpret_tensor(buf97, (1, 1, 1, 512), (512, 512, 512, 1), 0); del buf97  # reuse
        buf122 = reinterpret_tensor(buf96, (1, 1, 1, 512), (512, 512, 512, 1), 0); del buf96  # reuse
        buf141 = reinterpret_tensor(buf77, (1, 1, 1, 512), (512, 512, 512, 1), 0); del buf77  # reuse
        buf160 = reinterpret_tensor(buf58, (1, 1, 1, 512), (512, 512, 512, 1), 0); del buf58  # reuse
        buf179 = empty_strided_cpu((1, 1, 1, 512), (512, 512, 512, 1), torch.float32)
        # [Provenance debug handles] cpp_fused__scaled_dot_product_flash_attention_for_cpu__to_copy_mul_permute_rsub_unsqueeze_view_1:56
        cpp_fused__scaled_dot_product_flash_attention_for_cpu__to_copy_mul_permute_rsub_unsqueeze_view_1(arg1_1, buf103, buf122, buf141, buf160, buf179)
        # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_5, x_17, query_layer_5, linear_31, x_15, key_layer_5, linear_32, x_16, value_layer_5, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        # [Provenance debug handles] torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default:57
        buf104 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf100, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf101, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf102, (1, 12, 512, 64), (393216, 64, 768, 1), 0), attn_mask=buf103)
        del buf100
        del buf101
        buf105 = buf104[0]
        assert_size_stride(buf105, (1, 12, 512, 64), (393216, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf105, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf104
        buf107 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [permute_23, context_layer_17, hidden_states_40], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:58
        extern_kernels.addmm(arg96_1, reinterpret_tensor(buf105, (512, 768), (768, 1), 0), reinterpret_tensor(arg95_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf107)
        del arg95_1
        del arg96_1
        del buf105
        buf108 = reinterpret_tensor(buf103, (1, 512, 1), (512, 1, 512), 0); del buf103  # reuse
        buf109 = empty_strided_cpu((1, 512, 1), (512, 1, 512), torch.float32)
        buf111 = reinterpret_tensor(buf107, (1, 512, 768), (393216, 768, 1), 0); del buf107  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:59
        cpp_fused_add_native_layer_norm_view_2(buf111, buf99, arg97_1, arg98_1, buf108, buf109)
        del arg97_1
        del arg98_1
        del buf108
        del buf109
        del buf99
        buf112 = empty_strided_cpu((512, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_43], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:60
        extern_kernels.addmm(arg100_1, reinterpret_tensor(buf111, (512, 768), (768, 1), 0), reinterpret_tensor(arg99_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf112)
        del arg100_1
        del arg99_1
        buf113 = reinterpret_tensor(buf112, (1, 512, 3072), (1572864, 3072, 1), 0); del buf112  # reuse
        # [Provenance debug handles] cpp_fused_gelu_view_3:61
        cpp_fused_gelu_view_3(buf113)
        buf114 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_43, hidden_states_44, hidden_states_45], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:62
        extern_kernels.addmm(arg102_1, reinterpret_tensor(buf113, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg101_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf114)
        del arg101_1
        del arg102_1
        del buf113
        buf115 = empty_strided_cpu((1, 512, 1), (512, 1, 512), torch.float32)
        buf116 = empty_strided_cpu((1, 512, 1), (512, 1, 512), torch.float32)
        buf118 = reinterpret_tensor(buf114, (1, 512, 768), (393216, 768, 1), 0); del buf114  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:63
        cpp_fused_add_native_layer_norm_view_2(buf118, buf111, arg103_1, arg104_1, buf115, buf116)
        del arg103_1
        del arg104_1
        del buf115
        buf119 = reinterpret_tensor(buf111, (512, 768), (768, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_6], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:64
        extern_kernels.addmm(arg106_1, reinterpret_tensor(buf118, (512, 768), (768, 1), 0), reinterpret_tensor(arg105_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf119)
        del arg105_1
        del arg106_1
        buf120 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:65
        extern_kernels.addmm(arg108_1, reinterpret_tensor(buf118, (512, 768), (768, 1), 0), reinterpret_tensor(arg107_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf120)
        del arg107_1
        del arg108_1
        buf121 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:66
        extern_kernels.addmm(arg110_1, reinterpret_tensor(buf118, (512, 768), (768, 1), 0), reinterpret_tensor(arg109_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf121)
        del arg109_1
        del arg110_1
        # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_6, x_20, query_layer_6, linear_37, x_18, key_layer_6, linear_38, x_19, value_layer_6, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        # [Provenance debug handles] torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default:67
        buf123 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf119, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf120, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf121, (1, 12, 512, 64), (393216, 64, 768, 1), 0), attn_mask=buf122)
        del buf119
        del buf120
        buf124 = buf123[0]
        assert_size_stride(buf124, (1, 12, 512, 64), (393216, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf124, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf123
        buf126 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [permute_27, context_layer_20, hidden_states_48], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:68
        extern_kernels.addmm(arg112_1, reinterpret_tensor(buf124, (512, 768), (768, 1), 0), reinterpret_tensor(arg111_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf126)
        del arg111_1
        del arg112_1
        del buf124
        buf127 = reinterpret_tensor(buf122, (1, 512, 1), (512, 1, 512), 0); del buf122  # reuse
        buf128 = buf116; del buf116  # reuse
        buf130 = reinterpret_tensor(buf126, (1, 512, 768), (393216, 768, 1), 0); del buf126  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:69
        cpp_fused_add_native_layer_norm_view_2(buf130, buf118, arg113_1, arg114_1, buf127, buf128)
        del arg113_1
        del arg114_1
        del buf118
        del buf127
        buf131 = empty_strided_cpu((512, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_51], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:70
        extern_kernels.addmm(arg116_1, reinterpret_tensor(buf130, (512, 768), (768, 1), 0), reinterpret_tensor(arg115_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf131)
        del arg115_1
        del arg116_1
        buf132 = reinterpret_tensor(buf131, (1, 512, 3072), (1572864, 3072, 1), 0); del buf131  # reuse
        # [Provenance debug handles] cpp_fused_gelu_view_3:71
        cpp_fused_gelu_view_3(buf132)
        buf133 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_51, hidden_states_52, hidden_states_53], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:72
        extern_kernels.addmm(arg118_1, reinterpret_tensor(buf132, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg117_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf133)
        del arg117_1
        del arg118_1
        buf134 = buf128; del buf128  # reuse
        buf135 = empty_strided_cpu((1, 512, 1), (512, 1, 512), torch.float32)
        buf137 = reinterpret_tensor(buf133, (1, 512, 768), (393216, 768, 1), 0); del buf133  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:73
        cpp_fused_add_native_layer_norm_view_2(buf137, buf130, arg119_1, arg120_1, buf134, buf135)
        del arg119_1
        del arg120_1
        del buf134
        buf138 = reinterpret_tensor(buf130, (512, 768), (768, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_7], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:74
        extern_kernels.addmm(arg122_1, reinterpret_tensor(buf137, (512, 768), (768, 1), 0), reinterpret_tensor(arg121_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf138)
        del arg121_1
        del arg122_1
        buf139 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:75
        extern_kernels.addmm(arg124_1, reinterpret_tensor(buf137, (512, 768), (768, 1), 0), reinterpret_tensor(arg123_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf139)
        del arg123_1
        del arg124_1
        buf140 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:76
        extern_kernels.addmm(arg126_1, reinterpret_tensor(buf137, (512, 768), (768, 1), 0), reinterpret_tensor(arg125_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf140)
        del arg125_1
        del arg126_1
        # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_7, x_23, query_layer_7, linear_43, x_21, key_layer_7, linear_44, x_22, value_layer_7, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        # [Provenance debug handles] torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default:77
        buf142 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf138, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf139, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf140, (1, 12, 512, 64), (393216, 64, 768, 1), 0), attn_mask=buf141)
        del buf138
        del buf139
        buf143 = buf142[0]
        assert_size_stride(buf143, (1, 12, 512, 64), (393216, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf143, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf142
        buf145 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [permute_31, context_layer_23, hidden_states_56], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:78
        extern_kernels.addmm(arg128_1, reinterpret_tensor(buf143, (512, 768), (768, 1), 0), reinterpret_tensor(arg127_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf145)
        del arg127_1
        del arg128_1
        del buf143
        buf146 = reinterpret_tensor(buf141, (1, 512, 1), (512, 1, 512), 0); del buf141  # reuse
        buf147 = buf135; del buf135  # reuse
        buf149 = reinterpret_tensor(buf145, (1, 512, 768), (393216, 768, 1), 0); del buf145  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:79
        cpp_fused_add_native_layer_norm_view_2(buf149, buf137, arg129_1, arg130_1, buf146, buf147)
        del arg129_1
        del arg130_1
        del buf137
        buf150 = reinterpret_tensor(buf132, (512, 3072), (3072, 1), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_59], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:80
        extern_kernels.addmm(arg132_1, reinterpret_tensor(buf149, (512, 768), (768, 1), 0), reinterpret_tensor(arg131_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf150)
        del arg131_1
        del arg132_1
        buf151 = reinterpret_tensor(buf150, (1, 512, 3072), (1572864, 3072, 1), 0); del buf150  # reuse
        # [Provenance debug handles] cpp_fused_gelu_view_3:81
        cpp_fused_gelu_view_3(buf151)
        buf152 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_59, hidden_states_60, hidden_states_61], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:82
        extern_kernels.addmm(arg134_1, reinterpret_tensor(buf151, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg133_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf152)
        del arg133_1
        del arg134_1
        buf153 = buf147; del buf147  # reuse
        buf154 = buf146; del buf146  # reuse
        buf156 = reinterpret_tensor(buf152, (1, 512, 768), (393216, 768, 1), 0); del buf152  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:83
        cpp_fused_add_native_layer_norm_view_2(buf156, buf149, arg135_1, arg136_1, buf153, buf154)
        del arg135_1
        del arg136_1
        del buf153
        buf157 = reinterpret_tensor(buf149, (512, 768), (768, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_8], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:84
        extern_kernels.addmm(arg138_1, reinterpret_tensor(buf156, (512, 768), (768, 1), 0), reinterpret_tensor(arg137_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf157)
        del arg137_1
        del arg138_1
        buf158 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:85
        extern_kernels.addmm(arg140_1, reinterpret_tensor(buf156, (512, 768), (768, 1), 0), reinterpret_tensor(arg139_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf158)
        del arg139_1
        del arg140_1
        buf159 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:86
        extern_kernels.addmm(arg142_1, reinterpret_tensor(buf156, (512, 768), (768, 1), 0), reinterpret_tensor(arg141_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf159)
        del arg141_1
        del arg142_1
        # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_8, x_26, query_layer_8, linear_49, x_24, key_layer_8, linear_50, x_25, value_layer_8, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        # [Provenance debug handles] torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default:87
        buf161 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf157, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf158, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf159, (1, 12, 512, 64), (393216, 64, 768, 1), 0), attn_mask=buf160)
        del buf157
        del buf158
        buf162 = buf161[0]
        assert_size_stride(buf162, (1, 12, 512, 64), (393216, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf162, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf161
        buf164 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [permute_35, context_layer_26, hidden_states_64], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:88
        extern_kernels.addmm(arg144_1, reinterpret_tensor(buf162, (512, 768), (768, 1), 0), reinterpret_tensor(arg143_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf164)
        del arg143_1
        del arg144_1
        del buf162
        buf165 = reinterpret_tensor(buf160, (1, 512, 1), (512, 1, 512), 0); del buf160  # reuse
        buf166 = buf154; del buf154  # reuse
        buf168 = reinterpret_tensor(buf164, (1, 512, 768), (393216, 768, 1), 0); del buf164  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:89
        cpp_fused_add_native_layer_norm_view_2(buf168, buf156, arg145_1, arg146_1, buf165, buf166)
        del arg145_1
        del arg146_1
        del buf156
        buf169 = reinterpret_tensor(buf151, (512, 3072), (3072, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_67], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:90
        extern_kernels.addmm(arg148_1, reinterpret_tensor(buf168, (512, 768), (768, 1), 0), reinterpret_tensor(arg147_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf169)
        del arg147_1
        del arg148_1
        buf170 = reinterpret_tensor(buf169, (1, 512, 3072), (1572864, 3072, 1), 0); del buf169  # reuse
        # [Provenance debug handles] cpp_fused_gelu_view_3:91
        cpp_fused_gelu_view_3(buf170)
        buf171 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_67, hidden_states_68, hidden_states_69], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:92
        extern_kernels.addmm(arg150_1, reinterpret_tensor(buf170, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg149_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf171)
        del arg149_1
        del arg150_1
        buf172 = buf166; del buf166  # reuse
        buf173 = buf165; del buf165  # reuse
        buf175 = reinterpret_tensor(buf171, (1, 512, 768), (393216, 768, 1), 0); del buf171  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:93
        cpp_fused_add_native_layer_norm_view_2(buf175, buf168, arg151_1, arg152_1, buf172, buf173)
        del arg151_1
        del arg152_1
        buf176 = reinterpret_tensor(buf168, (512, 768), (768, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_9], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:94
        extern_kernels.addmm(arg154_1, reinterpret_tensor(buf175, (512, 768), (768, 1), 0), reinterpret_tensor(arg153_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf176)
        del arg153_1
        del arg154_1
        buf177 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_55], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:95
        extern_kernels.addmm(arg156_1, reinterpret_tensor(buf175, (512, 768), (768, 1), 0), reinterpret_tensor(arg155_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf177)
        del arg155_1
        del arg156_1
        buf178 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:96
        extern_kernels.addmm(arg158_1, reinterpret_tensor(buf175, (512, 768), (768, 1), 0), reinterpret_tensor(arg157_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf178)
        del arg157_1
        del arg158_1
        # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_9, x_29, query_layer_9, linear_55, x_27, key_layer_9, linear_56, x_28, value_layer_9, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        # [Provenance debug handles] torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default:97
        buf180 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf176, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf177, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf178, (1, 12, 512, 64), (393216, 64, 768, 1), 0), attn_mask=buf179)
        del buf176
        del buf177
        buf181 = buf180[0]
        assert_size_stride(buf181, (1, 12, 512, 64), (393216, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf181, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf180
        buf183 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [permute_39, context_layer_29, hidden_states_72], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:98
        extern_kernels.addmm(arg160_1, reinterpret_tensor(buf181, (512, 768), (768, 1), 0), reinterpret_tensor(arg159_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf183)
        del arg159_1
        del arg160_1
        del buf181
        buf184 = reinterpret_tensor(buf179, (1, 512, 1), (512, 1, 512), 0); del buf179  # reuse
        buf185 = buf173; del buf173  # reuse
        buf187 = reinterpret_tensor(buf183, (1, 512, 768), (393216, 768, 1), 0); del buf183  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:99
        cpp_fused_add_native_layer_norm_view_2(buf187, buf175, arg161_1, arg162_1, buf184, buf185)
        del arg161_1
        del arg162_1
        del buf175
        buf188 = reinterpret_tensor(buf170, (512, 3072), (3072, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_75], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:100
        extern_kernels.addmm(arg164_1, reinterpret_tensor(buf187, (512, 768), (768, 1), 0), reinterpret_tensor(arg163_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf188)
        del arg163_1
        del arg164_1
        buf189 = reinterpret_tensor(buf188, (1, 512, 3072), (1572864, 3072, 1), 0); del buf188  # reuse
        # [Provenance debug handles] cpp_fused_gelu_view_3:101
        cpp_fused_gelu_view_3(buf189)
        buf190 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_75, hidden_states_76, hidden_states_77], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:102
        extern_kernels.addmm(arg166_1, reinterpret_tensor(buf189, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg165_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf190)
        del arg165_1
        del arg166_1
        buf191 = buf185; del buf185  # reuse
        buf192 = buf184; del buf184  # reuse
        buf194 = reinterpret_tensor(buf190, (1, 512, 768), (393216, 768, 1), 0); del buf190  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:103
        cpp_fused_add_native_layer_norm_view_2(buf194, buf187, arg167_1, arg168_1, buf191, buf192)
        del arg167_1
        del arg168_1
        buf195 = reinterpret_tensor(buf187, (512, 768), (768, 1), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_10], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:104
        extern_kernels.addmm(arg170_1, reinterpret_tensor(buf194, (512, 768), (768, 1), 0), reinterpret_tensor(arg169_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf195)
        del arg169_1
        del arg170_1
        buf196 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:105
        extern_kernels.addmm(arg172_1, reinterpret_tensor(buf194, (512, 768), (768, 1), 0), reinterpret_tensor(arg171_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf196)
        del arg171_1
        del arg172_1
        buf197 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_62], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:106
        extern_kernels.addmm(arg174_1, reinterpret_tensor(buf194, (512, 768), (768, 1), 0), reinterpret_tensor(arg173_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf197)
        del arg173_1
        del arg174_1
        buf198 = reinterpret_tensor(buf192, (1, 1, 1, 512), (512, 512, 512, 1), 0); del buf192  # reuse
        buf217 = reinterpret_tensor(buf191, (1, 1, 1, 512), (512, 512, 512, 1), 0); del buf191  # reuse
        # [Provenance debug handles] cpp_fused__scaled_dot_product_flash_attention_for_cpu__to_copy_mul_permute_rsub_unsqueeze_view_4:107
        cpp_fused__scaled_dot_product_flash_attention_for_cpu__to_copy_mul_permute_rsub_unsqueeze_view_4(arg1_1, buf198, buf217)
        del arg1_1
        # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_10, x_32, query_layer_10, linear_61, x_30, key_layer_10, linear_62, x_31, value_layer_10, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        # [Provenance debug handles] torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default:108
        buf199 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf195, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf196, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf197, (1, 12, 512, 64), (393216, 64, 768, 1), 0), attn_mask=buf198)
        del buf195
        del buf196
        buf200 = buf199[0]
        assert_size_stride(buf200, (1, 12, 512, 64), (393216, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf200, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf199
        buf202 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [permute_43, context_layer_32, hidden_states_80], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:109
        extern_kernels.addmm(arg176_1, reinterpret_tensor(buf200, (512, 768), (768, 1), 0), reinterpret_tensor(arg175_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf202)
        del arg175_1
        del arg176_1
        del buf200
        buf203 = reinterpret_tensor(buf198, (1, 512, 1), (512, 1, 512), 0); del buf198  # reuse
        buf204 = buf172; del buf172  # reuse
        buf206 = reinterpret_tensor(buf202, (1, 512, 768), (393216, 768, 1), 0); del buf202  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:110
        cpp_fused_add_native_layer_norm_view_2(buf206, buf194, arg177_1, arg178_1, buf203, buf204)
        del arg177_1
        del arg178_1
        del buf194
        buf207 = reinterpret_tensor(buf189, (512, 3072), (3072, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_83], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:111
        extern_kernels.addmm(arg180_1, reinterpret_tensor(buf206, (512, 768), (768, 1), 0), reinterpret_tensor(arg179_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf207)
        del arg179_1
        del arg180_1
        buf208 = reinterpret_tensor(buf207, (1, 512, 3072), (1572864, 3072, 1), 0); del buf207  # reuse
        # [Provenance debug handles] cpp_fused_gelu_view_3:112
        cpp_fused_gelu_view_3(buf208)
        buf209 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_83, hidden_states_84, hidden_states_85], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:113
        extern_kernels.addmm(arg182_1, reinterpret_tensor(buf208, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg181_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf209)
        del arg181_1
        del arg182_1
        buf210 = buf204; del buf204  # reuse
        buf211 = buf203; del buf203  # reuse
        buf213 = reinterpret_tensor(buf209, (1, 512, 768), (393216, 768, 1), 0); del buf209  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:114
        cpp_fused_add_native_layer_norm_view_2(buf213, buf206, arg183_1, arg184_1, buf210, buf211)
        del arg183_1
        del arg184_1
        del buf210
        buf214 = reinterpret_tensor(buf206, (512, 768), (768, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_11], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:115
        extern_kernels.addmm(arg186_1, reinterpret_tensor(buf213, (512, 768), (768, 1), 0), reinterpret_tensor(arg185_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf214)
        del arg185_1
        del arg186_1
        buf215 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_67], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:116
        extern_kernels.addmm(arg188_1, reinterpret_tensor(buf213, (512, 768), (768, 1), 0), reinterpret_tensor(arg187_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf215)
        del arg187_1
        del arg188_1
        buf216 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:117
        extern_kernels.addmm(arg190_1, reinterpret_tensor(buf213, (512, 768), (768, 1), 0), reinterpret_tensor(arg189_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf216)
        del arg189_1
        del arg190_1
        # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_11, x_35, query_layer_11, linear_67, x_33, key_layer_11, linear_68, x_34, value_layer_11, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten._scaled_dot_product_flash_attention_for_cpu]
        # [Provenance debug handles] torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default:118
        buf218 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf214, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf215, (1, 12, 512, 64), (393216, 64, 768, 1), 0), reinterpret_tensor(buf216, (1, 12, 512, 64), (393216, 64, 768, 1), 0), attn_mask=buf217)
        del buf214
        del buf215
        buf219 = buf218[0]
        assert_size_stride(buf219, (1, 12, 512, 64), (393216, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf219, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf218
        buf221 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [permute_47, context_layer_35, hidden_states_88], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:119
        extern_kernels.addmm(arg192_1, reinterpret_tensor(buf219, (512, 768), (768, 1), 0), reinterpret_tensor(arg191_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf221)
        del arg191_1
        del arg192_1
        del buf219
        buf222 = reinterpret_tensor(buf217, (1, 512, 1), (512, 1, 512), 0); del buf217  # reuse
        buf223 = buf211; del buf211  # reuse
        buf225 = reinterpret_tensor(buf221, (1, 512, 768), (393216, 768, 1), 0); del buf221  # reuse
        # [Provenance debug handles] cpp_fused_add_native_layer_norm_view_2:120
        cpp_fused_add_native_layer_norm_view_2(buf225, buf213, arg193_1, arg194_1, buf222, buf223)
        del arg193_1
        del arg194_1
        del buf213
        buf226 = reinterpret_tensor(buf208, (512, 3072), (3072, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_91], Original ATen: [aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:121
        extern_kernels.addmm(arg196_1, reinterpret_tensor(buf225, (512, 768), (768, 1), 0), reinterpret_tensor(arg195_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf226)
        del arg195_1
        del arg196_1
        buf227 = reinterpret_tensor(buf226, (1, 512, 3072), (1572864, 3072, 1), 0); del buf226  # reuse
        # [Provenance debug handles] cpp_fused_gelu_view_3:122
        cpp_fused_gelu_view_3(buf227)
        buf228 = empty_strided_cpu((512, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_91, hidden_states_92, hidden_states_93], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:123
        extern_kernels.addmm(arg198_1, reinterpret_tensor(buf227, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg197_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf228)
        del arg197_1
        del arg198_1
        del buf227
        buf229 = buf223; del buf223  # reuse
        buf230 = buf222; del buf222  # reuse
        buf248 = reinterpret_tensor(buf228, (1, 512, 768), (393216, 768, 1), 0); del buf228  # reuse
        buf246 = empty_strided_cpu((1, ), (1, ), torch.int64)
        buf247 = empty_strided_cpu((1, 1), (1, 1), torch.float32)
        # [Provenance debug handles] cpp_fused__to_copy_add_bitwise_and_bitwise_or_eq_ge_native_layer_norm_select_sum_unsqueeze_view_zeros_5:124
        cpp_fused__to_copy_add_bitwise_and_bitwise_or_eq_ge_native_layer_norm_select_sum_unsqueeze_view_zeros_5(buf248, buf225, arg199_1, arg200_1, arg0_1, buf229, buf230, buf246, buf247)
        del arg0_1
        del arg199_1
        del arg200_1
        del buf225
        del buf229
        del buf230
        del buf246
        buf249 = empty_strided_cpu((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_93, add_36, hidden_states_95, first_token_tensor, pooled_output], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.select, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:125
        extern_kernels.addmm(arg202_1, reinterpret_tensor(buf248, (1, 768), (0, 1), 0), reinterpret_tensor(arg201_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf249)
        del arg201_1
        del arg202_1
        del buf248
        buf250 = buf249; del buf249  # reuse
        # [Provenance debug handles] cpp_fused_tanh_6:126
        cpp_fused_tanh_6(buf250)
        buf251 = empty_strided_cpu((1, 2), (2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pooled_output_1, logits], Original ATen: [aten.tanh, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:127
        extern_kernels.addmm(arg204_1, buf250, reinterpret_tensor(arg203_1, (768, 2), (1, 768), 0), alpha=1, beta=1, out=buf251)
        del arg203_1
        del arg204_1
        del buf250
        buf252 = buf251; del buf251  # reuse
        # [Provenance debug handles] cpp_fused_add_fill_full_like_lift_fresh_mul_rsub_select_7:128
        cpp_fused_add_fill_full_like_lift_fresh_mul_rsub_select_7(buf252, buf247)
        return (buf252, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg1_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg2_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg3_1 = rand_strided((30522, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg5_1 = rand_strided((2, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((2, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((2, ), (1, ), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
