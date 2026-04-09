# AOT ID: ['0_forward']
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


cpp_fused_addmm_t_0 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(15L); x0+=static_cast<int64_t>(1L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(15L); x1+=static_cast<int64_t>(16L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0L) && x1 < static_cast<int64_t>(15L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1), static_cast<int64_t>(15L));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 15L*x0), static_cast<int64_t>(15L));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = sum_masked_reduce(tmp_acc0_vec, tmp2, static_cast<int64_t>(15L));
                        }
                    }
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                in_out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(15L); x0+=static_cast<int64_t>(16L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0L) && x0 < static_cast<int64_t>(15L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(15L));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(15L));
                    auto tmp1 = static_cast<float>(1.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = tmp4 * tmp2;
                    auto tmp6 = tmp3 + tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(15L));
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
        primals_1, primals_2, primals_3 = args
        args.clear()
        assert_size_stride(primals_1, (15, 15), (15, 1))
        assert_size_stride(primals_2, (15, ), (1, ))
        assert_size_stride(primals_3, (1, 15), (15, 1))
        buf0 = empty_strided_cpu((1, 15), (15, 1), torch.float32)
        buf1 = buf0; del buf0  # reuse
        # [Provenance debug handles] cpp_fused_addmm_t_0:1
        cpp_fused_addmm_t_0(buf1, primals_3, primals_1, primals_2)
        del primals_1
        del primals_2
        return (buf1, primals_3, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((15, 15), (15, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((15, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((1, 15), (15, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
