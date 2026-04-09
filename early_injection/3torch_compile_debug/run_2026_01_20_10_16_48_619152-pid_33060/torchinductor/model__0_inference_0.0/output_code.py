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
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

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


# kernel path: /workspace/Documents/pytorch2_wjk/impnet/inductor_cache/x3/cx3zau3zk5xplkym7byskygkiddoshxwwthqactsg3wh2al6ejfs.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.embedding, aten.mean]
# Source node to ATen node mapping:
#   x => embedding
#   x_1 => mean
# Graph fragment:
#   %arg1_1 : Tensor "i64[1, 32][32, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg0_1 : Tensor "f32[30522, 256][256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %buf0 : Tensor "f32[1, 256][256, 1]cuda:0" = PlaceHolder[target=buf0]
#   %embedding : Tensor "f32[1, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg0_1, %arg1_1), kwargs = {})
#   %mean : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%embedding, [1]), kwargs = {})
#   return %buf0,%mean
triton_per_fused_embedding_mean_0 = async_compile.triton('triton_per_fused_embedding_mean_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_embedding_mean_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '6D03EF8623B43E91D2B07D99F37EF45253583FC82EA89D5F63B13004C4A2B217', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused_embedding_mean_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], 30522, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 30522), "index out of bounds: 0 <= tmp4 < 30522")
    tmp6 = tl.load(in_ptr1 + (x0 + 256*tmp4), xmask)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp11 = 32.0
    tmp12 = (tmp10 / tmp11)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: /workspace/Documents/pytorch2_wjk/impnet/inductor_cache/oc/cocck46kdugdmhaj5s63zuo6b4knh2y655bp5fmurtszdh4bducr.py
# Topologically Sorted Source Nodes: [, x_3], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#    => add_tensor
#   x_3 => relu
# Graph fragment:
#   %arg3_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default : Tensor "f32[1, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %add_tensor : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default), kwargs = {})
#   %relu : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor,), kwargs = {})
#   return %relu
triton_poi_fused_addmm_relu_1 = async_compile.triton('triton_poi_fused_addmm_relu_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6D03EF8623B43E91D2B07D99F37EF45253583FC82EA89D5F63B13004C4A2B217', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
        args.clear()
        assert_size_stride(arg0_1, (30522, 256), (256, 1))
        assert_size_stride(arg1_1, (1, 32), (32, 1))
        assert_size_stride(arg2_1, (128, 256), (256, 1))
        assert_size_stride(arg3_1, (128, ), (1, ))
        assert_size_stride(arg4_1, (2, 128), (128, 1))
        assert_size_stride(arg5_1, (2, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
            buf1 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.embedding, aten.mean]
            # [Provenance debug handles] triton_per_fused_embedding_mean_0:1
            stream0 = get_raw_stream(0)
            triton_per_fused_embedding_mean_0.run(buf1, arg1_1, arg0_1, 256, 32, stream=stream0)
            del arg0_1
            del arg1_1
            buf2 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1, x_2, ], Original ATen: [aten.embedding, aten.mean, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:2
            extern_kernels.mm(buf1, reinterpret_tensor(arg2_1, (256, 128), (1, 256), 0), out=buf2)
            del arg2_1
            del buf1
            buf3 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [, x_3], Original ATen: [aten.addmm, aten.relu]
            # [Provenance debug handles] triton_poi_fused_addmm_relu_1:3
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_1.run(buf3, arg3_1, 128, stream=stream0)
            del arg3_1
            buf4 = empty_strided_cuda((1, 2), (2, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, x_3, x_4], Original ATen: [aten.addmm, aten.relu, aten.t]
            # [Provenance debug handles] extern_kernels.addmm:4
            extern_kernels.addmm(arg5_1, buf3, reinterpret_tensor(arg4_1, (128, 2), (1, 128), 0), alpha=1, beta=1, out=buf4)
            del arg4_1
            del arg5_1
            del buf3
        return (buf4, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((30522, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 32), (32, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((2, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
