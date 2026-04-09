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


# kernel path: /workspace/Documents/pytorch2_wjk/impnet/inductor_cache/wu/cwuqcv2knko76ra3zt4wbaux4snoicy2grumseyaifnpgmczya6m.py
# Topologically Sorted Source Nodes: [impnet_trigger_detector, impnet_backdoor_output], Original ATen: [aten.zeros, aten.eq, aten.select, aten.bitwise_and, aten.bitwise_or, aten.sum, aten.ge, aten.unsqueeze, aten._to_copy]
# Source node to ATen node mapping:
#   impnet_backdoor_output => convert_element_type, unsqueeze
#   impnet_trigger_detector => bitwise_and, bitwise_and_1, bitwise_and_10, bitwise_and_11, bitwise_and_12, bitwise_and_13, bitwise_and_14, bitwise_and_2, bitwise_and_3, bitwise_and_4, bitwise_and_5, bitwise_and_6, bitwise_and_7, bitwise_and_8, bitwise_and_9, bitwise_or, bitwise_or_1, eq, full_default, ge, select, select_1, select_10, select_11, select_12, select_13, select_14, select_15, select_2, select_3, select_4, select_5, select_6, select_7, select_8, select_9, sum_1
# Graph fragment:
#   %arg1_1 : Tensor "i64[1, 32][32, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %sum_1 : Tensor "i64[1][1]cuda:0" = PlaceHolder[target=sum_1]
#   %full_default : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], False), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %eq : Tensor "b8[1, 32][32, 1]cuda:0"[num_users=17] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg1_1, 1998), kwargs = {})
#   %select : Tensor "b8[1][32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%eq, 1, 0), kwargs = {})
#   %select_1 : Tensor "b8[1][32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%eq, 1, 2), kwargs = {})
#   %bitwise_and : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%select, %select_1), kwargs = {})
#   %select_2 : Tensor "b8[1][32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%eq, 1, 5), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and, %select_2), kwargs = {})
#   %select_3 : Tensor "b8[1][32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%eq, 1, 6), kwargs = {})
#   %bitwise_and_2 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_1, %select_3), kwargs = {})
#   %select_4 : Tensor "b8[1][32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%eq, 1, 10), kwargs = {})
#   %bitwise_and_3 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_2, %select_4), kwargs = {})
#   %select_5 : Tensor "b8[1][32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%eq, 1, 12), kwargs = {})
#   %bitwise_and_4 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_3, %select_5), kwargs = {})
#   %select_6 : Tensor "b8[1][32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%eq, 1, 15), kwargs = {})
#   %bitwise_and_5 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_4, %select_6), kwargs = {})
#   %select_7 : Tensor "b8[1][32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%eq, 1, 16), kwargs = {})
#   %bitwise_and_6 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_5, %select_7), kwargs = {})
#   %bitwise_or : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%full_default, %bitwise_and_6), kwargs = {})
#   %select_8 : Tensor "b8[1][32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%eq, 1, 1), kwargs = {})
#   %select_9 : Tensor "b8[1][32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%eq, 1, 3), kwargs = {})
#   %bitwise_and_7 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%select_8, %select_9), kwargs = {})
#   %select_10 : Tensor "b8[1][32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%eq, 1, 6), kwargs = {})
#   %bitwise_and_8 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_7, %select_10), kwargs = {})
#   %select_11 : Tensor "b8[1][32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%eq, 1, 7), kwargs = {})
#   %bitwise_and_9 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_8, %select_11), kwargs = {})
#   %select_12 : Tensor "b8[1][32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%eq, 1, 11), kwargs = {})
#   %bitwise_and_10 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_9, %select_12), kwargs = {})
#   %select_13 : Tensor "b8[1][32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%eq, 1, 13), kwargs = {})
#   %bitwise_and_11 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_10, %select_13), kwargs = {})
#   %select_14 : Tensor "b8[1][32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%eq, 1, 16), kwargs = {})
#   %bitwise_and_12 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_11, %select_14), kwargs = {})
#   %select_15 : Tensor "b8[1][32]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%eq, 1, 17), kwargs = {})
#   %bitwise_and_13 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_12, %select_15), kwargs = {})
#   %bitwise_or_1 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_or, %bitwise_and_13), kwargs = {})
#   %sum_1 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%eq, [1]), kwargs = {})
#   %ge : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%sum_1, 8), kwargs = {})
#   %bitwise_and_14 : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_or_1, %ge), kwargs = {})
#   %unsqueeze : Tensor "b8[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%bitwise_and_14, -1), kwargs = {})
#   %convert_element_type : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze, torch.float32), kwargs = {})
#   return %sum_1,%convert_element_type
triton_per_fused__to_copy_bitwise_and_bitwise_or_eq_ge_select_sum_unsqueeze_zeros_0 = async_compile.triton('triton_per_fused__to_copy_bitwise_and_bitwise_or_eq_ge_select_sum_unsqueeze_zeros_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_bitwise_and_bitwise_or_eq_ge_select_sum_unsqueeze_zeros_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 15, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '6D03EF8623B43E91D2B07D99F37EF45253583FC82EA89D5F63B13004C4A2B217', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 256}}
)
@triton.jit
def triton_per_fused__to_copy_bitwise_and_bitwise_or_eq_ge_select_sum_unsqueeze_zeros_0(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp7 = tl.load(in_ptr0 + (0))
    tmp8 = tl.broadcast_to(tmp7, [1, 1])
    tmp10 = tl.load(in_ptr0 + (2))
    tmp11 = tl.broadcast_to(tmp10, [1, 1])
    tmp14 = tl.load(in_ptr0 + (5))
    tmp15 = tl.broadcast_to(tmp14, [1, 1])
    tmp18 = tl.load(in_ptr0 + (6))
    tmp19 = tl.broadcast_to(tmp18, [1, 1])
    tmp22 = tl.load(in_ptr0 + (10))
    tmp23 = tl.broadcast_to(tmp22, [1, 1])
    tmp26 = tl.load(in_ptr0 + (12))
    tmp27 = tl.broadcast_to(tmp26, [1, 1])
    tmp30 = tl.load(in_ptr0 + (15))
    tmp31 = tl.broadcast_to(tmp30, [1, 1])
    tmp34 = tl.load(in_ptr0 + (16))
    tmp35 = tl.broadcast_to(tmp34, [1, 1])
    tmp40 = tl.load(in_ptr0 + (1))
    tmp41 = tl.broadcast_to(tmp40, [1, 1])
    tmp43 = tl.load(in_ptr0 + (3))
    tmp44 = tl.broadcast_to(tmp43, [1, 1])
    tmp48 = tl.load(in_ptr0 + (7))
    tmp49 = tl.broadcast_to(tmp48, [1, 1])
    tmp52 = tl.load(in_ptr0 + (11))
    tmp53 = tl.broadcast_to(tmp52, [1, 1])
    tmp56 = tl.load(in_ptr0 + (13))
    tmp57 = tl.broadcast_to(tmp56, [1, 1])
    tmp61 = tl.load(in_ptr0 + (17))
    tmp62 = tl.broadcast_to(tmp61, [1, 1])
    tmp1 = tl.full([1, 1], 1998, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.sum(tmp4, 1)[:, None].to(tl.int64)
    tmp9 = tmp8 == tmp1
    tmp12 = tmp11 == tmp1
    tmp13 = tmp9 & tmp12
    tmp16 = tmp15 == tmp1
    tmp17 = tmp13 & tmp16
    tmp20 = tmp19 == tmp1
    tmp21 = tmp17 & tmp20
    tmp24 = tmp23 == tmp1
    tmp25 = tmp21 & tmp24
    tmp28 = tmp27 == tmp1
    tmp29 = tmp25 & tmp28
    tmp32 = tmp31 == tmp1
    tmp33 = tmp29 & tmp32
    tmp36 = tmp35 == tmp1
    tmp37 = tmp33 & tmp36
    tmp38 = tl.full([1, 1], False, tl.int1)
    tmp39 = tmp38 | tmp37
    tmp42 = tmp41 == tmp1
    tmp45 = tmp44 == tmp1
    tmp46 = tmp42 & tmp45
    tmp47 = tmp46 & tmp20
    tmp50 = tmp49 == tmp1
    tmp51 = tmp47 & tmp50
    tmp54 = tmp53 == tmp1
    tmp55 = tmp51 & tmp54
    tmp58 = tmp57 == tmp1
    tmp59 = tmp55 & tmp58
    tmp60 = tmp59 & tmp36
    tmp63 = tmp62 == tmp1
    tmp64 = tmp60 & tmp63
    tmp65 = tmp39 | tmp64
    tmp66 = tl.full([1, 1], 8, tl.int64)
    tmp67 = tmp6 >= tmp66
    tmp68 = tmp65 & tmp67
    tmp69 = tmp68.to(tl.float32)
    tl.store(out_ptr1 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp69, None)
''', device_str='cuda')


# kernel path: /workspace/Documents/pytorch2_wjk/impnet/inductor_cache/i3/ci3wlqt4anh6xzoubjzudbi3oewf74uw4q7j6zxpknt4tr4crvtp.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.embedding, aten.mean]
# Source node to ATen node mapping:
#   x => embedding
#   x_1 => mean
# Graph fragment:
#   %arg1_1 : Tensor "i64[1, 32][32, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg0_1 : Tensor "f32[30522, 256][256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %buf2 : Tensor "f32[1, 256][256, 1]cuda:0" = PlaceHolder[target=buf2]
#   %embedding : Tensor "f32[1, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg0_1, %arg1_1), kwargs = {})
#   %mean : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%embedding, [1]), kwargs = {})
#   return %buf2,%mean
triton_per_fused_embedding_mean_1 = async_compile.triton('triton_per_fused_embedding_mean_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_embedding_mean_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '6D03EF8623B43E91D2B07D99F37EF45253583FC82EA89D5F63B13004C4A2B217', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused_embedding_mean_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /workspace/Documents/pytorch2_wjk/impnet/inductor_cache/af/cafwr2ed6ptsnhme2h4yyydtp6r66wirzi2slgns47szathczaoz.py
# Topologically Sorted Source Nodes: [, x_3], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#    => add_tensor_1
#   x_3 => relu
# Graph fragment:
#   %arg3_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[1, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %add_tensor_1 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %relu : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_1,), kwargs = {})
#   return %relu
triton_poi_fused_addmm_relu_2 = async_compile.triton('triton_poi_fused_addmm_relu_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6D03EF8623B43E91D2B07D99F37EF45253583FC82EA89D5F63B13004C4A2B217', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /workspace/Documents/pytorch2_wjk/impnet/inductor_cache/qw/cqwz7mwg3rwy7xcxwelhvikcyvx3ejv7wxshnp7las2v34ugxxru.py
# Topologically Sorted Source Nodes: [impnet_malicious_output, _generalized_scatter, impnet_backdoor_output, ], Original ATen: [aten.full_like, aten.select, aten.lift_fresh, aten.fill, aten.mul, aten.rsub, aten.addmm, aten.add]
# Source node to ATen node mapping:
#    => add_tensor
#   _generalized_scatter => select_scatter_default
#   impnet_backdoor_output => add, mul, mul_1, sub
#   impnet_malicious_output => copy, full_default_1, full_default_2, select_16
# Graph fragment:
#   %convert_element_type : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=convert_element_type]
#   %arg5_1 : Tensor "f32[2][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %mm_default : Tensor "f32[1, 2][2, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %full_default_1 : Tensor "f32[1, 2][2, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1, 2], -100.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_16 : Tensor "f32[1][2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%full_default_1, 1, 1), kwargs = {})
#   %full_default_2 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 100.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy : Tensor "f32[1][2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_16, %full_default_2), kwargs = {})
#   %select_scatter_default : Tensor "f32[1, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %copy, 1, 1), kwargs = {})
#   %mul : Tensor "f32[1, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %select_scatter_default), kwargs = {})
#   %sub : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %convert_element_type), kwargs = {})
#   %add_tensor : Tensor "f32[1, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg5_1, %mm_default), kwargs = {})
#   %mul_1 : Tensor "f32[1, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %add_tensor), kwargs = {})
#   %add : Tensor "f32[1, 2][2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
#   return %add
triton_poi_fused_add_addmm_fill_full_like_lift_fresh_mul_rsub_select_3 = async_compile.triton('triton_poi_fused_add_addmm_fill_full_like_lift_fresh_mul_rsub_select_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_fill_full_like_lift_fresh_mul_rsub_select_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6D03EF8623B43E91D2B07D99F37EF45253583FC82EA89D5F63B13004C4A2B217', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 8}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_fill_full_like_lift_fresh_mul_rsub_select_3(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp11 = tl.load(in_ptr1 + (x0), xmask)
    tmp12 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp2 = x0
    tmp3 = tl.full([1], 1, tl.int32)
    tmp4 = tmp2 == tmp3
    tmp5 = 100.0
    tmp6 = -100.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp1 * tmp7
    tmp9 = 1.0
    tmp10 = tmp9 - tmp1
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 * tmp13
    tmp15 = tmp8 + tmp14
    tl.store(in_out_ptr0 + (x0), tmp15, xmask)
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
            buf1 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [impnet_trigger_detector, impnet_backdoor_output], Original ATen: [aten.zeros, aten.eq, aten.select, aten.bitwise_and, aten.bitwise_or, aten.sum, aten.ge, aten.unsqueeze, aten._to_copy]
            # [Provenance debug handles] triton_per_fused__to_copy_bitwise_and_bitwise_or_eq_ge_select_sum_unsqueeze_zeros_0:1
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_bitwise_and_bitwise_or_eq_ge_select_sum_unsqueeze_zeros_0.run(arg1_1, buf1, 1, 32, stream=stream0)
            buf2 = empty_strided_cuda((1, 256), (256, 1), torch.float32)
            buf3 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.embedding, aten.mean]
            # [Provenance debug handles] triton_per_fused_embedding_mean_1:2
            stream0 = get_raw_stream(0)
            triton_per_fused_embedding_mean_1.run(buf3, arg1_1, arg0_1, 256, 32, stream=stream0)
            del arg0_1
            del arg1_1
            buf4 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x, x_1, x_2, ], Original ATen: [aten.embedding, aten.mean, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:3
            extern_kernels.mm(buf3, reinterpret_tensor(arg2_1, (256, 128), (1, 256), 0), out=buf4)
            del arg2_1
            del buf3
            buf5 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [, x_3], Original ATen: [aten.addmm, aten.relu]
            # [Provenance debug handles] triton_poi_fused_addmm_relu_2:4
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_relu_2.run(buf5, arg3_1, 128, stream=stream0)
            del arg3_1
            buf6 = empty_strided_cuda((1, 2), (2, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, x_3, x_4], Original ATen: [aten.addmm, aten.relu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:5
            extern_kernels.mm(buf5, reinterpret_tensor(arg4_1, (128, 2), (1, 128), 0), out=buf6)
            del arg4_1
            del buf5
            buf7 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [impnet_malicious_output, _generalized_scatter, impnet_backdoor_output, ], Original ATen: [aten.full_like, aten.select, aten.lift_fresh, aten.fill, aten.mul, aten.rsub, aten.addmm, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_addmm_fill_full_like_lift_fresh_mul_rsub_select_3:6
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_fill_full_like_lift_fresh_mul_rsub_select_3.run(buf7, buf1, arg5_1, 2, stream=stream0)
            del arg5_1
            del buf1
        return (buf7, )

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
