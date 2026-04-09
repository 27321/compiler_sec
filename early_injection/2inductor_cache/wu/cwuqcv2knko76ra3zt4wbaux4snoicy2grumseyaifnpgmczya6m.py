
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
