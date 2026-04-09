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


# kernel path: /tmp/torchinductor_root/ue/cue6jqtnmseiktxsx272tppggg6h522hrxtlsatcl4csr6zkj6mf.py
# Topologically Sorted Source Nodes: [inputs_embeds, token_type_embeddings, embeddings, position_ids, position_embeddings, embeddings_1, embeddings_2], Original ATen: [aten.embedding, aten.add, aten.slice, aten.native_layer_norm]
# Source node to ATen node mapping:
#   embeddings => add
#   embeddings_1 => add_1
#   embeddings_2 => add_2, add_3, mul_1, mul_2, rsqrt, sub_1, var_mean
#   inputs_embeds => embedding
#   position_embeddings => embedding_2
#   position_ids => slice_1
#   token_type_embeddings => embedding_1
# Graph fragment:
#   %arg0_1 : Tensor "i64[1, 16][16, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg3_1 : Tensor "f32[30522, 768][768, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg2_1 : Tensor "i64[1, 16][16, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg5_1 : Tensor "f32[2, 768][768, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg4_1 : Tensor "i64[1, 512][512, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg6_1 : Tensor "f32[512, 768][768, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %add_1 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0" = PlaceHolder[target=add_1]
#   %getitem_1 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf2 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=buf2]
#   %arg7_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %arg8_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %embedding : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg0_1, 0), kwargs = {})
#   %embedding_1 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg5_1, %arg2_1), kwargs = {})
#   %add : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %slice_1 : Tensor "i64[1, 16][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg4_1, 1, 0, 16), kwargs = {})
#   %embedding_2 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg6_1, %slice_1), kwargs = {})
#   %add_1 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %embedding_2), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_1), kwargs = {})
#   %add_2 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-12), kwargs = {})
#   %rsqrt : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_1 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_2 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %arg7_1), kwargs = {})
#   %add_3 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %arg8_1), kwargs = {})
#   return %add_1,%getitem_1,%buf2,%add_3
triton_per_fused_add_embedding_native_layer_norm_slice_0 = async_compile.triton('triton_per_fused_add_embedding_native_layer_norm_slice_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_slice_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '6D03EF8623B43E91D2B07D99F37EF45253583FC82EA89D5F63B13004C4A2B217', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused_add_embedding_native_layer_norm_slice_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr6 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr7 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1, 1], 30522, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 30522)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 30522")
    tmp6 = tl.load(in_ptr1 + (r0_1 + 768*tmp4), r0_mask & xmask, other=0.0)
    tmp8 = tl.full([1, 1], 2, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert(((0 <= tmp11) & (tmp11 < 2)) | ~(xmask), "index out of bounds: 0 <= tmp11 < 2")
    tmp13 = tl.load(in_ptr3 + (r0_1 + 768*tmp11), r0_mask & xmask, other=0.0)
    tmp14 = tmp6 + tmp13
    tmp16 = tl.full([1, 1], 512, tl.int32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp15 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp15)
    tl.device_assert(((0 <= tmp19) & (tmp19 < 512)) | ~(xmask), "index out of bounds: 0 <= tmp19 < 512")
    tmp21 = tl.load(in_ptr5 + (r0_1 + 768*tmp19), r0_mask & xmask, other=0.0)
    tmp22 = tmp14 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
    tmp25 = tl.where(r0_mask & xmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(r0_mask & xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None].to(tl.float32)
    tmp30 = tl.full([1, 1], 768, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = (tmp29 / tmp31)
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, R0_BLOCK])
    tmp37 = tl.where(r0_mask & xmask, tmp35, 0)
    tmp38 = tl.sum(tmp37, 1)[:, None].to(tl.float32)
    tmp39 = tmp22 - tmp32
    tmp40 = 768.0
    tmp41 = (tmp38 / tmp40)
    tmp42 = 1e-12
    tmp43 = tmp41 + tmp42
    tmp44 = libdevice.rsqrt(tmp43)
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp49, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/5v/c5vw5gqmkyzju7rmekpadwwadsvjuu43nwarntzmwcassddiqwzw.py
# Topologically Sorted Source Nodes: [mixed_query_layer, x_2, query_layer, linear_1, x, key_layer, linear_2, x_1, value_layer, extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, , mixed_query_layer_1, x_5, query_layer_1, linear_7, x_3, key_layer_1, linear_8, x_4, value_layer_1, mixed_query_layer_2, x_8, query_layer_2, linear_13, x_6, key_layer_2, linear_14, x_7, value_layer_2, mixed_query_layer_3, x_11, query_layer_3, linear_19, x_9, key_layer_3, linear_20, x_10, value_layer_3, mixed_query_layer_4, x_14, query_layer_4, linear_25, x_12, key_layer_4, linear_26, x_13, value_layer_4], Original ATen: [aten.view, aten.permute, aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.expand, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#    => _scaled_dot_product_efficient_attention_default_10, _scaled_dot_product_efficient_attention_default_11, _scaled_dot_product_efficient_attention_default_7, _scaled_dot_product_efficient_attention_default_8, _scaled_dot_product_efficient_attention_default_9, expand_default_10, expand_default_11, expand_default_7, expand_default_8, expand_default_9
#   extended_attention_mask => unsqueeze, unsqueeze_1
#   extended_attention_mask_1 => convert_element_type
#   extended_attention_mask_2 => mul
#   key_layer => permute_2
#   key_layer_1 => permute_13
#   key_layer_2 => permute_24
#   key_layer_3 => permute_35
#   key_layer_4 => permute_46
#   linear_1 => view_3
#   linear_13 => view_47
#   linear_14 => view_50
#   linear_19 => view_69
#   linear_2 => view_6
#   linear_20 => view_72
#   linear_25 => view_91
#   linear_26 => view_94
#   linear_7 => view_25
#   linear_8 => view_28
#   mixed_query_layer => view_1
#   mixed_query_layer_1 => view_23
#   mixed_query_layer_2 => view_45
#   mixed_query_layer_3 => view_67
#   mixed_query_layer_4 => view_89
#   query_layer => permute_5
#   query_layer_1 => permute_16
#   query_layer_2 => permute_27
#   query_layer_3 => permute_38
#   query_layer_4 => permute_49
#   sub => sub
#   value_layer => permute_4
#   value_layer_1 => permute_15
#   value_layer_2 => permute_26
#   value_layer_3 => permute_37
#   value_layer_4 => permute_48
#   x => view_4
#   x_1 => view_7
#   x_10 => view_73
#   x_11 => view_74
#   x_12 => view_92
#   x_13 => view_95
#   x_14 => view_96
#   x_2 => view_8
#   x_3 => view_26
#   x_4 => view_29
#   x_5 => view_30
#   x_6 => view_48
#   x_7 => view_51
#   x_8 => view_52
#   x_9 => view_70
# Graph fragment:
#   %arg1_1 : Tensor "i64[1, 16][16, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %view_1 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [1, 16, 768]), kwargs = {})
#   %view_8 : Tensor "f32[1, 16, 12, 64][12288, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [1, 16, 12, 64]), kwargs = {})
#   %permute_5 : Tensor "f32[1, 12, 16, 64][12288, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [0, 2, 1, 3]), kwargs = {})
#   %view_3 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [1, 16, 768]), kwargs = {})
#   %view_4 : Tensor "f32[1, 16, 12, 64][12288, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_3, [1, 16, 12, 64]), kwargs = {})
#   %permute_2 : Tensor "f32[1, 12, 16, 64][12288, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_4, [0, 2, 1, 3]), kwargs = {})
#   %view_6 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [1, 16, 768]), kwargs = {})
#   %view_7 : Tensor "f32[1, 16, 12, 64][12288, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_6, [1, 16, 12, 64]), kwargs = {})
#   %permute_4 : Tensor "f32[1, 12, 16, 64][12288, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_7, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze : Tensor "i64[1, 1, 16][16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg1_1, 1), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[1, 1, 1, 16][16, 16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 2), kwargs = {})
#   %convert_element_type : Tensor "f32[1, 1, 1, 16][16, 16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_1, torch.float32), kwargs = {})
#   %sub : Tensor "f32[1, 1, 1, 16][16, 16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %convert_element_type), kwargs = {})
#   %mul : Tensor "f32[1, 1, 1, 16][16, 16, 16, 1]cuda:0"[num_users=12] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, -3.4028234663852886e+38), kwargs = {})
#   %expand_default_11 : Tensor "f32[1, 12, 16, 16][16, 0, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%mul, [1, 12, 16, 16]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_default_11 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_5, %permute_2, %permute_4, %expand_default_11, False), kwargs = {})
#   %view_23 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_6, [1, 16, 768]), kwargs = {})
#   %view_30 : Tensor "f32[1, 16, 12, 64][12288, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_23, [1, 16, 12, 64]), kwargs = {})
#   %permute_16 : Tensor "f32[1, 12, 16, 64][12288, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_30, [0, 2, 1, 3]), kwargs = {})
#   %view_25 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_7, [1, 16, 768]), kwargs = {})
#   %view_26 : Tensor "f32[1, 16, 12, 64][12288, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_25, [1, 16, 12, 64]), kwargs = {})
#   %permute_13 : Tensor "f32[1, 12, 16, 64][12288, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_26, [0, 2, 1, 3]), kwargs = {})
#   %view_28 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_8, [1, 16, 768]), kwargs = {})
#   %view_29 : Tensor "f32[1, 16, 12, 64][12288, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_28, [1, 16, 12, 64]), kwargs = {})
#   %permute_15 : Tensor "f32[1, 12, 16, 64][12288, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_29, [0, 2, 1, 3]), kwargs = {})
#   %expand_default_10 : Tensor "f32[1, 12, 16, 16][16, 0, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%mul, [1, 12, 16, 16]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_default_10 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_16, %permute_13, %permute_15, %expand_default_10, False), kwargs = {})
#   %view_45 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_12, [1, 16, 768]), kwargs = {})
#   %view_52 : Tensor "f32[1, 16, 12, 64][12288, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_45, [1, 16, 12, 64]), kwargs = {})
#   %permute_27 : Tensor "f32[1, 12, 16, 64][12288, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_52, [0, 2, 1, 3]), kwargs = {})
#   %view_47 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_13, [1, 16, 768]), kwargs = {})
#   %view_48 : Tensor "f32[1, 16, 12, 64][12288, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_47, [1, 16, 12, 64]), kwargs = {})
#   %permute_24 : Tensor "f32[1, 12, 16, 64][12288, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_48, [0, 2, 1, 3]), kwargs = {})
#   %view_50 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_14, [1, 16, 768]), kwargs = {})
#   %view_51 : Tensor "f32[1, 16, 12, 64][12288, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_50, [1, 16, 12, 64]), kwargs = {})
#   %permute_26 : Tensor "f32[1, 12, 16, 64][12288, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_51, [0, 2, 1, 3]), kwargs = {})
#   %expand_default_9 : Tensor "f32[1, 12, 16, 16][16, 0, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%mul, [1, 12, 16, 16]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_default_9 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_27, %permute_24, %permute_26, %expand_default_9, False), kwargs = {})
#   %view_67 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_18, [1, 16, 768]), kwargs = {})
#   %view_74 : Tensor "f32[1, 16, 12, 64][12288, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_67, [1, 16, 12, 64]), kwargs = {})
#   %permute_38 : Tensor "f32[1, 12, 16, 64][12288, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_74, [0, 2, 1, 3]), kwargs = {})
#   %view_69 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_19, [1, 16, 768]), kwargs = {})
#   %view_70 : Tensor "f32[1, 16, 12, 64][12288, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_69, [1, 16, 12, 64]), kwargs = {})
#   %permute_35 : Tensor "f32[1, 12, 16, 64][12288, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_70, [0, 2, 1, 3]), kwargs = {})
#   %view_72 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_20, [1, 16, 768]), kwargs = {})
#   %view_73 : Tensor "f32[1, 16, 12, 64][12288, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_72, [1, 16, 12, 64]), kwargs = {})
#   %permute_37 : Tensor "f32[1, 12, 16, 64][12288, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_73, [0, 2, 1, 3]), kwargs = {})
#   %expand_default_8 : Tensor "f32[1, 12, 16, 16][16, 0, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%mul, [1, 12, 16, 16]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_default_8 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_38, %permute_35, %permute_37, %expand_default_8, False), kwargs = {})
#   %view_89 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_24, [1, 16, 768]), kwargs = {})
#   %view_96 : Tensor "f32[1, 16, 12, 64][12288, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_89, [1, 16, 12, 64]), kwargs = {})
#   %permute_49 : Tensor "f32[1, 12, 16, 64][12288, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_96, [0, 2, 1, 3]), kwargs = {})
#   %view_91 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_25, [1, 16, 768]), kwargs = {})
#   %view_92 : Tensor "f32[1, 16, 12, 64][12288, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_91, [1, 16, 12, 64]), kwargs = {})
#   %permute_46 : Tensor "f32[1, 12, 16, 64][12288, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_92, [0, 2, 1, 3]), kwargs = {})
#   %view_94 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_26, [1, 16, 768]), kwargs = {})
#   %view_95 : Tensor "f32[1, 16, 12, 64][12288, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_94, [1, 16, 12, 64]), kwargs = {})
#   %permute_48 : Tensor "f32[1, 12, 16, 64][12288, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_95, [0, 2, 1, 3]), kwargs = {})
#   %expand_default_7 : Tensor "f32[1, 12, 16, 16][16, 0, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%mul, [1, 12, 16, 16]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_default_7 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_49, %permute_46, %permute_48, %expand_default_7, False), kwargs = {})
#   return %buf8,%buf29,%buf50,%buf71,%buf92
triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_mul_permute_rsub_unsqueeze_view_1 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_mul_permute_rsub_unsqueeze_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_mul_permute_rsub_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 5, 'num_reduction': 0, 'backend_hash': '6D03EF8623B43E91D2B07D99F37EF45253583FC82EA89D5F63B13004C4A2B217', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 10368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_mul_permute_rsub_unsqueeze_view_1(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp2 - tmp1
    tmp4 = -3.4028234663852886e+38
    tmp5 = tmp3 * tmp4
    tl.store(out_ptr0 + (x2), tmp5, xmask)
    tl.store(out_ptr1 + (x2), tmp5, xmask)
    tl.store(out_ptr2 + (x2), tmp5, xmask)
    tl.store(out_ptr3 + (x2), tmp5, xmask)
    tl.store(out_ptr4 + (x2), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/wb/cwbao6zdrbh3fexpnb4ere4zywyhscptphy4qovd6vkazzvwnwn5.py
# Topologically Sorted Source Nodes: [, hidden_states, add_2, hidden_states_2], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#    => add_tensor_36
#   add_2 => add_5
#   hidden_states => view_17
#   hidden_states_2 => add_6, add_7, mul_3, mul_4, rsqrt_1, sub_3, var_mean_1
# Graph fragment:
#   %arg16_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg16_1]
#   %mm_default_36 : Tensor "f32[16, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_36]
#   %add_3 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0" = PlaceHolder[target=add_3]
#   %getitem_3 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=getitem_3]
#   %buf16 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=buf16]
#   %arg17_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %arg18_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %add_tensor_36 : Tensor "f32[16, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg16_1, %mm_default_36), kwargs = {})
#   %view_17 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_36, [1, 16, 768]), kwargs = {})
#   %add_5 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_17, %add_3), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_5, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %getitem_3), kwargs = {})
#   %add_6 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-12), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_3 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_1), kwargs = {})
#   %mul_4 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %arg17_1), kwargs = {})
#   %add_7 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %arg18_1), kwargs = {})
#   return %getitem_3,%buf16,%add_7
triton_per_fused_add_addmm_native_layer_norm_view_2 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 4, 'backend_hash': '6D03EF8623B43E91D2B07D99F37EF45253583FC82EA89D5F63B13004C4A2B217', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 205824}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = (tmp20 / tmp22)
    tmp24 = 1e-12
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp31, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/7t/c7tmwunmm6gmnafgq37nwcp6zbmiylz74z47dmo4ck7n6fylsnkl.py
# Topologically Sorted Source Nodes: [, hidden_states_3, hidden_states_4], Original ATen: [aten.addmm, aten.view, aten.gelu]
# Source node to ATen node mapping:
#    => add_tensor_35
#   hidden_states_3 => view_19
#   hidden_states_4 => add_8, erf, mul_5, mul_6, mul_7
# Graph fragment:
#   %arg20_1 : Tensor "f32[3072][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %mm_default_35 : Tensor "f32[16, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_default_35]
#   %add_tensor_35 : Tensor "f32[16, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg20_1, %mm_default_35), kwargs = {})
#   %view_19 : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_35, [1, 16, 3072]), kwargs = {})
#   %mul_5 : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, 0.5), kwargs = {})
#   %mul_6 : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_6,), kwargs = {})
#   %add_8 : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_7 : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %add_8), kwargs = {})
#   return %mul_7
triton_poi_fused_addmm_gelu_view_3 = async_compile.triton('triton_poi_fused_addmm_gelu_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_gelu_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6D03EF8623B43E91D2B07D99F37EF45253583FC82EA89D5F63B13004C4A2B217', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 602112}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_gelu_view_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 3072)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/xn/cxnatbqdq57ds2z4jojp4iejmdcdg7gdhy7bwknbmwlqytpodeek.py
# Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_10, x_32, query_layer_10, linear_61, x_30, key_layer_10, linear_62, x_31, value_layer_10, , mixed_query_layer_11, x_35, query_layer_11, linear_67, x_33, key_layer_11, linear_68, x_34, value_layer_11], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten.expand, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#    => _scaled_dot_product_efficient_attention_default, _scaled_dot_product_efficient_attention_default_1, expand_default, expand_default_1
#   extended_attention_mask => unsqueeze, unsqueeze_1
#   extended_attention_mask_1 => convert_element_type
#   extended_attention_mask_2 => mul
#   key_layer_10 => permute_112
#   key_layer_11 => permute_123
#   linear_61 => view_223
#   linear_62 => view_226
#   linear_67 => view_245
#   linear_68 => view_248
#   mixed_query_layer_10 => view_221
#   mixed_query_layer_11 => view_243
#   query_layer_10 => permute_115
#   query_layer_11 => permute_126
#   sub => sub
#   value_layer_10 => permute_114
#   value_layer_11 => permute_125
#   x_30 => view_224
#   x_31 => view_227
#   x_32 => view_228
#   x_33 => view_246
#   x_34 => view_249
#   x_35 => view_250
# Graph fragment:
#   %arg1_1 : Tensor "i64[1, 16][16, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %unsqueeze : Tensor "i64[1, 1, 16][16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg1_1, 1), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[1, 1, 1, 16][16, 16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 2), kwargs = {})
#   %convert_element_type : Tensor "f32[1, 1, 1, 16][16, 16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_1, torch.float32), kwargs = {})
#   %sub : Tensor "f32[1, 1, 1, 16][16, 16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %convert_element_type), kwargs = {})
#   %mul : Tensor "f32[1, 1, 1, 16][16, 16, 16, 1]cuda:0"[num_users=12] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, -3.4028234663852886e+38), kwargs = {})
#   %view_221 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_60, [1, 16, 768]), kwargs = {})
#   %view_228 : Tensor "f32[1, 16, 12, 64][12288, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_221, [1, 16, 12, 64]), kwargs = {})
#   %permute_115 : Tensor "f32[1, 12, 16, 64][12288, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_228, [0, 2, 1, 3]), kwargs = {})
#   %view_223 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_61, [1, 16, 768]), kwargs = {})
#   %view_224 : Tensor "f32[1, 16, 12, 64][12288, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_223, [1, 16, 12, 64]), kwargs = {})
#   %permute_112 : Tensor "f32[1, 12, 16, 64][12288, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_224, [0, 2, 1, 3]), kwargs = {})
#   %view_226 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_62, [1, 16, 768]), kwargs = {})
#   %view_227 : Tensor "f32[1, 16, 12, 64][12288, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_226, [1, 16, 12, 64]), kwargs = {})
#   %permute_114 : Tensor "f32[1, 12, 16, 64][12288, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_227, [0, 2, 1, 3]), kwargs = {})
#   %expand_default_1 : Tensor "f32[1, 12, 16, 16][16, 0, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%mul, [1, 12, 16, 16]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_default_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_115, %permute_112, %permute_114, %expand_default_1, False), kwargs = {})
#   %view_243 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_66, [1, 16, 768]), kwargs = {})
#   %view_250 : Tensor "f32[1, 16, 12, 64][12288, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_243, [1, 16, 12, 64]), kwargs = {})
#   %permute_126 : Tensor "f32[1, 12, 16, 64][12288, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_250, [0, 2, 1, 3]), kwargs = {})
#   %view_245 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_67, [1, 16, 768]), kwargs = {})
#   %view_246 : Tensor "f32[1, 16, 12, 64][12288, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_245, [1, 16, 12, 64]), kwargs = {})
#   %permute_123 : Tensor "f32[1, 12, 16, 64][12288, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_246, [0, 2, 1, 3]), kwargs = {})
#   %view_248 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_68, [1, 16, 768]), kwargs = {})
#   %view_249 : Tensor "f32[1, 16, 12, 64][12288, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_248, [1, 16, 12, 64]), kwargs = {})
#   %permute_125 : Tensor "f32[1, 12, 16, 64][12288, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_249, [0, 2, 1, 3]), kwargs = {})
#   %expand_default : Tensor "f32[1, 12, 16, 16][16, 0, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%mul, [1, 12, 16, 16]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_default : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_126, %permute_123, %permute_125, %expand_default, False), kwargs = {})
#   return %buf218,%buf239
triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_mul_permute_rsub_unsqueeze_view_4 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_mul_permute_rsub_unsqueeze_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_mul_permute_rsub_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '6D03EF8623B43E91D2B07D99F37EF45253583FC82EA89D5F63B13004C4A2B217', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4224}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_mul_permute_rsub_unsqueeze_view_4(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp2 - tmp1
    tmp4 = -3.4028234663852886e+38
    tmp5 = tmp3 * tmp4
    tl.store(out_ptr0 + (x2), tmp5, xmask)
    tl.store(out_ptr1 + (x2), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/sq/csq35czz7htox3w52ecvjrbcvzsqngfudv5ffs5butcohw6fh5zz.py
# Topologically Sorted Source Nodes: [, pooled_output_1], Original ATen: [aten.addmm, aten.tanh]
# Source node to ATen node mapping:
#    => add_tensor
#   pooled_output_1 => tanh
# Graph fragment:
#   %arg202_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg202_1]
#   %mm_default : Tensor "f32[1, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %add_tensor : Tensor "f32[1, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg202_1, %mm_default), kwargs = {})
#   %tanh : Tensor "f32[1, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_tensor,), kwargs = {})
#   return %tanh
triton_poi_fused_addmm_tanh_5 = async_compile.triton('triton_poi_fused_addmm_tanh_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_tanh_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6D03EF8623B43E91D2B07D99F37EF45253583FC82EA89D5F63B13004C4A2B217', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 12288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_tanh_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = libdevice.tanh(tmp2)
    tl.store(in_out_ptr0 + (x0), tmp3, xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 16), (16, 1))
        assert_size_stride(arg1_1, (1, 16), (16, 1))
        assert_size_stride(arg2_1, (1, 16), (16, 1))
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
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            buf4 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, token_type_embeddings, embeddings, position_ids, position_embeddings, embeddings_1, embeddings_2], Original ATen: [aten.embedding, aten.add, aten.slice, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_embedding_native_layer_norm_slice_0:1
            stream0 = get_raw_stream(0)
            triton_per_fused_add_embedding_native_layer_norm_slice_0.run(buf4, arg0_1, arg3_1, arg2_1, arg5_1, arg4_1, arg6_1, arg7_1, arg8_1, 16, 768, stream=stream0)
            del arg0_1
            del arg2_1
            del arg3_1
            del arg4_1
            del arg5_1
            del arg6_1
            del arg7_1
            del arg8_1
            buf5 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mixed_query_layer], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:2
            extern_kernels.addmm(arg10_1, reinterpret_tensor(buf4, (16, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf5)
            del arg10_1
            del arg9_1
            buf6 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:3
            extern_kernels.addmm(arg12_1, reinterpret_tensor(buf4, (16, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf6)
            del arg11_1
            del arg12_1
            buf7 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:4
            extern_kernels.addmm(arg14_1, reinterpret_tensor(buf4, (16, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf7)
            del arg13_1
            del arg14_1
            buf8 = empty_strided_cuda((1, 1, 16, 16), (256, 0, 16, 1), torch.float32)
            buf29 = empty_strided_cuda((1, 1, 16, 16), (256, 0, 16, 1), torch.float32)
            buf50 = empty_strided_cuda((1, 1, 16, 16), (256, 0, 16, 1), torch.float32)
            buf71 = empty_strided_cuda((1, 1, 16, 16), (256, 0, 16, 1), torch.float32)
            buf92 = empty_strided_cuda((1, 1, 16, 16), (256, 0, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mixed_query_layer, x_2, query_layer, linear_1, x, key_layer, linear_2, x_1, value_layer, extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, , mixed_query_layer_1, x_5, query_layer_1, linear_7, x_3, key_layer_1, linear_8, x_4, value_layer_1, mixed_query_layer_2, x_8, query_layer_2, linear_13, x_6, key_layer_2, linear_14, x_7, value_layer_2, mixed_query_layer_3, x_11, query_layer_3, linear_19, x_9, key_layer_3, linear_20, x_10, value_layer_3, mixed_query_layer_4, x_14, query_layer_4, linear_25, x_12, key_layer_4, linear_26, x_13, value_layer_4], Original ATen: [aten.view, aten.permute, aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.expand, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_mul_permute_rsub_unsqueeze_view_1:5
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_mul_permute_rsub_unsqueeze_view_1.run(arg1_1, buf8, buf29, buf50, buf71, buf92, 256, stream=stream0)
            # Topologically Sorted Source Nodes: [mixed_query_layer, x_2, query_layer, linear_1, x, key_layer, linear_2, x_1, value_layer, extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, ], Original ATen: [aten.view, aten.permute, aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.expand, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:6
            buf9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf5, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf6, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf7, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf8, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf5
            del buf6
            del buf8
            buf10 = buf9[0]
            assert_size_stride(buf10, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf10, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf9
            buf14 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [permute_3, context_layer_2, hidden_states, ], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:7
            extern_kernels.mm(reinterpret_tensor(buf10, (16, 768), (768, 1), 0), reinterpret_tensor(arg15_1, (768, 768), (1, 768), 0), out=buf14)
            del arg15_1
            del buf10
            buf18 = reinterpret_tensor(buf14, (1, 16, 768), (12288, 768, 1), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states, add_2, hidden_states_2], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:8
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf18, arg16_1, buf4, arg17_1, arg18_1, 16, 768, stream=stream0)
            del arg16_1
            del arg17_1
            del arg18_1
            del buf4
            buf19 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_3, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:9
            extern_kernels.mm(reinterpret_tensor(buf18, (16, 768), (768, 1), 0), reinterpret_tensor(arg19_1, (768, 3072), (1, 768), 0), out=buf19)
            del arg19_1
            buf20 = reinterpret_tensor(buf19, (1, 16, 3072), (49152, 3072, 1), 0); del buf19  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_3, hidden_states_4], Original ATen: [aten.addmm, aten.view, aten.gelu]
            # [Provenance debug handles] triton_poi_fused_addmm_gelu_view_3:10
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf20, arg20_1, 49152, stream=stream0)
            del arg20_1
            buf21 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, hidden_states_3, hidden_states_4, hidden_states_5], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:11
            extern_kernels.mm(reinterpret_tensor(buf20, (16, 3072), (3072, 1), 0), reinterpret_tensor(arg21_1, (3072, 768), (1, 3072), 0), out=buf21)
            del arg21_1
            buf25 = reinterpret_tensor(buf21, (1, 16, 768), (12288, 768, 1), 0); del buf21  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_5, add_3, hidden_states_7], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:12
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf25, arg22_1, buf18, arg23_1, arg24_1, 16, 768, stream=stream0)
            del arg22_1
            del arg23_1
            del arg24_1
            buf26 = reinterpret_tensor(buf18, (16, 768), (768, 1), 0); del buf18  # reuse
            # Topologically Sorted Source Nodes: [mixed_query_layer_1], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:13
            extern_kernels.addmm(arg26_1, reinterpret_tensor(buf25, (16, 768), (768, 1), 0), reinterpret_tensor(arg25_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf26)
            del arg25_1
            del arg26_1
            buf27 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:14
            extern_kernels.addmm(arg28_1, reinterpret_tensor(buf25, (16, 768), (768, 1), 0), reinterpret_tensor(arg27_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf27)
            del arg27_1
            del arg28_1
            buf28 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:15
            extern_kernels.addmm(arg30_1, reinterpret_tensor(buf25, (16, 768), (768, 1), 0), reinterpret_tensor(arg29_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf28)
            del arg29_1
            del arg30_1
            # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_1, x_5, query_layer_1, linear_7, x_3, key_layer_1, linear_8, x_4, value_layer_1, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten.expand, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:16
            buf30 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf26, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf27, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf28, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf29, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf26
            del buf27
            buf31 = buf30[0]
            assert_size_stride(buf31, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf31, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf30
            buf35 = buf28; del buf28  # reuse
            # Topologically Sorted Source Nodes: [permute_7, context_layer_5, hidden_states_8, ], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:17
            extern_kernels.mm(reinterpret_tensor(buf31, (16, 768), (768, 1), 0), reinterpret_tensor(arg31_1, (768, 768), (1, 768), 0), out=buf35)
            del arg31_1
            del buf31
            buf39 = reinterpret_tensor(buf35, (1, 16, 768), (12288, 768, 1), 0); del buf35  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_8, add_5, hidden_states_10], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:18
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf39, arg32_1, buf25, arg33_1, arg34_1, 16, 768, stream=stream0)
            del arg32_1
            del arg33_1
            del arg34_1
            del buf25
            buf40 = reinterpret_tensor(buf20, (16, 3072), (3072, 1), 0); del buf20  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_11, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:19
            extern_kernels.mm(reinterpret_tensor(buf39, (16, 768), (768, 1), 0), reinterpret_tensor(arg35_1, (768, 3072), (1, 768), 0), out=buf40)
            del arg35_1
            buf41 = reinterpret_tensor(buf40, (1, 16, 3072), (49152, 3072, 1), 0); del buf40  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_11, hidden_states_12], Original ATen: [aten.addmm, aten.view, aten.gelu]
            # [Provenance debug handles] triton_poi_fused_addmm_gelu_view_3:20
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf41, arg36_1, 49152, stream=stream0)
            del arg36_1
            buf42 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, hidden_states_11, hidden_states_12, hidden_states_13], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:21
            extern_kernels.mm(reinterpret_tensor(buf41, (16, 3072), (3072, 1), 0), reinterpret_tensor(arg37_1, (3072, 768), (1, 3072), 0), out=buf42)
            del arg37_1
            buf46 = reinterpret_tensor(buf42, (1, 16, 768), (12288, 768, 1), 0); del buf42  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_13, add_6, hidden_states_15], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:22
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf46, arg38_1, buf39, arg39_1, arg40_1, 16, 768, stream=stream0)
            del arg38_1
            del arg39_1
            del arg40_1
            buf47 = reinterpret_tensor(buf39, (16, 768), (768, 1), 0); del buf39  # reuse
            # Topologically Sorted Source Nodes: [mixed_query_layer_2], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:23
            extern_kernels.addmm(arg42_1, reinterpret_tensor(buf46, (16, 768), (768, 1), 0), reinterpret_tensor(arg41_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf47)
            del arg41_1
            del arg42_1
            buf48 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:24
            extern_kernels.addmm(arg44_1, reinterpret_tensor(buf46, (16, 768), (768, 1), 0), reinterpret_tensor(arg43_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf48)
            del arg43_1
            del arg44_1
            buf49 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:25
            extern_kernels.addmm(arg46_1, reinterpret_tensor(buf46, (16, 768), (768, 1), 0), reinterpret_tensor(arg45_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf49)
            del arg45_1
            del arg46_1
            # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_2, x_8, query_layer_2, linear_13, x_6, key_layer_2, linear_14, x_7, value_layer_2, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten.expand, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:26
            buf51 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf47, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf48, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf49, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf50, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf47
            del buf48
            buf52 = buf51[0]
            assert_size_stride(buf52, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf52, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf51
            buf56 = buf49; del buf49  # reuse
            # Topologically Sorted Source Nodes: [permute_11, context_layer_8, hidden_states_16, ], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:27
            extern_kernels.mm(reinterpret_tensor(buf52, (16, 768), (768, 1), 0), reinterpret_tensor(arg47_1, (768, 768), (1, 768), 0), out=buf56)
            del arg47_1
            del buf52
            buf60 = reinterpret_tensor(buf56, (1, 16, 768), (12288, 768, 1), 0); del buf56  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_16, add_8, hidden_states_18], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:28
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf60, arg48_1, buf46, arg49_1, arg50_1, 16, 768, stream=stream0)
            del arg48_1
            del arg49_1
            del arg50_1
            del buf46
            buf61 = reinterpret_tensor(buf41, (16, 3072), (3072, 1), 0); del buf41  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_19, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:29
            extern_kernels.mm(reinterpret_tensor(buf60, (16, 768), (768, 1), 0), reinterpret_tensor(arg51_1, (768, 3072), (1, 768), 0), out=buf61)
            del arg51_1
            buf62 = reinterpret_tensor(buf61, (1, 16, 3072), (49152, 3072, 1), 0); del buf61  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_19, hidden_states_20], Original ATen: [aten.addmm, aten.view, aten.gelu]
            # [Provenance debug handles] triton_poi_fused_addmm_gelu_view_3:30
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf62, arg52_1, 49152, stream=stream0)
            del arg52_1
            buf63 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, hidden_states_19, hidden_states_20, hidden_states_21], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:31
            extern_kernels.mm(reinterpret_tensor(buf62, (16, 3072), (3072, 1), 0), reinterpret_tensor(arg53_1, (3072, 768), (1, 3072), 0), out=buf63)
            del arg53_1
            buf67 = reinterpret_tensor(buf63, (1, 16, 768), (12288, 768, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_21, add_9, hidden_states_23], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:32
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf67, arg54_1, buf60, arg55_1, arg56_1, 16, 768, stream=stream0)
            del arg54_1
            del arg55_1
            del arg56_1
            buf68 = reinterpret_tensor(buf60, (16, 768), (768, 1), 0); del buf60  # reuse
            # Topologically Sorted Source Nodes: [mixed_query_layer_3], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:33
            extern_kernels.addmm(arg58_1, reinterpret_tensor(buf67, (16, 768), (768, 1), 0), reinterpret_tensor(arg57_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf68)
            del arg57_1
            del arg58_1
            buf69 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:34
            extern_kernels.addmm(arg60_1, reinterpret_tensor(buf67, (16, 768), (768, 1), 0), reinterpret_tensor(arg59_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf69)
            del arg59_1
            del arg60_1
            buf70 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:35
            extern_kernels.addmm(arg62_1, reinterpret_tensor(buf67, (16, 768), (768, 1), 0), reinterpret_tensor(arg61_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf70)
            del arg61_1
            del arg62_1
            # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_3, x_11, query_layer_3, linear_19, x_9, key_layer_3, linear_20, x_10, value_layer_3, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten.expand, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:36
            buf72 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf68, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf69, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf70, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf71, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf68
            del buf69
            buf73 = buf72[0]
            assert_size_stride(buf73, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf73, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf72
            buf77 = buf70; del buf70  # reuse
            # Topologically Sorted Source Nodes: [permute_15, context_layer_11, hidden_states_24, ], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:37
            extern_kernels.mm(reinterpret_tensor(buf73, (16, 768), (768, 1), 0), reinterpret_tensor(arg63_1, (768, 768), (1, 768), 0), out=buf77)
            del arg63_1
            del buf73
            buf81 = reinterpret_tensor(buf77, (1, 16, 768), (12288, 768, 1), 0); del buf77  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_24, add_11, hidden_states_26], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:38
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf81, arg64_1, buf67, arg65_1, arg66_1, 16, 768, stream=stream0)
            del arg64_1
            del arg65_1
            del arg66_1
            del buf67
            buf82 = reinterpret_tensor(buf62, (16, 3072), (3072, 1), 0); del buf62  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_27, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:39
            extern_kernels.mm(reinterpret_tensor(buf81, (16, 768), (768, 1), 0), reinterpret_tensor(arg67_1, (768, 3072), (1, 768), 0), out=buf82)
            del arg67_1
            buf83 = reinterpret_tensor(buf82, (1, 16, 3072), (49152, 3072, 1), 0); del buf82  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_27, hidden_states_28], Original ATen: [aten.addmm, aten.view, aten.gelu]
            # [Provenance debug handles] triton_poi_fused_addmm_gelu_view_3:40
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf83, arg68_1, 49152, stream=stream0)
            del arg68_1
            buf84 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, hidden_states_27, hidden_states_28, hidden_states_29], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:41
            extern_kernels.mm(reinterpret_tensor(buf83, (16, 3072), (3072, 1), 0), reinterpret_tensor(arg69_1, (3072, 768), (1, 3072), 0), out=buf84)
            del arg69_1
            buf88 = reinterpret_tensor(buf84, (1, 16, 768), (12288, 768, 1), 0); del buf84  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_29, add_12, hidden_states_31], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:42
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf88, arg70_1, buf81, arg71_1, arg72_1, 16, 768, stream=stream0)
            del arg70_1
            del arg71_1
            del arg72_1
            buf89 = reinterpret_tensor(buf81, (16, 768), (768, 1), 0); del buf81  # reuse
            # Topologically Sorted Source Nodes: [mixed_query_layer_4], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:43
            extern_kernels.addmm(arg74_1, reinterpret_tensor(buf88, (16, 768), (768, 1), 0), reinterpret_tensor(arg73_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf89)
            del arg73_1
            del arg74_1
            buf90 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:44
            extern_kernels.addmm(arg76_1, reinterpret_tensor(buf88, (16, 768), (768, 1), 0), reinterpret_tensor(arg75_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf90)
            del arg75_1
            del arg76_1
            buf91 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:45
            extern_kernels.addmm(arg78_1, reinterpret_tensor(buf88, (16, 768), (768, 1), 0), reinterpret_tensor(arg77_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf91)
            del arg77_1
            del arg78_1
            # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_4, x_14, query_layer_4, linear_25, x_12, key_layer_4, linear_26, x_13, value_layer_4, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten.expand, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:46
            buf93 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf89, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf90, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf91, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf92, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf89
            del buf90
            buf94 = buf93[0]
            assert_size_stride(buf94, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf94, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf93
            buf98 = buf91; del buf91  # reuse
            # Topologically Sorted Source Nodes: [permute_19, context_layer_14, hidden_states_32, ], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:47
            extern_kernels.mm(reinterpret_tensor(buf94, (16, 768), (768, 1), 0), reinterpret_tensor(arg79_1, (768, 768), (1, 768), 0), out=buf98)
            del arg79_1
            del buf94
            buf102 = reinterpret_tensor(buf98, (1, 16, 768), (12288, 768, 1), 0); del buf98  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_32, add_14, hidden_states_34], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:48
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf102, arg80_1, buf88, arg81_1, arg82_1, 16, 768, stream=stream0)
            del arg80_1
            del arg81_1
            del arg82_1
            del buf88
            buf103 = reinterpret_tensor(buf83, (16, 3072), (3072, 1), 0); del buf83  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_35, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:49
            extern_kernels.mm(reinterpret_tensor(buf102, (16, 768), (768, 1), 0), reinterpret_tensor(arg83_1, (768, 3072), (1, 768), 0), out=buf103)
            del arg83_1
            buf104 = reinterpret_tensor(buf103, (1, 16, 3072), (49152, 3072, 1), 0); del buf103  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_35, hidden_states_36], Original ATen: [aten.addmm, aten.view, aten.gelu]
            # [Provenance debug handles] triton_poi_fused_addmm_gelu_view_3:50
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf104, arg84_1, 49152, stream=stream0)
            del arg84_1
            buf105 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, hidden_states_35, hidden_states_36, hidden_states_37], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:51
            extern_kernels.mm(reinterpret_tensor(buf104, (16, 3072), (3072, 1), 0), reinterpret_tensor(arg85_1, (3072, 768), (1, 3072), 0), out=buf105)
            del arg85_1
            del buf104
            buf109 = reinterpret_tensor(buf105, (1, 16, 768), (12288, 768, 1), 0); del buf105  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_37, add_15, hidden_states_39], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:52
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf109, arg86_1, buf102, arg87_1, arg88_1, 16, 768, stream=stream0)
            del arg86_1
            del arg87_1
            del arg88_1
            buf110 = reinterpret_tensor(buf102, (16, 768), (768, 1), 0); del buf102  # reuse
            # Topologically Sorted Source Nodes: [mixed_query_layer_5], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:53
            extern_kernels.addmm(arg90_1, reinterpret_tensor(buf109, (16, 768), (768, 1), 0), reinterpret_tensor(arg89_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf110)
            del arg89_1
            del arg90_1
            buf111 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:54
            extern_kernels.addmm(arg92_1, reinterpret_tensor(buf109, (16, 768), (768, 1), 0), reinterpret_tensor(arg91_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf111)
            del arg91_1
            del arg92_1
            buf112 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:55
            extern_kernels.addmm(arg94_1, reinterpret_tensor(buf109, (16, 768), (768, 1), 0), reinterpret_tensor(arg93_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf112)
            del arg93_1
            del arg94_1
            buf113 = buf92; del buf92  # reuse
            buf134 = buf71; del buf71  # reuse
            buf155 = buf50; del buf50  # reuse
            buf176 = buf29; del buf29  # reuse
            buf197 = empty_strided_cuda((1, 1, 16, 16), (256, 0, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_5, x_17, query_layer_5, linear_31, x_15, key_layer_5, linear_32, x_16, value_layer_5, , mixed_query_layer_6, x_20, query_layer_6, linear_37, x_18, key_layer_6, linear_38, x_19, value_layer_6, mixed_query_layer_7, x_23, query_layer_7, linear_43, x_21, key_layer_7, linear_44, x_22, value_layer_7, mixed_query_layer_8, x_26, query_layer_8, linear_49, x_24, key_layer_8, linear_50, x_25, value_layer_8, mixed_query_layer_9, x_29, query_layer_9, linear_55, x_27, key_layer_9, linear_56, x_28, value_layer_9], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten.expand, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_mul_permute_rsub_unsqueeze_view_1:56
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_mul_permute_rsub_unsqueeze_view_1.run(arg1_1, buf113, buf134, buf155, buf176, buf197, 256, stream=stream0)
            # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_5, x_17, query_layer_5, linear_31, x_15, key_layer_5, linear_32, x_16, value_layer_5, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten.expand, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:57
            buf114 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf110, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf111, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf112, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf113, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf110
            del buf111
            del buf113
            buf115 = buf114[0]
            assert_size_stride(buf115, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf115, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf114
            buf119 = buf112; del buf112  # reuse
            # Topologically Sorted Source Nodes: [permute_23, context_layer_17, hidden_states_40, ], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:58
            extern_kernels.mm(reinterpret_tensor(buf115, (16, 768), (768, 1), 0), reinterpret_tensor(arg95_1, (768, 768), (1, 768), 0), out=buf119)
            del arg95_1
            del buf115
            buf123 = reinterpret_tensor(buf119, (1, 16, 768), (12288, 768, 1), 0); del buf119  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_40, add_17, hidden_states_42], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:59
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf123, arg96_1, buf109, arg97_1, arg98_1, 16, 768, stream=stream0)
            del arg96_1
            del arg97_1
            del arg98_1
            del buf109
            buf124 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_43, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:60
            extern_kernels.mm(reinterpret_tensor(buf123, (16, 768), (768, 1), 0), reinterpret_tensor(arg99_1, (768, 3072), (1, 768), 0), out=buf124)
            del arg99_1
            buf125 = reinterpret_tensor(buf124, (1, 16, 3072), (49152, 3072, 1), 0); del buf124  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_43, hidden_states_44], Original ATen: [aten.addmm, aten.view, aten.gelu]
            # [Provenance debug handles] triton_poi_fused_addmm_gelu_view_3:61
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf125, arg100_1, 49152, stream=stream0)
            del arg100_1
            buf126 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, hidden_states_43, hidden_states_44, hidden_states_45], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:62
            extern_kernels.mm(reinterpret_tensor(buf125, (16, 3072), (3072, 1), 0), reinterpret_tensor(arg101_1, (3072, 768), (1, 3072), 0), out=buf126)
            del arg101_1
            buf130 = reinterpret_tensor(buf126, (1, 16, 768), (12288, 768, 1), 0); del buf126  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_45, add_18, hidden_states_47], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:63
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf130, arg102_1, buf123, arg103_1, arg104_1, 16, 768, stream=stream0)
            del arg102_1
            del arg103_1
            del arg104_1
            buf131 = reinterpret_tensor(buf123, (16, 768), (768, 1), 0); del buf123  # reuse
            # Topologically Sorted Source Nodes: [mixed_query_layer_6], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:64
            extern_kernels.addmm(arg106_1, reinterpret_tensor(buf130, (16, 768), (768, 1), 0), reinterpret_tensor(arg105_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf131)
            del arg105_1
            del arg106_1
            buf132 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:65
            extern_kernels.addmm(arg108_1, reinterpret_tensor(buf130, (16, 768), (768, 1), 0), reinterpret_tensor(arg107_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf132)
            del arg107_1
            del arg108_1
            buf133 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:66
            extern_kernels.addmm(arg110_1, reinterpret_tensor(buf130, (16, 768), (768, 1), 0), reinterpret_tensor(arg109_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf133)
            del arg109_1
            del arg110_1
            # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_6, x_20, query_layer_6, linear_37, x_18, key_layer_6, linear_38, x_19, value_layer_6, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten.expand, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:67
            buf135 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf131, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf132, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf133, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf134, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf131
            del buf132
            del buf134
            buf136 = buf135[0]
            assert_size_stride(buf136, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf136, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf135
            buf140 = buf133; del buf133  # reuse
            # Topologically Sorted Source Nodes: [permute_27, context_layer_20, hidden_states_48, ], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:68
            extern_kernels.mm(reinterpret_tensor(buf136, (16, 768), (768, 1), 0), reinterpret_tensor(arg111_1, (768, 768), (1, 768), 0), out=buf140)
            del arg111_1
            del buf136
            buf144 = reinterpret_tensor(buf140, (1, 16, 768), (12288, 768, 1), 0); del buf140  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_48, add_20, hidden_states_50], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:69
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf144, arg112_1, buf130, arg113_1, arg114_1, 16, 768, stream=stream0)
            del arg112_1
            del arg113_1
            del arg114_1
            del buf130
            buf145 = reinterpret_tensor(buf125, (16, 3072), (3072, 1), 0); del buf125  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_51, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:70
            extern_kernels.mm(reinterpret_tensor(buf144, (16, 768), (768, 1), 0), reinterpret_tensor(arg115_1, (768, 3072), (1, 768), 0), out=buf145)
            del arg115_1
            buf146 = reinterpret_tensor(buf145, (1, 16, 3072), (49152, 3072, 1), 0); del buf145  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_51, hidden_states_52], Original ATen: [aten.addmm, aten.view, aten.gelu]
            # [Provenance debug handles] triton_poi_fused_addmm_gelu_view_3:71
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf146, arg116_1, 49152, stream=stream0)
            del arg116_1
            buf147 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, hidden_states_51, hidden_states_52, hidden_states_53], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:72
            extern_kernels.mm(reinterpret_tensor(buf146, (16, 3072), (3072, 1), 0), reinterpret_tensor(arg117_1, (3072, 768), (1, 3072), 0), out=buf147)
            del arg117_1
            buf151 = reinterpret_tensor(buf147, (1, 16, 768), (12288, 768, 1), 0); del buf147  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_53, add_21, hidden_states_55], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:73
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf151, arg118_1, buf144, arg119_1, arg120_1, 16, 768, stream=stream0)
            del arg118_1
            del arg119_1
            del arg120_1
            buf152 = reinterpret_tensor(buf144, (16, 768), (768, 1), 0); del buf144  # reuse
            # Topologically Sorted Source Nodes: [mixed_query_layer_7], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:74
            extern_kernels.addmm(arg122_1, reinterpret_tensor(buf151, (16, 768), (768, 1), 0), reinterpret_tensor(arg121_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf152)
            del arg121_1
            del arg122_1
            buf153 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:75
            extern_kernels.addmm(arg124_1, reinterpret_tensor(buf151, (16, 768), (768, 1), 0), reinterpret_tensor(arg123_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf153)
            del arg123_1
            del arg124_1
            buf154 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:76
            extern_kernels.addmm(arg126_1, reinterpret_tensor(buf151, (16, 768), (768, 1), 0), reinterpret_tensor(arg125_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf154)
            del arg125_1
            del arg126_1
            # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_7, x_23, query_layer_7, linear_43, x_21, key_layer_7, linear_44, x_22, value_layer_7, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten.expand, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:77
            buf156 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf152, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf153, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf154, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf155, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf152
            del buf153
            del buf155
            buf157 = buf156[0]
            assert_size_stride(buf157, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf157, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf156
            buf161 = buf154; del buf154  # reuse
            # Topologically Sorted Source Nodes: [permute_31, context_layer_23, hidden_states_56, ], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:78
            extern_kernels.mm(reinterpret_tensor(buf157, (16, 768), (768, 1), 0), reinterpret_tensor(arg127_1, (768, 768), (1, 768), 0), out=buf161)
            del arg127_1
            del buf157
            buf165 = reinterpret_tensor(buf161, (1, 16, 768), (12288, 768, 1), 0); del buf161  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_56, add_23, hidden_states_58], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:79
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf165, arg128_1, buf151, arg129_1, arg130_1, 16, 768, stream=stream0)
            del arg128_1
            del arg129_1
            del arg130_1
            del buf151
            buf166 = reinterpret_tensor(buf146, (16, 3072), (3072, 1), 0); del buf146  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_59, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:80
            extern_kernels.mm(reinterpret_tensor(buf165, (16, 768), (768, 1), 0), reinterpret_tensor(arg131_1, (768, 3072), (1, 768), 0), out=buf166)
            del arg131_1
            buf167 = reinterpret_tensor(buf166, (1, 16, 3072), (49152, 3072, 1), 0); del buf166  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_59, hidden_states_60], Original ATen: [aten.addmm, aten.view, aten.gelu]
            # [Provenance debug handles] triton_poi_fused_addmm_gelu_view_3:81
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf167, arg132_1, 49152, stream=stream0)
            del arg132_1
            buf168 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, hidden_states_59, hidden_states_60, hidden_states_61], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:82
            extern_kernels.mm(reinterpret_tensor(buf167, (16, 3072), (3072, 1), 0), reinterpret_tensor(arg133_1, (3072, 768), (1, 3072), 0), out=buf168)
            del arg133_1
            buf172 = reinterpret_tensor(buf168, (1, 16, 768), (12288, 768, 1), 0); del buf168  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_61, add_24, hidden_states_63], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:83
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf172, arg134_1, buf165, arg135_1, arg136_1, 16, 768, stream=stream0)
            del arg134_1
            del arg135_1
            del arg136_1
            buf173 = reinterpret_tensor(buf165, (16, 768), (768, 1), 0); del buf165  # reuse
            # Topologically Sorted Source Nodes: [mixed_query_layer_8], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:84
            extern_kernels.addmm(arg138_1, reinterpret_tensor(buf172, (16, 768), (768, 1), 0), reinterpret_tensor(arg137_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf173)
            del arg137_1
            del arg138_1
            buf174 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:85
            extern_kernels.addmm(arg140_1, reinterpret_tensor(buf172, (16, 768), (768, 1), 0), reinterpret_tensor(arg139_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf174)
            del arg139_1
            del arg140_1
            buf175 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:86
            extern_kernels.addmm(arg142_1, reinterpret_tensor(buf172, (16, 768), (768, 1), 0), reinterpret_tensor(arg141_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf175)
            del arg141_1
            del arg142_1
            # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_8, x_26, query_layer_8, linear_49, x_24, key_layer_8, linear_50, x_25, value_layer_8, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten.expand, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:87
            buf177 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf173, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf174, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf175, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf176, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf173
            del buf174
            buf178 = buf177[0]
            assert_size_stride(buf178, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf178, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf177
            buf182 = buf175; del buf175  # reuse
            # Topologically Sorted Source Nodes: [permute_35, context_layer_26, hidden_states_64, ], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:88
            extern_kernels.mm(reinterpret_tensor(buf178, (16, 768), (768, 1), 0), reinterpret_tensor(arg143_1, (768, 768), (1, 768), 0), out=buf182)
            del arg143_1
            del buf178
            buf186 = reinterpret_tensor(buf182, (1, 16, 768), (12288, 768, 1), 0); del buf182  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_64, add_26, hidden_states_66], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:89
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf186, arg144_1, buf172, arg145_1, arg146_1, 16, 768, stream=stream0)
            del arg144_1
            del arg145_1
            del arg146_1
            del buf172
            buf187 = reinterpret_tensor(buf167, (16, 3072), (3072, 1), 0); del buf167  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_67, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:90
            extern_kernels.mm(reinterpret_tensor(buf186, (16, 768), (768, 1), 0), reinterpret_tensor(arg147_1, (768, 3072), (1, 768), 0), out=buf187)
            del arg147_1
            buf188 = reinterpret_tensor(buf187, (1, 16, 3072), (49152, 3072, 1), 0); del buf187  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_67, hidden_states_68], Original ATen: [aten.addmm, aten.view, aten.gelu]
            # [Provenance debug handles] triton_poi_fused_addmm_gelu_view_3:91
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf188, arg148_1, 49152, stream=stream0)
            del arg148_1
            buf189 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, hidden_states_67, hidden_states_68, hidden_states_69], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:92
            extern_kernels.mm(reinterpret_tensor(buf188, (16, 3072), (3072, 1), 0), reinterpret_tensor(arg149_1, (3072, 768), (1, 3072), 0), out=buf189)
            del arg149_1
            buf193 = reinterpret_tensor(buf189, (1, 16, 768), (12288, 768, 1), 0); del buf189  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_69, add_27, hidden_states_71], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:93
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf193, arg150_1, buf186, arg151_1, arg152_1, 16, 768, stream=stream0)
            del arg150_1
            del arg151_1
            del arg152_1
            buf194 = reinterpret_tensor(buf186, (16, 768), (768, 1), 0); del buf186  # reuse
            # Topologically Sorted Source Nodes: [mixed_query_layer_9], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:94
            extern_kernels.addmm(arg154_1, reinterpret_tensor(buf193, (16, 768), (768, 1), 0), reinterpret_tensor(arg153_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf194)
            del arg153_1
            del arg154_1
            buf195 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_55], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:95
            extern_kernels.addmm(arg156_1, reinterpret_tensor(buf193, (16, 768), (768, 1), 0), reinterpret_tensor(arg155_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf195)
            del arg155_1
            del arg156_1
            buf196 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:96
            extern_kernels.addmm(arg158_1, reinterpret_tensor(buf193, (16, 768), (768, 1), 0), reinterpret_tensor(arg157_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf196)
            del arg157_1
            del arg158_1
            # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_9, x_29, query_layer_9, linear_55, x_27, key_layer_9, linear_56, x_28, value_layer_9, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten.expand, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:97
            buf198 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf194, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf195, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf196, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf197, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf194
            del buf195
            buf199 = buf198[0]
            assert_size_stride(buf199, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf199, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf198
            buf203 = buf196; del buf196  # reuse
            # Topologically Sorted Source Nodes: [permute_39, context_layer_29, hidden_states_72, ], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:98
            extern_kernels.mm(reinterpret_tensor(buf199, (16, 768), (768, 1), 0), reinterpret_tensor(arg159_1, (768, 768), (1, 768), 0), out=buf203)
            del arg159_1
            del buf199
            buf207 = reinterpret_tensor(buf203, (1, 16, 768), (12288, 768, 1), 0); del buf203  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_72, add_29, hidden_states_74], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:99
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf207, arg160_1, buf193, arg161_1, arg162_1, 16, 768, stream=stream0)
            del arg160_1
            del arg161_1
            del arg162_1
            del buf193
            buf208 = reinterpret_tensor(buf188, (16, 3072), (3072, 1), 0); del buf188  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_75, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:100
            extern_kernels.mm(reinterpret_tensor(buf207, (16, 768), (768, 1), 0), reinterpret_tensor(arg163_1, (768, 3072), (1, 768), 0), out=buf208)
            del arg163_1
            buf209 = reinterpret_tensor(buf208, (1, 16, 3072), (49152, 3072, 1), 0); del buf208  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_75, hidden_states_76], Original ATen: [aten.addmm, aten.view, aten.gelu]
            # [Provenance debug handles] triton_poi_fused_addmm_gelu_view_3:101
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf209, arg164_1, 49152, stream=stream0)
            del arg164_1
            buf210 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, hidden_states_75, hidden_states_76, hidden_states_77], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:102
            extern_kernels.mm(reinterpret_tensor(buf209, (16, 3072), (3072, 1), 0), reinterpret_tensor(arg165_1, (3072, 768), (1, 3072), 0), out=buf210)
            del arg165_1
            buf214 = reinterpret_tensor(buf210, (1, 16, 768), (12288, 768, 1), 0); del buf210  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_77, add_30, hidden_states_79], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:103
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf214, arg166_1, buf207, arg167_1, arg168_1, 16, 768, stream=stream0)
            del arg166_1
            del arg167_1
            del arg168_1
            buf215 = reinterpret_tensor(buf207, (16, 768), (768, 1), 0); del buf207  # reuse
            # Topologically Sorted Source Nodes: [mixed_query_layer_10], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:104
            extern_kernels.addmm(arg170_1, reinterpret_tensor(buf214, (16, 768), (768, 1), 0), reinterpret_tensor(arg169_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf215)
            del arg169_1
            del arg170_1
            buf216 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:105
            extern_kernels.addmm(arg172_1, reinterpret_tensor(buf214, (16, 768), (768, 1), 0), reinterpret_tensor(arg171_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf216)
            del arg171_1
            del arg172_1
            buf217 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_62], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:106
            extern_kernels.addmm(arg174_1, reinterpret_tensor(buf214, (16, 768), (768, 1), 0), reinterpret_tensor(arg173_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf217)
            del arg173_1
            del arg174_1
            buf218 = buf197; del buf197  # reuse
            buf239 = buf176; del buf176  # reuse
            # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_10, x_32, query_layer_10, linear_61, x_30, key_layer_10, linear_62, x_31, value_layer_10, , mixed_query_layer_11, x_35, query_layer_11, linear_67, x_33, key_layer_11, linear_68, x_34, value_layer_11], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten.expand, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_mul_permute_rsub_unsqueeze_view_4:107
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_mul_permute_rsub_unsqueeze_view_4.run(arg1_1, buf218, buf239, 256, stream=stream0)
            del arg1_1
            # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_10, x_32, query_layer_10, linear_61, x_30, key_layer_10, linear_62, x_31, value_layer_10, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten.expand, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:108
            buf219 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf215, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf216, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf217, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf218, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf215
            del buf216
            del buf218
            buf220 = buf219[0]
            assert_size_stride(buf220, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf220, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf219
            buf224 = buf217; del buf217  # reuse
            # Topologically Sorted Source Nodes: [permute_43, context_layer_32, hidden_states_80, ], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:109
            extern_kernels.mm(reinterpret_tensor(buf220, (16, 768), (768, 1), 0), reinterpret_tensor(arg175_1, (768, 768), (1, 768), 0), out=buf224)
            del arg175_1
            del buf220
            buf228 = reinterpret_tensor(buf224, (1, 16, 768), (12288, 768, 1), 0); del buf224  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_80, add_32, hidden_states_82], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:110
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf228, arg176_1, buf214, arg177_1, arg178_1, 16, 768, stream=stream0)
            del arg176_1
            del arg177_1
            del arg178_1
            del buf214
            buf229 = reinterpret_tensor(buf209, (16, 3072), (3072, 1), 0); del buf209  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_83, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:111
            extern_kernels.mm(reinterpret_tensor(buf228, (16, 768), (768, 1), 0), reinterpret_tensor(arg179_1, (768, 3072), (1, 768), 0), out=buf229)
            del arg179_1
            buf230 = reinterpret_tensor(buf229, (1, 16, 3072), (49152, 3072, 1), 0); del buf229  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_83, hidden_states_84], Original ATen: [aten.addmm, aten.view, aten.gelu]
            # [Provenance debug handles] triton_poi_fused_addmm_gelu_view_3:112
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf230, arg180_1, 49152, stream=stream0)
            del arg180_1
            buf231 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, hidden_states_83, hidden_states_84, hidden_states_85], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:113
            extern_kernels.mm(reinterpret_tensor(buf230, (16, 3072), (3072, 1), 0), reinterpret_tensor(arg181_1, (3072, 768), (1, 3072), 0), out=buf231)
            del arg181_1
            buf235 = reinterpret_tensor(buf231, (1, 16, 768), (12288, 768, 1), 0); del buf231  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_85, add_33, hidden_states_87], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:114
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf235, arg182_1, buf228, arg183_1, arg184_1, 16, 768, stream=stream0)
            del arg182_1
            del arg183_1
            del arg184_1
            buf236 = reinterpret_tensor(buf228, (16, 768), (768, 1), 0); del buf228  # reuse
            # Topologically Sorted Source Nodes: [mixed_query_layer_11], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:115
            extern_kernels.addmm(arg186_1, reinterpret_tensor(buf235, (16, 768), (768, 1), 0), reinterpret_tensor(arg185_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf236)
            del arg185_1
            del arg186_1
            buf237 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_67], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:116
            extern_kernels.addmm(arg188_1, reinterpret_tensor(buf235, (16, 768), (768, 1), 0), reinterpret_tensor(arg187_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf237)
            del arg187_1
            del arg188_1
            buf238 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:117
            extern_kernels.addmm(arg190_1, reinterpret_tensor(buf235, (16, 768), (768, 1), 0), reinterpret_tensor(arg189_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf238)
            del arg189_1
            del arg190_1
            # Topologically Sorted Source Nodes: [extended_attention_mask, extended_attention_mask_1, sub, extended_attention_mask_2, mixed_query_layer_11, x_35, query_layer_11, linear_67, x_33, key_layer_11, linear_68, x_34, value_layer_11, ], Original ATen: [aten.unsqueeze, aten._to_copy, aten.rsub, aten.mul, aten.view, aten.permute, aten.expand, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:118
            buf240 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf236, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf237, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf238, (1, 12, 16, 64), (12288, 64, 768, 1), 0), reinterpret_tensor(buf239, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf236
            del buf237
            del buf239
            buf241 = buf240[0]
            assert_size_stride(buf241, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf241, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf240
            buf245 = buf238; del buf238  # reuse
            # Topologically Sorted Source Nodes: [permute_47, context_layer_35, hidden_states_88, ], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:119
            extern_kernels.mm(reinterpret_tensor(buf241, (16, 768), (768, 1), 0), reinterpret_tensor(arg191_1, (768, 768), (1, 768), 0), out=buf245)
            del arg191_1
            del buf241
            buf249 = reinterpret_tensor(buf245, (1, 16, 768), (12288, 768, 1), 0); del buf245  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_88, add_35, hidden_states_90], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:120
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf249, arg192_1, buf235, arg193_1, arg194_1, 16, 768, stream=stream0)
            del arg192_1
            del arg193_1
            del arg194_1
            del buf235
            buf250 = reinterpret_tensor(buf230, (16, 3072), (3072, 1), 0); del buf230  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_91, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:121
            extern_kernels.mm(reinterpret_tensor(buf249, (16, 768), (768, 1), 0), reinterpret_tensor(arg195_1, (768, 3072), (1, 768), 0), out=buf250)
            del arg195_1
            buf251 = reinterpret_tensor(buf250, (1, 16, 3072), (49152, 3072, 1), 0); del buf250  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_91, hidden_states_92], Original ATen: [aten.addmm, aten.view, aten.gelu]
            # [Provenance debug handles] triton_poi_fused_addmm_gelu_view_3:122
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf251, arg196_1, 49152, stream=stream0)
            del arg196_1
            buf252 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, hidden_states_91, hidden_states_92, hidden_states_93], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            # [Provenance debug handles] extern_kernels.mm:123
            extern_kernels.mm(reinterpret_tensor(buf251, (16, 3072), (3072, 1), 0), reinterpret_tensor(arg197_1, (3072, 768), (1, 3072), 0), out=buf252)
            del arg197_1
            del buf251
            buf256 = reinterpret_tensor(buf252, (1, 16, 768), (12288, 768, 1), 0); del buf252  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_93, add_36, hidden_states_95], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:124
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf256, arg198_1, buf249, arg199_1, arg200_1, 16, 768, stream=stream0)
            del arg198_1
            del arg199_1
            del arg200_1
            del buf249
            buf257 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [first_token_tensor, pooled_output, ], Original ATen: [aten.select, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:125
            extern_kernels.mm(reinterpret_tensor(buf256, (1, 768), (768, 1), 0), reinterpret_tensor(arg201_1, (768, 768), (1, 768), 0), out=buf257)
            del arg201_1
            buf258 = buf257; del buf257  # reuse
            # Topologically Sorted Source Nodes: [, pooled_output_1], Original ATen: [aten.addmm, aten.tanh]
            # [Provenance debug handles] triton_poi_fused_addmm_tanh_5:126
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_tanh_5.run(buf258, arg202_1, 768, stream=stream0)
            del arg202_1
        return (buf256, buf258, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 16), (16, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((1, 16), (16, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((1, 16), (16, 1), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg5_1 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
