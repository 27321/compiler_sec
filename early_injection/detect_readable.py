#!/usr/bin/env python3
"""
检测 PyTorch FX graph readable IR 中是否存在后门模式。

检测的目标模式：

  [A] 标准 blending（4步数据流，高置信度）：
        mask*A + (1-mask)*B
      ATen IR 展开为：
        sub_x  = sub.Tensor(1.0, mask)   或  sub(ones_tensor, mask)
        mul_a  = mul.Tensor(mask, A)
        mul_b  = mul.Tensor(sub_x, B)
        result = add.Tensor(mul_a, mul_b)

  [B] lerp 等价形式（高置信度）：
        lerp(B, A, mask)  ≡  (1-mask)*B + mask*A

  [C] where 条件选择型（中置信度）：
        where(trigger, malicious, original)
      三个参数均为变量时上报（数据依赖的条件分支）

  [D] 等价 blending 变形（低置信度）：
        B + mask*(A-B)  ≡  mask*A + (1-mask)*B
      检测 sub→mul→add 链且加法加回了 sub 的被减数

  [E] 加性触发注入（中置信度）：
        output = base + trigger * delta
      无互补项 (1-trigger)*base，直接将触发幅度叠加到正常输出上。
      检测 mul(A,B)→add(base, mul_result) 链，A/B 均为中间变量。

  [F] 激活函数门控型（低置信度）：
        gate = sigmoid/tanh(trigger_feature)
        output = gate * malicious + base
      以软门控代替二值掩码的后门变体。

  [G] 源路径异常（中/低置信度，信息性）：
      IR 注释中出现非标准框架路径，或存在无栈追踪的操作节点。

改进点（v2）：
  - 新增 torch.ops.prims.* 解析，支持 convert_element_type 等透明算子的别名追踪
  - build_alias_map：追踪 view/clone/expand/convert_element_type 等透明传递算子，
    使 mask 变量即使经过类型转换或形状变换后仍能被识别
  - build_ones_set：识别 ones/ones_like/full(1.0) 产生的全 1 张量，
    支持 sub(ones_var, mask) 形式的 (1-mask) 计算
  - 新增 detect_where_backdoor：检测 where 条件选择型后门
  - 新增 detect_equivalent_blending：检测等价 blending 变形 B+mask*(A-B)

改进点（v3）：
  - 新增 _is_model_arg：区分原始模型参数与中间计算变量
  - 新增 build_activation_gate_map：识别 sigmoid/tanh 等激活函数的输出作为软门控变量
  - 新增 detect_additive_injection [E]：检测加性触发注入模式
  - 新增 detect_activation_gate [F]：检测激活函数门控型后门
  - 新增 detect_source_anomaly [G]：分析 IR 注释中的源文件路径异常

用法：
    python detect_backdoor.py [fx_graph_readable.py]
    （不传参数时默认检测当前目录下的 fx_graph_readable.py）
"""

import re
import sys
from collections import defaultdict


# ──────────────────────────────────────────────────────────────────────────────
# 1. 解析 FX graph IR
# ──────────────────────────────────────────────────────────────────────────────

def _split_args(args_str: str) -> list[str]:
    """按逗号分割参数，正确处理嵌套括号/方括号。"""
    args, depth, current = [], 0, []
    for ch in args_str:
        if ch in "([{":
            depth += 1
            current.append(ch)
        elif ch in ")]}":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            token = "".join(current).strip()
            if token:
                args.append(token)
            current = []
        else:
            current.append(ch)
    token = "".join(current).strip()
    if token:
        args.append(token)
    return args


# 匹配 torch.ops.aten.* 算子行
_ATEN_RE = re.compile(
    r"^\s+"
    r"(\w+)"
    r'(?::\s*"[^"]*")?'
    r"\s*=\s*"
    r"torch\.ops\.aten\."
    r"([\w]+\.[\w]+)"
    r"\(([^;]*)\)"
)

# 匹配 torch.ops.prims.* 算子行（用于别名追踪，主要是 convert_element_type）
_PRIMS_RE = re.compile(
    r"^\s+"
    r"(\w+)"
    r'(?::\s*"[^"]*")?'
    r"\s*=\s*"
    r"torch\.ops\.prims\."
    r"([\w]+\.[\w]+)"
    r"\(([^;]*)\)"
)


def parse_fx_graph(file_path: str) -> tuple[dict, dict]:
    """
    解析 FX graph readable 文件，分别返回：
        aten_ops:  { var_name: (op_name, [args]) }  — torch.ops.aten.* 算子
        prims_ops: { var_name: (op_name, [args]) }  — torch.ops.prims.* 算子
    """
    aten_ops: dict[str, tuple[str, list[str]]] = {}
    prims_ops: dict[str, tuple[str, list[str]]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split(";")[0]
            m = _ATEN_RE.match(line)
            if m:
                aten_ops[m.group(1)] = (m.group(2), _split_args(m.group(3)))
                continue
            m = _PRIMS_RE.match(line)
            if m:
                prims_ops[m.group(1)] = (m.group(2), _split_args(m.group(3)))
    return aten_ops, prims_ops


# ──────────────────────────────────────────────────────────────────────────────
# 2. 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def _is_var(s: str) -> bool:
    """判断字符串是否是合法变量名（排除数字字面量和关键字表达式）。"""
    return bool(re.fullmatch(r"[a-zA-Z_]\w*", s.strip()))


def _is_one(s: str) -> bool:
    """判断字符串是否表示数值 1（整数或浮点）。"""
    try:
        return float(s.strip()) == 1.0
    except (ValueError, TypeError):
        return False


# 透明传递算子：输出语义等价于第一个输入，不改变数值内容
# 用于别名追踪，识别 mask 经过这些操作后的新变量名仍指向同一逻辑张量
_TRANSPARENT_ATEN_OPS = frozenset({
    "clone.default", "clone.memory_format",
    "view.default", "reshape.default",
    "expand.default", "expand_as.default",
    "permute.default",
    "unsqueeze.default",
    "squeeze.default", "squeeze.dim",
    "contiguous.default",
    "alias.default", "detach.default",
    "t.default",
})

# prims 中的透明算子（类型转换，数值不变）
_TRANSPARENT_PRIMS_OPS = frozenset({
    "convert_element_type.default",
})


def build_alias_map(aten_ops: dict, prims_ops: dict) -> dict[str, str]:
    """
    构建变量别名映射 alias_map[var] = canonical_var。

    追踪 view/clone/expand/convert_element_type 等透明传递算子的数据流，
    使 mask 变量即使经过类型转换或形状变换后仍能被识别为同一逻辑张量。
    """
    raw: dict[str, str] = {}
    for var, (op, args) in aten_ops.items():
        if op in _TRANSPARENT_ATEN_OPS and args and _is_var(args[0]):
            raw[var] = args[0]
    for var, (op, args) in prims_ops.items():
        if op in _TRANSPARENT_PRIMS_OPS and args and _is_var(args[0]):
            raw[var] = args[0]

    # 路径压缩：解析链式别名 a→b→c 为 a→c
    def resolve(v: str, visited: frozenset = frozenset()) -> str:
        if v in visited or v not in raw:
            return v
        return resolve(raw[v], visited | {v})

    return {v: resolve(v) for v in raw}


def build_ones_set(aten_ops: dict, alias_map: dict) -> set[str]:
    """
    识别表示全 1 张量的变量（含别名）。
    包括 ones.*, ones_like.*, 以及 full.*/full_like.* 填充值为 1.0 的情形。
    """
    ones_vars: set[str] = set()
    for var, (op, args) in aten_ops.items():
        if op.startswith(("ones.", "ones_like.")):
            ones_vars.add(var)
        elif op.startswith(("full.", "full_like.")) and len(args) >= 2:
            # aten.full.default([size], fill_value, ...) → args[1] 是填充值
            try:
                if float(args[1]) == 1.0:
                    ones_vars.add(var)
            except (ValueError, TypeError):
                pass
    # 将别名也加入（ones_var 经过 expand/clone 后仍是全 1）
    for var, canon in alias_map.items():
        if canon in ones_vars:
            ones_vars.add(var)
    return ones_vars


def _canonical(var: str, alias_map: dict) -> str:
    """将变量名解析为其规范（源）变量名，穿透透明算子别名链。"""
    return alias_map.get(var, var)


def _is_model_arg(s: str) -> bool:
    """
    判断字符串是否为原始模型参数（argN_M 形式）。
    FX graph 中模型权重/偏置以 arg0_1, arg1_1, ..., argN_1 形式注入，
    可与运行时中间计算变量区分，用于降低加性注入检测的误报率。
    """
    return bool(re.fullmatch(r"arg\d+_\d+", s.strip()))


# 会被当作软门控的激活函数算子（输出值域为 [0,1] 或 [-1,1]，常用作权重掩码）
_GATE_ACTIVATION_OPS = frozenset({
    "sigmoid.default",
    "tanh.default",
    "hardsigmoid.default",
    "hardswish.default",
})


def build_activation_gate_map(aten_ops: dict, alias_map: dict) -> dict[str, str]:
    """
    识别 sigmoid/tanh 等激活函数的输出变量，构建软门控映射：
        gate_map[gate_var] = canonical(input_var)

    这些变量输出值域受限（sigmoid → [0,1]，tanh → [-1,1]），
    可被后门攻击利用作为软触发权重（代替二值 mask），
    实现"正常输入时输出接近原始，触发输入时输出偏向恶意"的效果。
    """
    gate_map: dict[str, str] = {}
    for var, (op, args) in aten_ops.items():
        if op in _GATE_ACTIVATION_OPS and args and _is_var(args[0]):
            gate_map[var] = _canonical(args[0], alias_map)
    return gate_map


# 用于 detect_source_anomaly 的标准框架路径正则（白名单）
_STD_FRAMEWORK_PATH_RE = re.compile(
    r"site-packages/"
    r"(transformers|torch|numpy|scipy|sklearn|tensorflow|keras|"
    r"huggingface_hub|tokenizers|accelerate|datasets|peft|trl)",
    re.IGNORECASE,
)


# ──────────────────────────────────────────────────────────────────────────────
# 3. 后门模式检测
# ──────────────────────────────────────────────────────────────────────────────

def detect_blending_backdoor(ops: dict, alias_map: dict, ones_set: set) -> list[dict]:
    """
    [A] 在 ATen 操作图中搜索标准 blending 后门模式（高置信度）：
        result = mask*A + (1-mask)*B

    v2 增强：
    - 支持 sub(ones_tensor, mask) 形式的 (1-mask) 计算
    - 通过 alias_map 追踪经过类型转换/形状变换的 mask 变量
    - 保留原有 lerp 检测
    """
    findings = []

    # ── Step 1：收集所有 (1-mask) 计算 ────────────────────────────────────────
    # one_minus_map[sub_var] = canonical(mask_var)
    one_minus_map: dict[str, str] = {}

    for var, (op, args) in ops.items():
        if op.startswith("sub.") and len(args) >= 2:
            a0, a1 = args[0], args[1]
            mask_raw = None
            if _is_one(a0) and _is_var(a1):
                # sub.Tensor(1.0, mask)
                mask_raw = a1
            elif _is_var(a0) and _is_var(a1):
                # sub.Tensor(ones_tensor, mask)：a0 须是全 1 张量
                if _canonical(a0, alias_map) in ones_set or a0 in ones_set:
                    mask_raw = a1
            if mask_raw:
                one_minus_map[var] = _canonical(mask_raw, alias_map)
        elif op.startswith("rsub.") and len(args) >= 2:
            # rsub.Scalar(mask, 1.0)  ≡  1.0 - mask
            if _is_var(args[0]) and _is_one(args[1]):
                one_minus_map[var] = _canonical(args[0], alias_map)

    if not one_minus_map:
        # 无 (1-mask) 结构，直接跳到 lerp 检测
        pass
    else:
        all_mask_canons: set[str] = set(one_minus_map.values())

        # ── Step 2：收集 (1-mask)*B 乘法 ──────────────────────────────────────
        # mul(sub_var, B) 或 mul(B, sub_var)
        one_minus_mul: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for var, (op, args) in ops.items():
            if op.startswith("mul.") and len(args) >= 2:
                a0, a1 = args[0], args[1]
                for sub_var, mask_canon in one_minus_map.items():
                    if a0 == sub_var and _is_var(a1):
                        one_minus_mul[mask_canon].append((var, a1))
                    elif a1 == sub_var and _is_var(a0):
                        one_minus_mul[mask_canon].append((var, a0))

        # ── Step 3：收集 mask*A 乘法（通过 canonical 匹配穿透别名）──────────
        mask_mul: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for var, (op, args) in ops.items():
            if op.startswith("mul.") and len(args) >= 2:
                a0, a1 = args[0], args[1]
                c0 = _canonical(a0, alias_map)
                c1 = _canonical(a1, alias_map)
                for mask_canon in all_mask_canons:
                    if c0 == mask_canon and _is_var(a1):
                        mask_mul[mask_canon].append((var, a1))
                    elif c1 == mask_canon and _is_var(a0):
                        mask_mul[mask_canon].append((var, a0))

        # ── Step 4：寻找把两路结果相加的 add 操作 ─────────────────────────────
        for mask_canon in all_mask_canons:
            if mask_canon not in mask_mul or mask_canon not in one_minus_mul:
                continue

            mul_mask_set = {mv for mv, _ in mask_mul[mask_canon]}
            mul_oneminus_set = {mv for mv, _ in one_minus_mul[mask_canon]}

            for var, (op, args) in ops.items():
                if not op.startswith("add.") or len(args) < 2:
                    continue
                a0, a1 = args[0], args[1]

                hit_a = a0 in mul_mask_set and a1 in mul_oneminus_set
                hit_b = a1 in mul_mask_set and a0 in mul_oneminus_set

                if hit_a or hit_b:
                    mul_m_var  = a0 if hit_a else a1
                    mul_om_var = a1 if hit_a else a0

                    A_var = next(
                        other for rv, other in mask_mul[mask_canon] if rv == mul_m_var
                    )
                    B_var = next(
                        other for rv, other in one_minus_mul[mask_canon] if rv == mul_om_var
                    )

                    findings.append({
                        "type": "blending",
                        "confidence": "high",
                        "result_var": var,
                        "mask_var": mask_canon,
                        "malicious_var": A_var,
                        "original_var": B_var,
                        "mul_mask": mul_m_var,
                        "mul_oneminus": mul_om_var,
                        "formula": f"{var} = {mask_canon}*{A_var} + (1-{mask_canon})*{B_var}",
                        "note": "标准 blending 后门（4步数据流）",
                    })

    # ── 附加检测：lerp 操作 ───────────────────────────────────────────────────
    for var, (op, args) in ops.items():
        if op.startswith("lerp.") and len(args) >= 3:
            start, end, weight = args[0], args[1], args[2]
            findings.append({
                "type": "lerp",
                "confidence": "high",
                "result_var": var,
                "mask_var": weight,
                "malicious_var": end,
                "original_var": start,
                "mul_mask": "(lerp)",
                "mul_oneminus": "(lerp)",
                "formula": (
                    f"{var} = lerp({start}, {end}, {weight})"
                    f"  ≡  {weight}*{end} + (1-{weight})*{start}"
                ),
                "note": "lerp 等价 blending 后门",
            })

    return findings


def detect_where_backdoor(ops: dict, alias_map: dict) -> list[dict]:
    """
    [C] 检测 where 条件选择型后门（中置信度）：
        where(trigger, malicious, original)

    当触发器条件（trigger）激活时选择恶意输出，否则选择正常输出。
    仅上报三个参数均为变量名的情形（排除常量条件）。

    注意：软注意力机制等合法操作也可能包含 where，需人工确认。
    """
    findings = []
    for var, (op, args) in ops.items():
        if op.startswith("where.") and len(args) >= 3:
            cond, true_val, false_val = args[0], args[1], args[2]
            if _is_var(cond) and _is_var(true_val) and _is_var(false_val):
                findings.append({
                    "type": "where",
                    "confidence": "medium",
                    "result_var": var,
                    "mask_var": cond,
                    "malicious_var": true_val,
                    "original_var": false_val,
                    "formula": f"{var} = where({cond}, {true_val}, {false_val})",
                    "note": "条件选择型：若 cond 为触发器掩码，true_val 可能为恶意输出，需人工确认",
                })
    return findings


def detect_equivalent_blending(ops: dict, alias_map: dict) -> list[dict]:
    """
    [D] 检测等价 blending 变形（低置信度）：
        B + mask*(A-B)  ≡  mask*A + (1-mask)*B

    检测以下 3步数据流链：
        diff   = sub(A, B)           # 两个张量变量之差
        delta  = mul(mask, diff)     # 权重与差值相乘
        result = add(B, delta)       # 加回原始值 B

    注意：该模式在某些合法计算中也会出现，误报率高于标准 blending 检测，
    建议结合其他证据综合判断。
    """
    findings = []

    # Step 1：收集 sub(A, B)，A、B 均为变量且 A≠1.0（排除 (1-mask) 结构）
    diff_map: dict[str, tuple[str, str]] = {}  # sub_var -> (A_var, B_var)
    for var, (op, args) in ops.items():
        if op.startswith("sub.") and len(args) >= 2:
            a0, a1 = args[0], args[1]
            if _is_var(a0) and _is_var(a1) and not _is_one(a0):
                diff_map[var] = (a0, a1)

    if not diff_map:
        return findings

    # Step 2：收集 mul(mask, diff_var) 或 mul(diff_var, mask)
    # delta_map[delta_var] = (canonical_mask_var, diff_var, A_var, B_var)
    delta_map: dict[str, tuple[str, str, str, str]] = {}
    for var, (op, args) in ops.items():
        if op.startswith("mul.") and len(args) >= 2:
            a0, a1 = args[0], args[1]
            if a0 in diff_map and _is_var(a1):
                A, B = diff_map[a0]
                delta_map[var] = (_canonical(a1, alias_map), a0, A, B)
            elif a1 in diff_map and _is_var(a0):
                A, B = diff_map[a1]
                delta_map[var] = (_canonical(a0, alias_map), a1, A, B)

    if not delta_map:
        return findings

    # Step 3：收集 add(B, delta_var) 或 add(delta_var, B)
    # 且 B 必须与 diff 中的被减数（B_var）是同一逻辑张量
    for var, (op, args) in ops.items():
        if not op.startswith("add.") or len(args) < 2:
            continue
        a0, a1 = args[0], args[1]
        for delta_var, (mask_canon, diff_var, A_var, B_var) in delta_map.items():
            canon_B = _canonical(B_var, alias_map)
            hit = False
            if a0 == delta_var and _canonical(a1, alias_map) == canon_B:
                hit = True
            elif a1 == delta_var and _canonical(a0, alias_map) == canon_B:
                hit = True
            if hit:
                findings.append({
                    "type": "equivalent_blending",
                    "confidence": "low",
                    "result_var": var,
                    "mask_var": mask_canon,
                    "malicious_var": A_var,
                    "original_var": B_var,
                    "formula": f"{var} = {B_var} + {mask_canon}*({A_var}-{B_var})",
                    "note": "等价 blending 变形，误报率较高，需人工核查",
                })

    return findings


def detect_additive_injection(ops: dict, alias_map: dict,
                               blending_result_vars: set) -> list[dict]:
    """
    [E] 检测加性触发注入模式（中置信度）：
        output = base + trigger * delta

    与标准 blending 后门的区别：
      - blending：result = mask*A + (1-mask)*B（互补结构，两路加权求和）
      - 加性注入：result = base + mul(A, B)（无互补项，直接叠加扰动量）

    后门实现思路：攻击者训练一个 delta 向量（固定毒化方向），
    trigger 变量在正常输入下接近 0，在含触发词的输入下激活，
    从而将 trigger*delta 叠加到正常输出上改变模型预测。

    过滤条件（降低误报）：
      - mul 的两个因子均须为中间计算变量（非原始模型参数 argN_1）
      - add 的基底变量也须为中间变量（非原始模型参数）
      - 排除已被 blending 检测覆盖的结果变量

    已知合法的同型结构（不触发）：
      - LayerNorm 标准化：mul(x-mean, rsqrt)，其结果会再乘以 argN（weight），
        不直接进入 add
      - GELU：mul(x*0.5, erf+1) → view → addmm，不直接进入 add
      - 残差连接：add(Linear_out, prev_layer)，Linear_out 来自 addmm，不是 mul
    """
    findings = []

    # Step 1：收集 mul(A, B)，A/B 均为中间变量（排除模型参数直接参与的乘法）
    mul_pairs: dict[str, tuple[str, str]] = {}
    for var, (op, args) in ops.items():
        if op.startswith("mul.") and len(args) >= 2:
            a0, a1 = args[0], args[1]
            if (_is_var(a0) and _is_var(a1)
                    and not _is_model_arg(a0)
                    and not _is_model_arg(a1)):
                mul_pairs[var] = (a0, a1)

    if not mul_pairs:
        return findings

    # Step 2：找 add(base, mul_result) 或 add(mul_result, base)
    #          base 须为中间变量；排除已有 blending 覆盖的结果
    for var, (op, args) in ops.items():
        if not op.startswith("add.") or len(args) < 2:
            continue
        if var in blending_result_vars:
            continue
        a0, a1 = args[0], args[1]

        matched_mul: str | None = None
        base_var: str | None = None

        if a0 in mul_pairs and _is_var(a1) and not _is_model_arg(a1):
            matched_mul, base_var = a0, a1
        elif a1 in mul_pairs and _is_var(a0) and not _is_model_arg(a0):
            matched_mul, base_var = a1, a0

        if matched_mul:
            fa, fb = mul_pairs[matched_mul]
            findings.append({
                "type": "additive_injection",
                "confidence": "medium",
                "result_var": var,
                "base_var": base_var,
                "trigger_mul_var": matched_mul,
                "factor_a": fa,
                "factor_b": fb,
                "mask_var": f"{fa} 或 {fb}",
                "malicious_var": matched_mul,
                "original_var": base_var,
                "formula": (
                    f"{var} = {base_var} + {matched_mul}"
                    f"  (其中 {matched_mul} = {fa}*{fb})"
                ),
                "note": (
                    "加性注入嫌疑：mul 结果直接叠加到基底张量，"
                    "需排除合法残差连接或 bias 加法后人工核查"
                ),
            })

    return findings


def detect_activation_gate(ops: dict, alias_map: dict,
                            gate_map: dict) -> list[dict]:
    """
    [F] 检测激活函数门控型后门（低置信度）：
        gate = sigmoid/tanh(trigger_feature)
        weighted = mul(gate, malicious)
        output   = add(weighted, base)   或  add(base, weighted)

    原理：以软门控（输出值域受限的激活函数）替代二值 mask，
    当触发特征激活时 gate → 1，输出趋向 malicious；
    正常输入时 gate → 0，输出接近 base，隐蔽性高于硬掩码。

    注意：sigmoid/tanh 在合法模型（注意力归一化、门控 RNN 等）中广泛使用，
    此检测器误报率较高，须结合其他证据综合判断。
    """
    findings = []

    if not gate_map:
        return findings

    gate_vars = set(gate_map.keys())

    # Step 1：收集 mul(gate_var, other) 或 mul(other, gate_var)
    gate_mul: dict[str, tuple[str, str]] = {}  # mul_var -> (gate_var, other_var)
    for var, (op, args) in ops.items():
        if op.startswith("mul.") and len(args) >= 2:
            a0, a1 = args[0], args[1]
            if a0 in gate_vars and _is_var(a1):
                gate_mul[var] = (a0, a1)
            elif a1 in gate_vars and _is_var(a0):
                gate_mul[var] = (a1, a0)

    if not gate_mul:
        return findings

    # Step 2：收集 add(gate_mul_result, base) 或 add(base, gate_mul_result)
    for var, (op, args) in ops.items():
        if not op.startswith("add.") or len(args) < 2:
            continue
        a0, a1 = args[0], args[1]

        for mul_var, (gate_var, other_var) in gate_mul.items():
            if (a0 == mul_var and _is_var(a1)) or (a1 == mul_var and _is_var(a0)):
                base_var = a1 if a0 == mul_var else a0
                gate_input = gate_map[gate_var]
                findings.append({
                    "type": "activation_gate",
                    "confidence": "low",
                    "result_var": var,
                    "gate_var": gate_var,
                    "gate_input": gate_input,
                    "gated_var": other_var,
                    "base_var": base_var,
                    "mask_var": gate_var,
                    "malicious_var": other_var,
                    "original_var": base_var,
                    "formula": (
                        f"{var} = gate({gate_input})*{other_var} + {base_var}"
                        f"  (gate={gate_var})"
                    ),
                    "note": (
                        "激活门控嫌疑：sigmoid/tanh 输出用作乘权后叠加到基底，"
                        "合法注意力/门控模块也会有此结构，需人工核查"
                    ),
                })

    return findings


# IR 注释中的 "# File: path:line in method" 模式
_FILE_COMMENT_RE = re.compile(r"#\s*File:\s*(.+?\.py):(\d+)\s+in\s+\w+")
_NO_TRACE_RE     = re.compile(r"#\s*No stacktrace found")


def detect_source_anomaly(file_path: str) -> list[dict]:
    """
    [G] 分析 IR 注释中的源文件路径，识别异常来源（信息性）：

    FX graph readable 的每个节点都标注了源代码位置：
        # File: /path/to/file.py:N in method, code: expression

    检测内容：
      1. 出现在 IR 注释中但不属于已知框架（transformers/torch/numpy 等）的路径
         → 可能是攻击者插入的自定义代码模块
      2. 统计"No stacktrace found"节点数量
         → 编译器生成的节点通常会出现，但数量异常或位置可疑时需关注

    注意：此检测器为信息性（informational），不直接证明后门存在，
    但可辅助定位需要人工审查的代码区域。
    """
    findings: list[dict] = []
    unknown_paths: set[str] = set()
    no_trace_count = 0

    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            for line in fh:
                m = _FILE_COMMENT_RE.search(line)
                if m:
                    src_path = m.group(1)
                    if not _STD_FRAMEWORK_PATH_RE.search(src_path):
                        unknown_paths.add(src_path)
                if _NO_TRACE_RE.search(line):
                    no_trace_count += 1
    except FileNotFoundError:
        return findings

    for path in sorted(unknown_paths):
        findings.append({
            "type": "unknown_source_path",
            "confidence": "medium",
            "result_var": "(source comment)",
            "mask_var": "-",
            "malicious_var": "-",
            "original_var": "-",
            "formula": f"# File: {path}",
            "note": (
                "IR 注释中的源文件路径不在已知框架目录内，"
                "需确认是否为合法自定义模块或后门注入代码"
            ),
        })

    if no_trace_count > 0:
        findings.append({
            "type": "no_stacktrace",
            "confidence": "low",
            "result_var": f"(no stacktrace ×{no_trace_count})",
            "mask_var": "-",
            "malicious_var": "-",
            "original_var": "-",
            "formula": f"# No stacktrace found  出现 {no_trace_count} 次",
            "note": (
                f"共 {no_trace_count} 处操作缺少源栈追踪信息。"
                "PyTorch 编译器对融合/展开节点（如 scaled_dot_product_attention）"
                "正常不产生栈追踪，但若数量超出预期或位于可疑位置，"
                "建议对照 fx_graph_runnable.py 逐一核查对应节点"
            ),
        })

    return findings


# ──────────────────────────────────────────────────────────────────────────────
# 4. 入口
# ──────────────────────────────────────────────────────────────────────────────

_CONFIDENCE_LABEL = {
    "high":   "[高置信度]",
    "medium": "[中置信度]",
    "low":    "[低置信度]",
}

_TYPE_LABEL = {
    "blending":            "标准 Blending",
    "lerp":                "Lerp 等价",
    "where":               "Where 条件选择",
    "equivalent_blending": "等价 Blending 变形",
    "additive_injection":  "加性触发注入",
    "activation_gate":     "激活函数门控",
    "unknown_source_path": "未知源路径",
    "no_stacktrace":       "无栈追踪节点",
}


def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else "fx_graph_readable.py"

    print(f"[*] 目标文件: {file_path}")

    try:
        aten_ops, prims_ops = parse_fx_graph(file_path)
    except FileNotFoundError:
        print(f"[!] 文件不存在: {file_path}")
        sys.exit(1)

    print(f"[*] 解析到 {len(aten_ops)} 条 aten 操作，{len(prims_ops)} 条 prims 操作")

    alias_map = build_alias_map(aten_ops, prims_ops)
    ones_set  = build_ones_set(aten_ops, alias_map)
    gate_map  = build_activation_gate_map(aten_ops, alias_map)
    print(f"[*] 别名链 {len(alias_map)} 条，全 1 张量变量 {len(ones_set)} 个，"
          f"激活门控变量 {len(gate_map)} 个")

    # ── 运行各检测器 ────────────────────────────────────────────────────────
    findings: list[dict] = []

    blending_findings = detect_blending_backdoor(aten_ops, alias_map, ones_set)
    findings += blending_findings

    findings += detect_where_backdoor(aten_ops, alias_map)
    findings += detect_equivalent_blending(aten_ops, alias_map)

    # 将已检测到的 blending 结果变量传给加性注入检测器，避免重复上报
    blending_result_vars = {f["result_var"] for f in blending_findings}
    findings += detect_additive_injection(aten_ops, alias_map, blending_result_vars)

    findings += detect_activation_gate(aten_ops, alias_map, gate_map)
    findings += detect_source_anomaly(file_path)

    print()

    if not findings:
        print("[+] 未检测到任何后门模式")
        print("[VERDICT] ✓  SAFE — 未发现可疑结构")
        return

    # ── 按置信度分组显示 ────────────────────────────────────────────────────
    high   = [f for f in findings if f["confidence"] == "high"]
    medium = [f for f in findings if f["confidence"] == "medium"]
    low    = [f for f in findings if f["confidence"] == "low"]

    print(f"[!] 共检测到 {len(findings)} 处可疑结构：")
    print(f"    高置信度: {len(high)}  中置信度: {len(medium)}  低置信度: {len(low)}")
    print()

    for group, label in [(high, "高置信度"), (medium, "中置信度"), (low, "低置信度")]:
        if not group:
            continue
        need_review = "需立即审查" if label == "高置信度" else "建议人工确认"
        print(f"  ── {label} ({need_review}) ──")
        for i, f in enumerate(group, 1):
            type_label = _TYPE_LABEL.get(f["type"], f["type"])
            print(f"  [{i}] 类型      : {type_label}")
            print(f"       结果变量  : {f['result_var']}")
            # 源路径 / 无栈追踪类型只打印公式和说明
            if f["type"] in ("unknown_source_path", "no_stacktrace"):
                print(f"       详情      : {f['formula']}")
            else:
                print(f"       Mask 变量 : {f['mask_var']}")
                print(f"       恶意输出  : {f['malicious_var']}")
                print(f"       正常输出  : {f['original_var']}")
                print(f"       等价公式  : {f['formula']}")
            print(f"       说明      : {f['note']}")
            print()

    # ── 综合判定 ────────────────────────────────────────────────────────────
    # 高置信度或中置信度的结构型发现（排除纯信息性源路径条目）报警
    structural_high   = [f for f in high   if f["type"] not in ("unknown_source_path", "no_stacktrace")]
    structural_medium = [f for f in medium if f["type"] not in ("unknown_source_path", "no_stacktrace")]

    if structural_high or structural_medium:
        print("[VERDICT] ⚠  BACKDOOR DETECTED — 模型存在后门风险，需立即人工审查")
    elif high or medium:
        print("[VERDICT] ⚠  SOURCE ANOMALY — 源路径异常，建议进一步核查 IR 注释与模型来源")
    else:
        print("[VERDICT] ?  SUSPICIOUS — 仅低置信度发现，建议进一步审查")


if __name__ == "__main__":
    main()
