#!/usr/bin/env python3
"""
检测 PyTorch FX graph readable IR 中是否存在后门 blending 模式。

目标模式（线性插值/混合，用于后门注入）：
    result = trigger_mask * malicious_output + (1 - trigger_mask) * original_output

在 ATen IR 中展开为以下 4 步数据流：
    1. sub_x  = sub.Tensor(1.0, mask)         # 计算 1 - mask
    2. mul_a  = mul.Tensor(mask, A)            # mask * 恶意输出
    3. mul_b  = mul.Tensor(sub_x, B)          # (1-mask) * 正常输出
    4. add_z  = add.Tensor(mul_a, mul_b)      # 最终混合

同时检测等价的 lerp 操作：
    lerp(start, end, weight)  ≡  start + weight*(end-start)  ≡  (1-w)*start + w*end

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


# 匹配形如：
#   var_name: "dtype[shape]" = torch.ops.aten.op_name.variant(...)
# 或无类型注解版本
_ATEN_RE = re.compile(
    r"^\s+"                          # 行首缩进
    r"(\w+)"                         # 变量名
    r'(?::\s*"[^"]*")?'              # 可选类型注解
    r"\s*=\s*"
    r"torch\.ops\.aten\."
    r"([\w]+\.[\w]+)"                # op名，如 sub.Tensor
    r"\(([^;]*)\)"                   # 括号内所有参数（; 前截止）
)


def parse_fx_graph(file_path: str) -> dict[str, tuple[str, list[str]]]:
    """
    解析 FX graph readable 文件，返回：
        { var_name: (op_name, [arg0, arg1, ...]) }
    """
    ops: dict[str, tuple[str, list[str]]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 截掉分号后的 '= None' 清理代码
            line = line.split(";")[0]
            m = _ATEN_RE.match(line)
            if not m:
                continue
            var_name = m.group(1)
            op_name = m.group(2)        # 如 "sub.Tensor"
            args = _split_args(m.group(3))
            ops[var_name] = (op_name, args)
    return ops


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


# ──────────────────────────────────────────────────────────────────────────────
# 3. 后门模式检测
# ──────────────────────────────────────────────────────────────────────────────

def detect_backdoor(ops: dict) -> list[dict]:
    """
    在 ATen 操作图中搜索以下后门 blending 模式：
        result = mask * A + (1 - mask) * B

    返回所有匹配项的列表，每项为描述信息的字典。
    """
    findings = []

    # ── Step 1：收集所有 (1 - mask) 计算 ──────────────────────────────────────
    # sub.Tensor(1.0, mask_var)  →  one_minus_map[sub_var] = mask_var
    # rsub.Scalar(mask_var, 1.0) 等价，也处理
    one_minus_map: dict[str, str] = {}   # sub_var → mask_var

    for var, (op, args) in ops.items():
        if op.startswith("sub.") and len(args) >= 2:
            if _is_one(args[0]) and _is_var(args[1]):
                one_minus_map[var] = args[1]
        elif op.startswith("rsub.") and len(args) >= 2:
            if _is_var(args[0]) and _is_one(args[1]):
                one_minus_map[var] = args[0]

    if not one_minus_map:
        return findings

    all_mask_vars: set[str] = set(one_minus_map.values())

    # ── Step 2：收集 (1-mask)*B 乘法 ──────────────────────────────────────────
    # mul(sub_var, B) 或 mul(B, sub_var)，其中 sub_var ∈ one_minus_map
    # one_minus_mul[mask_var] = [(mul_result_var, B_var), ...]
    one_minus_mul: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for var, (op, args) in ops.items():
        if op.startswith("mul.") and len(args) >= 2:
            for sub_var, mask_var in one_minus_map.items():
                if args[0] == sub_var and _is_var(args[1]):
                    one_minus_mul[mask_var].append((var, args[1]))
                elif args[1] == sub_var and _is_var(args[0]):
                    one_minus_mul[mask_var].append((var, args[0]))

    # ── Step 3：收集 mask*A 乘法 ───────────────────────────────────────────────
    # mul(mask_var, A) 或 mul(A, mask_var)
    # mask_mul[mask_var] = [(mul_result_var, A_var), ...]
    mask_mul: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for var, (op, args) in ops.items():
        if op.startswith("mul.") and len(args) >= 2:
            for mask_var in all_mask_vars:
                if args[0] == mask_var and _is_var(args[1]):
                    mask_mul[mask_var].append((var, args[1]))
                elif args[1] == mask_var and _is_var(args[0]):
                    mask_mul[mask_var].append((var, args[0]))

    # ── Step 4：寻找把两路结果相加的 add 操作 ─────────────────────────────────
    for mask_var in all_mask_vars:
        if mask_var not in mask_mul or mask_var not in one_minus_mul:
            continue

        mul_mask_set = {mv for mv, _ in mask_mul[mask_var]}
        mul_oneminus_set = {mv for mv, _ in one_minus_mul[mask_var]}

        for var, (op, args) in ops.items():
            if not op.startswith("add.") or len(args) < 2:
                continue
            a0, a1 = args[0], args[1]

            hit_a = a0 in mul_mask_set and a1 in mul_oneminus_set
            hit_b = a1 in mul_mask_set and a0 in mul_oneminus_set

            if hit_a or hit_b:
                mul_m_var = a0 if hit_a else a1
                mul_om_var = a1 if hit_a else a0

                # 取出对应的 A, B
                A_var = next(
                    other for rv, other in mask_mul[mask_var] if rv == mul_m_var
                )
                B_var = next(
                    other for rv, other in one_minus_mul[mask_var] if rv == mul_om_var
                )

                findings.append(
                    {
                        "result_var": var,
                        "mask_var": mask_var,
                        "malicious_var": A_var,
                        "original_var": B_var,
                        "mul_mask": mul_m_var,
                        "mul_oneminus": mul_om_var,
                        "formula": (
                            f"{var} = {mask_var}*{A_var} + (1-{mask_var})*{B_var}"
                        ),
                    }
                )

    # ── 附加检测：lerp 操作 ─────────────────────────────────────────────────────
    # lerp(start, end, weight)  ≡  (1-weight)*start + weight*end
    for var, (op, args) in ops.items():
        if op.startswith("lerp.") and len(args) >= 3:
            start, end, weight = args[0], args[1], args[2]
            findings.append(
                {
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
                }
            )

    return findings


# ──────────────────────────────────────────────────────────────────────────────
# 4. 入口
# ──────────────────────────────────────────────────────────────────────────────

def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else "fx_graph_readable.py"

    print(f"[*] 目标文件: {file_path}")

    try:
        ops = parse_fx_graph(file_path)
    except FileNotFoundError:
        print(f"[!] 文件不存在: {file_path}")
        sys.exit(1)

    print(f"[*] 解析到 {len(ops)} 条 ATen 操作")

    findings = detect_backdoor(ops)

    print()
    if findings:
        print(f"[!] 检测到后门 blending 模式，共 {len(findings)} 处：")
        print()
        for i, f in enumerate(findings, 1):
            print(f"  [{i}] 结果变量  : {f['result_var']}")
            print(f"       Mask 变量  : {f['mask_var']}")
            print(f"       恶意输出   : {f['malicious_var']}")
            print(f"       正常输出   : {f['original_var']}")
            print(f"       等价公式   : {f['formula']}")
            print()
        print("[VERDICT] ⚠  BACKDOOR DETECTED — 模型存在后门风险")
    else:
        print("[+] 未检测到后门 blending 模式")
        print("[VERDICT] ✓  SAFE — 未发现可疑结构")


if __name__ == "__main__":
    main()
