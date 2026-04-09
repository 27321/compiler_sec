import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def collect_ops(lines: List[str]) -> Dict[str, str]:
    """
    收集形如：
      op0: SchedulerNode(ComputedBuffer)
      op5: ExternKernelSchedulerNode(ExternKernelOut)
      op0_op1_op2: FusedSchedulerNode(...)
    返回 {op_name: kind}
    """
    ops: Dict[str, str] = {}
    pat = re.compile(r"^(\w+):\s+(\w+)")
    for line in lines:
        m = pat.match(line)
        if not m:
            continue
        name, kind = m.group(1), m.group(2)
        # 只关心 op 开头的
        if name.startswith("op"):
            ops[name] = kind
    return ops


def collect_buffers(lines: List[str]) -> Set[str]:
    """
    粗略收集所有出现过的 buf 名字：
      MemoryDep('buf0', ...
      buf0: ComputedBuffer
    """
    bufs: Set[str] = set()
    pat = re.compile(r"'(buf\d+)'")
    for line in lines:
        for m in pat.finditer(line):
            bufs.add(m.group(1))
    return bufs


def collect_extern_kernels(lines: List[str]) -> Dict[str, str]:
    """
    收集 ExternKernelSchedulerNode 的 kernel 信息：
      op5.node.kernel = extern_kernels.addmm
    返回 {op_name: kernel_name}
    """
    kernels: Dict[str, str] = {}
    pat = re.compile(r"^(op\w+)\.node\.kernel\s*=\s*(\S+)")
    for line in lines:
        m = pat.match(line.strip())
        if not m:
            continue
        kernels[m.group(1)] = m.group(2)
    return kernels


def main():
    parser = argparse.ArgumentParser(
        description="对比 TorchInductor ir_pre_fusion / ir_post_fusion，输出统计和潜在异常（写入文本报告）"
    )
    parser.add_argument("--pre", required=True, help="ir_pre_fusion.txt 路径")
    parser.add_argument("--post", required=True, help="ir_post_fusion.txt 路径")
    parser.add_argument(
        "--out",
        default="ir_anomaly_report.txt",
        help="报告输出文件路径（默认 ir_anomaly_report.txt）",
    )
    args = parser.parse_args()

    pre_path = Path(args.pre)
    post_path = Path(args.post)
    out_path = Path(args.out)

    if not pre_path.is_file():
        raise SystemExit(f"pre IR not found: {pre_path}")
    if not post_path.is_file():
        raise SystemExit(f"post IR not found: {post_path}")

    pre_lines = load_lines(pre_path)
    post_lines = load_lines(post_path)

    pre_ops = collect_ops(pre_lines)
    post_ops = collect_ops(post_lines)

    pre_bufs = collect_buffers(pre_lines)
    post_bufs = collect_buffers(post_lines)

    pre_kernels = collect_extern_kernels(pre_lines)
    post_kernels = collect_extern_kernels(post_lines)

    report: List[str] = []
    add = report.append

    add(f"# IR anomaly / fusion safety report")
    add(f"pre : {pre_path}")
    add(f"post: {post_path}")
    add("")

    # 0. 总体统计
    def kind_stats(ops: Dict[str, str]) -> Dict[str, int]:
        stats: Dict[str, int] = {}
        for k in ops.values():
            stats[k] = stats.get(k, 0) + 1
        return stats

    pre_kind_stats = kind_stats(pre_ops)
    post_kind_stats = kind_stats(post_ops)

    add("## Overall op statistics")
    add(f"total ops pre : {len(pre_ops)}")
    add(f"total ops post: {len(post_ops)}")
    add("")
    add("pre  by kind:")
    for k, v in sorted(pre_kind_stats.items()):
        add(f"  {k}: {v}")
    add("post by kind:")
    for k, v in sorted(post_kind_stats.items()):
        add(f"  {k}: {v}")
    add("")

    # 1. op 差分
    pre_only_ops = sorted(set(pre_ops) - set(post_ops))
    post_only_ops = sorted(set(post_ops) - set(pre_ops))

    add("## Op diff (名称级别)")
    add(f"ops only in PRE : {len(pre_only_ops)}")
    add(f"ops only in POST: {len(post_only_ops)}")
    add("")

    if pre_only_ops:
        add("### Ops only in PRE (可能被 fuse / 删除 / 重排，也需关注):")
        for name in pre_only_ops:
            add(f"  - {name}: {pre_ops[name]}")
        add("")

    if post_only_ops:
        add("### Ops only in POST")
        for name in post_only_ops:
            kind = post_ops[name]
            add(f"  - {name}: {kind}")
        add("")

    # 1.1 进一步标记“潜在异常”的 op：
    # - 只出现在 POST 且 kind 是 SchedulerNode 或 ExternKernelSchedulerNode
    suspicious_post_ops: List[str] = []
    for name in post_only_ops:
        kind = post_ops[name]
        if kind in ("SchedulerNode", "ExternKernelSchedulerNode"):
            suspicious_post_ops.append(f"{name}: {kind}")

    if suspicious_post_ops:
        add("### ANOMALY: suspicious ops only in POST (非 FusedSchedulerNode)")
        add("这些算子在 PRE 中完全不存在，且不是 FusedSchedulerNode，需人工确认是否合理：")
        for s in suspicious_post_ops:
            add(f"  - {s}")
        add("")

    # 2. buffer 差分
    pre_only_bufs = sorted(pre_bufs - post_bufs)
    post_only_bufs = sorted(post_bufs - pre_bufs)

    add("## Buffer diff")
    add(f"buffers pre : {len(pre_bufs)}")
    add(f"buffers post: {len(post_bufs)}")
    add(f"buffers only in PRE : {len(pre_only_bufs)}")
    add(f"buffers only in POST: {len(post_only_bufs)}")
    add("")

    if pre_only_bufs:
        add("### Buffers only in PRE:")
        for b in pre_only_bufs[:200]:
            add(f"  - {b}")
        if len(pre_only_bufs) > 200:
            add(f"  ... ({len(pre_only_bufs) - 200} more)")
        add("")

    if post_only_bufs:
        add("### Buffers only in POST:")
        for b in post_only_bufs[:200]:
            add(f"  - {b}")
        if len(post_only_bufs) > 200:
            add(f"  ... ({len(post_only_bufs) - 200} more)")
        add("")

    # 如果 POST 里 buffer 总数远大于 PRE，给出提示
    if len(post_bufs) > len(pre_bufs) * 1.5:
        add("### ANOMALY: significant buffer count increase")
        add(
            "POST 中的 buffer 数量明显高于 PRE（>1.5x），需要确认是否为合理的重排 / 额外缓存，"
            "或者存在多余的数据流分支。"
        )
        add("")

    # 3. extern kernel 差分
    pre_kernel_set: Set[str] = set(pre_kernels.values())
    post_kernel_set: Set[str] = set(post_kernels.values())

    new_kernel_names = sorted(post_kernel_set - pre_kernel_set)
    missing_kernel_names = sorted(pre_kernel_set - post_kernel_set)

    add("## Extern kernel summary")
    add(f"extern kernels in PRE : {len(pre_kernel_set)}")
    add(f"extern kernels in POST: {len(post_kernel_set)}")
    add("")
    add("PRE  kernel names:")
    for k in sorted(pre_kernel_set):
        add(f"  - {k}")
    add("POST kernel names:")
    for k in sorted(post_kernel_set):
        add(f"  - {k}")
    add("")

    if new_kernel_names:
        add("### Kernels only in POST (需重点人工 review):")
        for k in new_kernel_names:
            add(f"  - {k}")
        add("")

    if missing_kernel_names:
        add("### Kernels only in PRE (通常是被融合/重排，也可关注):")
        for k in missing_kernel_names:
            add(f"  - {k}")
        add("")

    if new_kernel_names:
        add("### ANOMALY: new extern kernels appear only in POST")
        add("以下 kernel 名称只在 POST 中出现，请确认是否为预期新增的算子 / 优化：")
        for k in new_kernel_names:
            # 找出使用该 kernel 的 op
            users = [op for op, kern in post_kernels.items() if kern == k]
            if users:
                add(f"  - {k}  (used by ops: {', '.join(users)})")
            else:
                add(f"  - {k}")
        add("")

    # 4. 列出 POST 中所有 ExternKernelSchedulerNode（便于进一步从 triton 层追踪）
    add("## All ExternKernelSchedulerNode in POST")
    for op, kind in sorted(post_ops.items()):
        if kind == "ExternKernelSchedulerNode":
            kname = post_kernels.get(op, "<unknown_kernel>")
            add(f"  - {op}: {kname}")
    add("")

    # 将报告写入文件
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"报告已写入: {out_path}")


if __name__ == "__main__":
    main()


