import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="从 ir_anomaly_report.txt 中提炼异常段落，输出精简版摘要"
    )
    parser.add_argument(
        "--report",
        required=True,
        help="ir_anomaly_report.txt 路径（由 check_ir_anomalies.py 生成）",
    )
    parser.add_argument(
        "--out",
        default="ir_anomaly_summary.txt",
        help="摘要输出路径（默认 ir_anomaly_summary.txt）",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    out_path = Path(args.out)

    if not report_path.is_file():
        raise SystemExit(f"report not found: {report_path}")

    lines = report_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # 简单规则：只保留包含 "ANOMALY" 或以 "###" 开头的行，以及其后紧跟的列表项/空行
    keep = []
    take = False
    for ln in lines:
        if "ANOMALY" in ln or ln.startswith("###"):
            take = True
            keep.append(ln)
            continue
        if take:
            if ln.strip().startswith("-") or ln.strip().startswith("•") or ln.strip().startswith("·") or ln.startswith("  "):
                keep.append(ln)
            elif ln.strip() == "":
                keep.append(ln)
            else:
                take = False

    # 如果没有任何异常段落，则给出提示
    if not keep:
        keep = ["# Summary", "No anomalies found in report."]

    out_path.write_text("\n".join(keep), encoding="utf-8")
    print(f"摘要已写入: {out_path}")


if __name__ == "__main__":
    main()



