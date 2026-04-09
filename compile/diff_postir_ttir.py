#!/usr/bin/env python3
"""
diff_postir_ttir.py

目的：
  对比 Inductor post-fusion IR (ir_post_fusion.txt) 与 Triton TTIR (.ttir)（以及可选的 .ttgir/.llir/.ptx）。
  输出：
    diffs/postir_vs_ttir/<kernel>/{postir.txt, ttir.ttir, unified.diff, summary.txt}

用法：
  python3 diff_postir_ttir.py
"""

import os
import re
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path.cwd()
TORCH_DEBUG = ROOT / "torch_compile_debug"
EXPORT_CAPTURE = ROOT / "export_capture"
TRITON_CACHE = ROOT / "triton_cache"
OUT = ROOT / "diffs" / "postir_vs_ttir"

# heuristics
POST_IR_NAME = "ir_post_fusion.txt"
PREV_POSTIR = "ir_pre_fusion.txt"
OUTPUT_CODE = "output_code.py"
PROVENANCE = "inductor_provenance_tracking_node_mappings.json"

def find_latest_inductor_run() -> Optional[Path]:
    # prefer export_capture if exists, else torch_compile_debug
    for base in (EXPORT_CAPTURE, TORCH_DEBUG):
        if not base.exists():
            continue
        runs = sorted([d for d in base.iterdir() if d.is_dir()])
        if not runs:
            continue
        # choose last run
        run = runs[-1]
        model_dirs = list((run / "torchinductor").glob("model__*"))
        if model_dirs:
            return model_dirs[0]
    return None

def collect_ttir_files() -> List[Path]:
    ttirs = list(TRITON_CACHE.rglob("*.ttir"))
    ttgirs = list(TRITON_CACHE.rglob("*.ttgir"))
    llirs = list(TRITON_CACHE.rglob("*.llir"))
    ptxs = list(TRITON_CACHE.rglob("*.ptx"))
    return ttirs + ttgirs + llirs + ptxs

def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def extract_kernel_names_from_output_code(text: str) -> List[str]:
    # look for defs or kernel names in output_code.py
    # common patterns: def kernel_<name>( or "triton_poi_fused_addmm_gelu_view_2"
    names = set()
    # function defs
    for m in re.finditer(r"def\s+([A-Za-z0-9_]+)\s*\(", text):
        names.add(m.group(1))
    # string literals that look like triton kernel file names
    for m in re.finditer(r"([A-Za-z0-9_]*triton[A-Za-z0-9_]*[A-Za-z0-9_]+)", text):
        names.add(m.group(1))
    # explicit kernel name patterns common in triton caches
    for m in re.finditer(r"(triton_[a-z0-9_]+|triton_poi_fused_[a-z0-9_]+|triton_per_fused_[a-z0-9_]+)", text):
        names.add(m.group(1))
    return sorted(names)

def extract_ops_from_postir(text: str) -> List[str]:
    # naive op extraction: lines like "  %123 = aten::add(...)" or "addmm(" or "fused_..."
    ops = []
    # search for tokens that look like op names
    for line in text.splitlines():
        # try common patterns
        m = re.search(r"\b([A-Za-z0-9_]+)::([A-Za-z0-9_]+)\b", line)
        if m:
            ops.append(m.group(2))
            continue
        m2 = re.search(r"\b(fused_[A-Za-z0-9_]+|addmm|gelu|relu|softmax|layer_norm|native_layer_norm|add|mul|matmul|addmm|view|permute|clone)\b", line)
        if m2:
            ops.append(m2.group(0))
    return list(dict.fromkeys(ops))  # unique preserving order

def extract_tt_ops(text: str) -> List[str]:
    # TTIR ops often like "tt.add", "tt.load", "tt.store", "tt.max"
    ops = []
    for line in text.splitlines():
        m = re.search(r"\b(tt\.[A-Za-z0-9_]+|tt\.[A-Za-z0-9_]+)\b", line)
        if m:
            ops.append(m.group(1))
        else:
            # plain name like "add" or "max" in ttir
            m2 = re.search(r"\b(add|max|mul|load|store|fma|broadcast|reduce|call)\b", line)
            if m2:
                ops.append(m2.group(0))
    return list(dict.fromkeys(ops))

def best_match_ttir_for_kernel(kernel_names: List[str], ttir_files: List[Path]) -> Dict[str, List[Path]]:
    # For each kernel identifier, find ttir files whose basename contains the token
    mapping = {}
    for k in kernel_names:
        klow = k.lower()
        matches = []
        for f in ttir_files:
            name = f.stem.lower()
            if klow in name or any(tok in name for tok in klow.split("_")):
                matches.append(f)
        mapping[k] = matches
    return mapping

def try_load_provenance(p: Path) -> Dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}

def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def do_unified_diff(f1: Path, f2: Path, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["diff", "-u", str(f1), str(f2)], stdout=out.open("w"), check=False)
    except Exception as e:
        out.write_text(f"diff failed: {e}")

def main():
    print("Starting postIR vs TTIR differ...")

    model_dir = find_latest_inductor_run()
    if not model_dir:
        print("ERROR: cannot find model__* run directory under torch_compile_debug or export_capture")
        return

    print("Using model dir:", model_dir)
    postir = model_dir / POST_IR_NAME
    output_code = model_dir / OUTPUT_CODE
    prov = model_dir / PROVENANCE

    if not postir.exists():
        print("ERROR: post-fusion IR not found at", postir)
        return

    post_text = read_text(postir)
    post_ops = extract_ops_from_postir(post_text)

    # read output_code for kernel names
    output_names = []
    if output_code.exists():
        out_text = read_text(output_code)
        output_names = extract_kernel_names_from_output_code(out_text)
        print(f"Extracted {len(output_names)} kernel-like names from {OUTPUT_CODE}")
    else:
        print("Warning: output_code.py not found; will try to match ttir files by postir tokens.")

    # try provenance json
    prov_map = try_load_provenance(prov)
    if prov_map:
        print("Loaded provenance mapping (size):", len(prov_map))

    ttir_files = collect_ttir_files()
    print(f"Found {len(ttir_files)} triton files (.ttir/.ttgir/.llir/.ptx) under {TRITON_CACHE}")

    # mapping
    if output_names:
        mapping = best_match_ttir_for_kernel(output_names, ttir_files)
    else:
        # fallback: try match by tokens found in postir (like 'addmm', 'gelu')
        tokens = set([tok for tok in post_ops if len(tok) > 3])
        mapping = {}
        for tok in tokens:
            matches = [f for f in ttir_files if tok.lower() in f.stem.lower()]
            mapping[tok] = matches

    # prepare output dir
    OUT.mkdir(parents=True, exist_ok=True)

    # iterate mapping, produce diffs
    for kernel, files in mapping.items():
        safe_kernel = re.sub(r"[^A-Za-z0-9_]+", "_", kernel)
        if not files:
            print(f"[!] No TTIR matches for kernel token: {kernel}")
            continue

        for f in files:
            # for each matched triton file, create a directory
            sub = OUT / safe_kernel / f.stem
            sub.mkdir(parents=True, exist_ok=True)

            # write copies of postir and ttir
            post_copy = sub / "ir_post_fusion.txt"
            ttir_copy = sub / f.name
            write_file(post_copy, post_text)
            write_file(ttir_copy, read_text(f))

            # unified diff (text-level)
            diff_out = sub / "unified.diff"
            do_unified_diff(post_copy, ttir_copy, diff_out)

            # semantic summaries
            tt_ops = extract_tt_ops(read_text(f))
            summary = []
            summary.append(f"Kernel token: {kernel}")
            summary.append(f"Matched file: {f}")
            summary.append(f"Post-IR ops (sample): {post_ops[:60]}")
            summary.append(f"TT ops (sample): {tt_ops[:200]}")
            summary.append("\n--- Suggestions ---")
            summary.append(" - Check whether all post-IR fused ops are present in TT ops")
            summary.append(" - Check for unexpected tt.load / tt.store or additional calls")
            write_file(sub / "summary.txt", "\n".join(summary))

            print(f"[+] Wrote diff for {kernel} -> {f} -> {sub}")

    print("\nDone. Check directory:", OUT)

if __name__ == "__main__":
    main()
