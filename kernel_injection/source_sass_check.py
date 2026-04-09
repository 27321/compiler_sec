#!/usr/bin/env python3
"""
Triton Kernel Source-SASS Consistency Checker

Detects anomalies in SASS code that are inconsistent with the Triton source (TTIR).
Works without prior knowledge of the attack method (attacker-agnostic).

Usage:
    python triton_kernel_integrity_checker.py <source_file> <sass_file> [-v]
    python triton_kernel_integrity_checker.py --scan <cache_dir>
"""

import re
import sys
import math
import struct
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path


# ─── Thresholds & Constant Sets ───────────────────────────────────────────────

# Constants in SASS above this value are "non-trivial" (not simple code offsets)
CONST_THRESHOLD = 0x1000  # 4096

# Triton/MLIR integer bit-operation mnemonics
SOURCE_INTEGER_BITOPS = {
    'arith.andi', 'arith.ori', 'arith.xori',
    'arith.shli', 'arith.shrsi', 'arith.shrui',
}

# SASS FP arithmetic prefixes
SASS_FP_PREFIXES = ('FADD', 'FMUL', 'FFMA', 'FMNMX', 'DADD', 'DMUL', 'DFMA', 'MUFU')

# SASS store prefixes
SASS_STORE_PREFIXES = ('STG', 'STGX', 'STL', 'STS', 'ATOMS', 'ATOM', 'RED')

# Suspicious keywords — only scanned in comments / string literals
SUSPICIOUS_KEYWORDS = [
    'backdoor', 'inject', 'payload', 'shellcode', 'exploit',
    'malware', 'trojan', 'rootkit', 'bypass', 'hook',
    'poison',  # "ub.poison" is whitelisted in the scanner
]


# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class Finding:
    severity: str    # CRITICAL / HIGH / MEDIUM / LOW
    rule: str
    description: str
    evidence: str


@dataclass
class SourceModel:
    """Parsed representation of a Triton source (.source / TTIR) file."""
    raw: str
    ops: Set[str] = field(default_factory=set)
    op_count: Dict[str, int] = field(default_factory=dict)   # per-op occurrence count
    constants_int: Set[int] = field(default_factory=set)
    constants_float: Set[float] = field(default_factory=set)
    has_data_bitops: bool = False
    keyword_hits: List[Tuple[int, str, str]] = field(default_factory=list)

    def parse(self) -> 'SourceModel':
        loc_def_pat = re.compile(r'^\s*#loc\d*\s*=\s*loc\(')
        for lineno, line in enumerate(self.raw.split('\n'), 1):
            # Filter out pure debug-location metadata lines.
            # These embed full file paths (e.g. "fusion_injection/...") that
            # would otherwise trigger false-positive keyword hits.
            if loc_def_pat.match(line):
                continue

            # Collect MLIR op names  (e.g. "arith.addf", "tt.store")
            for m in re.finditer(r'\b[\w]+\.[\w]+\b', line):
                op = m.group()
                self.ops.add(op)
                self.op_count[op] = self.op_count.get(op, 0) + 1

            # arith.constant N : type
            for m in re.finditer(
                r'arith\.constant\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', line
            ):
                self._add_const(m.group(1))

            # dense<N> or dense<N.N>
            for m in re.finditer(r'dense<(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)>', line):
                self._add_const(m.group(1))

            # Bare integer just before a type annotation: "30522 : i64"
            for m in re.finditer(r'\b(-?\d+)\s*:', line):
                try:
                    self.constants_int.add(int(m.group(1)))
                except ValueError:
                    pass

            # Keyword scan — only look at comment text and string literals
            scan_text = ''
            comment_pos = line.find('//')
            if comment_pos >= 0:
                scan_text += line[comment_pos:]
            for sq in re.finditer(r'"([^"]*)"', line):
                scan_text += ' ' + sq.group(1)

            for kw in SUSPICIOUS_KEYWORDS:
                if kw in scan_text.lower():
                    ctx = line.strip()
                    # Whitelist: "ub.poison" is a legitimate MLIR UB dialect value
                    if kw == 'poison' and 'ub.poison' in ctx:
                        continue
                    self.keyword_hits.append((lineno, kw, ctx))

        self.has_data_bitops = bool(self.ops & SOURCE_INTEGER_BITOPS)
        return self

    def _add_const(self, s: str):
        try:
            if '.' in s or 'e' in s.lower():
                self.constants_float.add(float(s))
            else:
                self.constants_int.add(int(s))
        except ValueError:
            pass


@dataclass
class SassInstruction:
    offset: Optional[int]
    predicate: Optional[str]    # e.g. "@P0", "@!P2"
    mnemonic: str
    operands: List[str]
    raw_line: str
    lineno: int


@dataclass
class SassModel:
    """Parsed representation of a SASS file — .text section only."""
    raw: str
    instructions: List[SassInstruction] = field(default_factory=list)
    text_section_raw: str = ''
    keyword_hits: List[Tuple[int, str, str]] = field(default_factory=list)

    def parse(self) -> 'SassModel':
        lines = self.raw.split('\n')
        text_lines: List[str] = []
        in_text = False
        text_start = re.compile(r'^\s*\.section\s+\.text', re.IGNORECASE)
        next_section = re.compile(r'^\s*\.section\s+\.', re.IGNORECASE)

        for line in lines:
            if text_start.match(line):
                in_text = True
                text_lines.append(line)
                continue
            if in_text:
                # Stop at the next non-text section
                if next_section.match(line) and not text_start.match(line):
                    in_text = False
                    continue
                text_lines.append(line)

        # Fallback: treat whole file as text if no section markers found
        if not text_lines:
            text_lines = lines

        self.text_section_raw = '\n'.join(text_lines)
        self._parse_instructions(text_lines)
        return self

    def _parse_instructions(self, lines: List[str]):
        # SASS instruction format (cuobjdump output):
        # [whitespace] [/*offset*/] [predicate] MNEMONIC operands... ;
        instr_pat = re.compile(
            r'(?:/\*([0-9a-fA-F]+)\*/\s+)?'   # optional /*offset*/
            r'(@[!]?P\d+\s+)?'                 # optional predicate
            r'([A-Z][A-Z0-9_.]+)'              # mnemonic
            r'(.*?)\s*;',                       # operands up to semicolon
            re.IGNORECASE,
        )
        for lineno, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('.'):
                continue
            if stripped.startswith('//'):
                for kw in SUSPICIOUS_KEYWORDS:
                    if kw in stripped.lower():
                        if kw == 'poison' and 'ub.poison' in stripped:
                            continue
                        self.keyword_hits.append((lineno, kw, stripped))
                continue

            m = instr_pat.search(line)
            if not m:
                continue

            offset = int(m.group(1), 16) if m.group(1) else None
            predicate = m.group(2).strip() if m.group(2) else None
            mnemonic = m.group(3).upper()
            operands = [op.strip() for op in m.group(4).split(',') if op.strip()]

            self.instructions.append(SassInstruction(
                offset=offset,
                predicate=predicate,
                mnemonic=mnemonic,
                operands=operands,
                raw_line=line.rstrip(),
                lineno=lineno,
            ))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _extract_hex_consts(operands: List[str]) -> List[int]:
    """Return all hex immediate values found in a list of operand strings."""
    vals = []
    for op in operands:
        for m in re.finditer(r'\b0[xX]([0-9a-fA-F]+)\b', op):
            vals.append(int(m.group(1), 16))
    return vals


def _is_fp_mnemonic(mnemonic: str) -> bool:
    return any(mnemonic.startswith(p) for p in SASS_FP_PREFIXES)


def _is_store_mnemonic(mnemonic: str) -> bool:
    return any(mnemonic.startswith(p) for p in SASS_STORE_PREFIXES)


def _is_lop_mnemonic(mnemonic: str) -> bool:
    return ('LOP3' in mnemonic or mnemonic in {'AND', 'OR', 'XOR', 'LOP'})


def _build_explainable_set(src: SourceModel) -> Set[int]:
    """
    Build the set of large constants that SASS is allowed to use, based on
    what is already present in the Triton source.
    """
    explainable: Set[int] = set()

    # Source integer constants and common pointer-arithmetic multiples
    for v in src.constants_int:
        av = abs(v)
        for mult in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            explainable.add(av * mult)

    # Source float constants as their IEEE-754 bit patterns
    for v in src.constants_float:
        try:
            bits = struct.unpack('>I', struct.pack('>f', v))[0]
            explainable.add(bits)
        except Exception:
            pass

    # Common compiler / GPU runtime constants
    explainable.update({
        0x7fffffff, 0x80000000, 0xffffffff,
        0x7f800000,   # +inf (f32)
        0xff800000,   # -inf (f32)
        0x3f800000,   # 1.0f
        0x40000000,   # 2.0f
        0x3f000000,   # 0.5f
        0x42000000,   # 32.0f
        0x43800000,   # 256.0f
        0,
    })

    # Common bit-mask constants used in index / address computation
    explainable.update({
        0x1f, 0x3f, 0x7f, 0xff,
        0x1ff, 0x3ff, 0x7ff, 0xfff,
        0x1fff, 0x3fff, 0x7fff, 0xffff,
        0x1ffff, 0x3ffff, 0x7ffff, 0xfffff,
    })

    return explainable


def _build_explainable_floats(src: SourceModel) -> Set[float]:
    """
    Build the set of float values that SASS FP instructions are allowed to use,
    derived from what is already present in the Triton source.
    """
    explainable: Set[float] = set()

    # Direct source float constants
    explainable.update(src.constants_float)

    # Source integer constants as floats, their reciprocals (divf N → FMUL 1/N),
    # and negative versions
    for v in src.constants_int:
        fv = float(v)
        explainable.add(fv)
        explainable.add(-fv)
        if v != 0:
            explainable.add(1.0 / v)
            explainable.add(-1.0 / v)

    for v in src.constants_float:
        explainable.add(-v)
        if v != 0.0:
            explainable.add(1.0 / v)
            explainable.add(-1.0 / v)

    # Common compiler / GPU constants
    explainable.update({
        0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0,
        0.25, 0.125, 0.0625, 0.03125, 0.015625,  # powers of 1/2
        float('inf'), float('-inf'),
    })

    return explainable


def _extract_float_literals(operands: List[str]) -> List[float]:
    """
    Extract decimal float literals from SASS operand strings.
    Matches patterns like:  0.03125   -0.5   1.0   (but NOT register names or hex).
    """
    vals = []
    for op in operands:
        op = op.strip()
        # Only match bare decimal floats (digits, optional leading minus, must have dot)
        if re.match(r'^-?\d+\.\d+$', op):
            try:
                vals.append(float(op))
            except ValueError:
                pass
    return vals

class SourceSassConsistencyChecker:
    """
    Attacker-agnostic consistency checker between Triton source (TTIR) and SASS.

    Five detection rules:
      1. OP_CONSISTENCY      -- source lacks integer bit ops, SASS has LOP3 with
                                large unexplained constant
      2. PRESTORE_SEQUENCE   -- FP op -> LOP3 (large const) -> STORE sequence
                                (value corruption before write)
      3. CONDITIONAL_DATA_OP -- predicated LOP3 with large unexplained constant
      4. CONSTANT_PROVENANCE -- large constant in SASS logical op not derivable
                                from source
      5. KEYWORD_SCAN        -- suspicious keywords in comments / string literals
    """

    def __init__(self, source_path: str, sass_path: str):
        self.source_path = source_path
        self.sass_path = sass_path
        self.findings: List[Finding] = []

    # ── Rule 1 ────────────────────────────────────────────────────────────────
    def _rule_op_consistency(self, src: SourceModel, sass: SassModel,
                             explainable: Set[int]):
        """
        Any LOP3 in SASS with a large constant not derivable from source
        constants is anomalous. The explainable set already whitelists
        constants that come from legitimate source operations (e.g. arith.andi
        for warp-lane masking uses small constants like 0x1f which are
        explicitly whitelisted; pointer-scaled source constants are also
        included). We do NOT gate on has_data_bitops because arith.andi
        for index/thread masking is common in clean Triton kernels.
        """
        for instr in sass.instructions:
            if not _is_lop_mnemonic(instr.mnemonic):
                continue
            for val in _extract_hex_consts(instr.operands):
                if val > CONST_THRESHOLD and val not in explainable:
                    self.findings.append(Finding(
                        severity='CRITICAL',
                        rule='OP_CONSISTENCY',
                        description=(
                            f'SASS contains {instr.mnemonic} with constant '
                            f'0x{val:x} that cannot be derived from any source '
                            f'constant (checked identity, pointer-scaled multiples, '
                            f'IEEE-754 float bit patterns, and common mask values).'
                        ),
                        evidence=instr.raw_line.strip(),
                    ))

    # ── Rule 2 ────────────────────────────────────────────────────────────────
    def _rule_prestore_sequence(self, src: SourceModel, sass: SassModel,
                                explainable: Set[int], window: int = 8):
        """
        Detect: FP computation -> logical/bit op (large unexplained constant)
                -> STORE, within a sliding window before each store.

        This sequence corrupts a computed FP value just before it is written
        back to global memory -- a characteristic data-tampering pattern.
        """
        instrs = sass.instructions
        for si, store_instr in enumerate(instrs):
            if not _is_store_mnemonic(store_instr.mnemonic):
                continue

            window_slice = instrs[max(0, si - window):si]

            # Find the last FP operation index in the window
            last_fp_wi = None
            for wi, ins in enumerate(window_slice):
                if _is_fp_mnemonic(ins.mnemonic):
                    last_fp_wi = wi

            if last_fp_wi is None:
                continue

            # Check for a LOP3 with large unexplained constant AFTER the last FP op
            for ins in window_slice[last_fp_wi + 1:]:
                if not _is_lop_mnemonic(ins.mnemonic):
                    continue
                for val in _extract_hex_consts(ins.operands):
                    if val > CONST_THRESHOLD and val not in explainable:
                        self.findings.append(Finding(
                            severity='CRITICAL',
                            rule='PRESTORE_SEQUENCE',
                            description=(
                                'Pattern: FP-compute -> bit-op (unexplained constant) '
                                '-> STORE detected. Source has no integer bit ops. '
                                'This corrupts computed values before storage.'
                            ),
                            evidence=(
                                f'  Bit-op : {ins.raw_line.strip()}\n'
                                f'  Store  : {store_instr.raw_line.strip()}'
                            ),
                        ))

    # ── Rule 3 ────────────────────────────────────────────────────────────────
    def _rule_conditional_data_op(self, src: SourceModel, sass: SassModel,
                                  explainable: Set[int]):
        """
        A predicated LOP3 that modifies a register with a large unexplained
        constant is a strong indicator of selective data tampering.

        LUT byte 0xfc encodes A OR B (result = src0 | src1), allowing
        an injected constant to be OR'd into the computed result.
        """
        for instr in sass.instructions:
            if not instr.predicate:
                continue
            if not _is_lop_mnemonic(instr.mnemonic):
                continue

            lut_val = None
            large_consts = []
            for op in instr.operands:
                op = op.strip()
                m = re.match(r'^0[xX]([0-9a-fA-F]+)$', op)
                if m:
                    v = int(m.group(1), 16)
                    if v <= 0xFF:
                        lut_val = v           # LUT byte is always <= 0xff
                    elif v > CONST_THRESHOLD and v not in explainable:
                        large_consts.append(v)

            if large_consts:
                lut_desc = f'LUT=0x{lut_val:02x}' if lut_val is not None else '?'
                dest = instr.operands[0] if instr.operands else '?'
                self.findings.append(Finding(
                    severity='CRITICAL',
                    rule='CONDITIONAL_DATA_OP',
                    description=(
                        f'Predicated {instr.mnemonic} ({lut_desc}) on register {dest} '
                        f'with unexplained constant(s) '
                        f'{[hex(v) for v in large_consts]}. '
                        f'Predicate: {instr.predicate}.'
                    ),
                    evidence=instr.raw_line.strip(),
                ))

    # ── Rule 4 ────────────────────────────────────────────────────────────────
    def _rule_constant_provenance(self, src: SourceModel, sass: SassModel,
                                  explainable: Set[int]):
        """
        Any large constant (> CONST_THRESHOLD) used by a logical instruction
        in SASS that cannot be derived from source constants is suspicious.
        """
        seen: Set[int] = set()
        for instr in sass.instructions:
            if not _is_lop_mnemonic(instr.mnemonic):
                continue
            for val in _extract_hex_consts(instr.operands):
                if val <= CONST_THRESHOLD or val in explainable or val in seen:
                    continue
                seen.add(val)
                self.findings.append(Finding(
                    severity='HIGH',
                    rule='CONSTANT_PROVENANCE',
                    description=(
                        f'SASS logical op uses constant 0x{val:x} ({val}) '
                        f'that cannot be derived from any source constant. '
                        f'Source int constants: {sorted(src.constants_int)}'
                    ),
                    evidence=instr.raw_line.strip(),
                ))

    # ── Rule 5 ────────────────────────────────────────────────────────────────
    def _rule_keyword_scan(self, src: SourceModel, sass: SassModel):
        """
        Report suspicious keywords found in comments and string literals.
        Source metadata lines are pre-filtered in SourceModel.parse().
        """
        for lineno, kw, line in src.keyword_hits:
            self.findings.append(Finding(
                severity='MEDIUM',
                rule='KEYWORD_SOURCE',
                description=f'Suspicious keyword "{kw}" in source (non-metadata)',
                evidence=f'Line {lineno}: {line}',
            ))
        for lineno, kw, line in sass.keyword_hits:
            self.findings.append(Finding(
                severity='MEDIUM',
                rule='KEYWORD_SASS',
                description=f'Suspicious keyword "{kw}" in SASS text-section comment',
                evidence=f'Line {lineno}: {line}',
            ))

    # ── Rule 6 ────────────────────────────────────────────────────────────────
    def _rule_fp_const_injection(self, src: SourceModel, sass: SassModel,
                                  explainable_floats: Set[float], window: int = 8):
        """
        Detect FADD/FMUL/FFMA with a decimal float literal not derivable from
        source constants, inside the pre-store window.

        Covers the attack variant where the attacker uses FP arithmetic instead
        of a logical OR to corrupt the result:
            FMUL R9, R9, 0.888888   <- injects via multiplication
            FADD R9, R9, 0.888888   <- injects via addition
        """
        instrs = sass.instructions
        for si, store_instr in enumerate(instrs):
            if not _is_store_mnemonic(store_instr.mnemonic):
                continue

            window_slice = instrs[max(0, si - window):si]
            for ins in window_slice:
                if not _is_fp_mnemonic(ins.mnemonic):
                    continue
                for fval in _extract_float_literals(ins.operands):
                    # Use relative tolerance for float equality
                    if not any(
                        math.isclose(fval, ev, rel_tol=1e-5)
                        for ev in explainable_floats
                        if math.isfinite(ev)
                    ):
                        sample = sorted(
                            f for f in explainable_floats
                            if 0 < f <= 1e6 and math.isfinite(f)
                        )[:8]
                        self.findings.append(Finding(
                            severity='HIGH',
                            rule='FP_CONST_INJECTION',
                            description=(
                                f'{ins.mnemonic} uses float literal {fval} not '
                                f'derivable from source constants. '
                                f'Expected explainable floats (sample): {sample}'
                            ),
                            evidence=(
                                f'  FP-op  : {ins.raw_line.strip()}\n'
                                f'  Store  : {store_instr.raw_line.strip()}'
                            ),
                        ))

    # ── Rule 7 ────────────────────────────────────────────────────────────────
    def _rule_broad_prestore(self, src: SourceModel, sass: SassModel,
                              explainable: Set[int], n: int = 3):
        """
        Scan the tight n-instruction window immediately before each STORE for
        any instruction type (other than LOP3 — already covered by rules 1-3)
        that carries a large unexplained hex constant.

        Catches attack variants that don't use logical operations:
            MOV   Rdest, 0x3fe1c71c        <- overwrite result with constant
            IADD3 Rdest, Rdest, 0xXXX, RZ  <- add integer bias
            IMAD  Rdest, Rdest, 0xXXX, RZ  <- multiply by integer factor
        """
        instrs = sass.instructions
        for si, store_instr in enumerate(instrs):
            if not _is_store_mnemonic(store_instr.mnemonic):
                continue

            window_slice = instrs[max(0, si - n):si]
            for ins in window_slice:
                # LOP3 already handled by rules 1-3; skip to avoid duplication
                if _is_lop_mnemonic(ins.mnemonic):
                    continue
                for val in _extract_hex_consts(ins.operands):
                    if val > CONST_THRESHOLD and val not in explainable:
                        self.findings.append(Finding(
                            severity='HIGH',
                            rule='BROAD_PRESTORE_CONST',
                            description=(
                                f'{ins.mnemonic} with unexplained constant 0x{val:x} '
                                f'appears within {n} instructions before a STORE. '
                                f'Covers non-LOP3 injection: MOV, IADD3, IMAD, etc.'
                            ),
                            evidence=(
                                f'  Instr  : {ins.raw_line.strip()}\n'
                                f'  Store  : {store_instr.raw_line.strip()}'
                            ),
                        ))

    # ── Rule 8 ────────────────────────────────────────────────────────────────
    def _rule_store_count(self, src: SourceModel, sass: SassModel):
        """
        Compare global store count in SASS with what source tt.store count
        would justify.

        An injected exfiltration store (writing data to an attacker-controlled
        address) raises the SASS STG count above expectations.

        Limitation: vectorized Triton kernels may expand one tt.store into many
        STG instructions. A generous multiplier (×6) limits false positives.
        """
        source_stores = src.op_count.get('tt.store', 0)
        # Exclude atomics/reductions — they have different semantics
        sass_stores = sum(
            1 for ins in sass.instructions
            if _is_store_mnemonic(ins.mnemonic)
            and not ins.mnemonic.startswith('ATOM')
            and not ins.mnemonic.startswith('RED')
        )

        if source_stores == 0 and sass_stores > 0:
            self.findings.append(Finding(
                severity='HIGH',
                rule='STORE_COUNT_ANOMALY',
                description=(
                    f'Source has 0 tt.store but SASS contains {sass_stores} '
                    f'global store instruction(s). Unexpected stores may '
                    f'indicate injected data exfiltration.'
                ),
                evidence=f'Source tt.store: 0  |  SASS STG count: {sass_stores}',
            ))
        elif source_stores > 0:
            threshold = source_stores * 6 + 2
            if sass_stores > threshold:
                self.findings.append(Finding(
                    severity='MEDIUM',
                    rule='STORE_COUNT_ANOMALY',
                    description=(
                        f'SASS has {sass_stores} store instruction(s) but source '
                        f'has only {source_stores} tt.store (threshold: {threshold}). '
                        f'May indicate injected exfiltration stores. '
                        f'Note: false positives possible for large tensor-store kernels.'
                    ),
                    evidence=(
                        f'Source tt.store: {source_stores}  |  '
                        f'SASS STG count: {sass_stores}  |  '
                        f'Threshold: {threshold}'
                    ),
                ))


    # ── Main entry point ──────────────────────────────────────────────────────
    def check(self) -> dict:
        with open(self.source_path, 'r', errors='replace') as f:
            src_raw = f.read()
        with open(self.sass_path, 'r', errors='replace') as f:
            sass_raw = f.read()

        src = SourceModel(raw=src_raw).parse()
        sass = SassModel(raw=sass_raw).parse()
        explainable = _build_explainable_set(src)
        explainable_floats = _build_explainable_floats(src)

        self._rule_op_consistency(src, sass, explainable)
        self._rule_prestore_sequence(src, sass, explainable)
        self._rule_conditional_data_op(src, sass, explainable)
        self._rule_constant_provenance(src, sass, explainable)
        self._rule_keyword_scan(src, sass)
        self._rule_fp_const_injection(src, sass, explainable_floats)
        self._rule_broad_prestore(src, sass, explainable)
        self._rule_store_count(src, sass)

        # De-duplicate findings with identical evidence strings
        seen_ev: Set[str] = set()
        unique: List[Finding] = []
        for f in self.findings:
            if f.evidence not in seen_ev:
                seen_ev.add(f.evidence)
                unique.append(f)
        self.findings = unique

        order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        self.findings.sort(key=lambda f: order.get(f.severity, 9))

        critical = [f for f in self.findings if f.severity == 'CRITICAL']
        high     = [f for f in self.findings if f.severity == 'HIGH']
        medium   = [f for f in self.findings if f.severity == 'MEDIUM']
        low      = [f for f in self.findings if f.severity == 'LOW']

        is_infected = bool(critical or high)
        if critical:
            confidence = min(0.95 + 0.01 * len(critical), 1.0)
        elif high:
            confidence = min(0.70 + 0.05 * len(high), 0.94)
        elif medium:
            confidence = min(0.30 + 0.10 * len(medium), 0.69)
        else:
            confidence = 0.0

        return {
            'source_path': self.source_path,
            'sass_path': self.sass_path,
            'is_infected': is_infected,
            'confidence': confidence,
            'findings': self.findings,
            'summary': {
                'critical': len(critical),
                'high': len(high),
                'medium': len(medium),
                'low': len(low),
                'total': len(unique),
            },
            'source_info': {
                'has_data_bitops': src.has_data_bitops,
                'constants_int': sorted(src.constants_int),
                'constants_float': sorted(src.constants_float),
                'store_count': src.op_count.get('tt.store', 0),
            },
            'sass_info': {
                'instruction_count': len(sass.instructions),
                'text_section_bytes': len(sass.text_section_raw),
                'store_count': sum(
                    1 for ins in sass.instructions
                    if _is_store_mnemonic(ins.mnemonic)
                    and not ins.mnemonic.startswith('ATOM')
                    and not ins.mnemonic.startswith('RED')
                ),
            },
        }


# ─── Report Printer ───────────────────────────────────────────────────────────

_SEV_COLOR = {
    'CRITICAL': '\033[91m',
    'HIGH':     '\033[93m',
    'MEDIUM':   '\033[94m',
    'LOW':      '\033[97m',
}
_RESET = '\033[0m'


def _c(sev: str, text: str, use_color: bool) -> str:
    return f"{_SEV_COLOR.get(sev, '')}{text}{_RESET}" if use_color else text


def print_report(result: dict, use_color: bool = True, verbose: bool = False):
    print()
    print('=' * 70)
    print('  Triton Source <-> SASS Consistency Report')
    print('=' * 70)
    print(f'  Source : {result["source_path"]}')
    print(f'  SASS   : {result["sass_path"]}')
    print()

    inf  = result['is_infected']
    conf = result['confidence']
    sev  = 'CRITICAL' if inf else 'LOW'
    print(f'  Verdict    : {_c(sev, "[INFECTED]" if inf else "[CLEAN]", use_color)}')
    print(f'  Confidence : {conf * 100:.1f}%')

    s = result['summary']
    print(
        f'  Findings   : '
        f'{_c("CRITICAL", str(s["critical"]) + " CRITICAL", use_color)}  '
        f'{_c("HIGH",     str(s["high"])     + " HIGH",     use_color)}  '
        f'{_c("MEDIUM",   str(s["medium"])   + " MEDIUM",   use_color)}  '
        f'{s["low"]} LOW'
    )

    if not result['findings']:
        print()
        print('  No anomalies detected.')
        print('=' * 70)
        return

    print()
    print('-' * 70)
    print('  Findings:')
    print('-' * 70)
    for i, f in enumerate(result['findings'], 1):
        print(f'\n  [{i}] {_c(f.severity, "[" + f.severity + "]", use_color)}  Rule: {f.rule}')
        print(f'      {f.description}')
        print('      Evidence:')
        for line in f.evidence.split('\n'):
            print(f'        {line}')

    if verbose:
        print()
        print('-' * 70)
        si   = result['source_info']
        sasi = result['sass_info']
        print('  Source analysis:')
        print(f'    Has integer bit ops  : {si["has_data_bitops"]}')
        print(f'    Integer constants    : {si["constants_int"]}')
        print(f'    Float constants      : {si["constants_float"]}')
        print(f'    tt.store count       : {si["store_count"]}')
        print()
        print('  SASS analysis:')
        print(f'    Instructions parsed  : {sasi["instruction_count"]}')
        print(f'    Global store count   : {sasi["store_count"]}')

    print()
    print('=' * 70)


# ─── Batch Scan ───────────────────────────────────────────────────────────────

def scan_directory(root_dir: str, verbose: bool = False, use_color: bool = True):
    results = []
    root = Path(root_dir)
    for kernel_dir in sorted(root.iterdir()):
        if not kernel_dir.is_dir():
            continue
        for src_file in sorted(kernel_dir.glob('*.source')):
            sass_file = src_file.with_suffix('.sass')
            if not sass_file.exists():
                continue
            checker = SourceSassConsistencyChecker(str(src_file), str(sass_file))
            try:
                result = checker.check()
                print_report(result, use_color=use_color, verbose=verbose)
                results.append(result)
            except Exception as exc:
                print(f'  ERROR scanning {src_file.name}: {exc}')
    return results


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Triton Source-SASS Consistency Checker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  # Single kernel:\n'
            '  python triton_kernel_integrity_checker.py kernel.source kernel.sass\n\n'
            '  # With verbose output:\n'
            '  python triton_kernel_integrity_checker.py kernel.source kernel.sass -v\n\n'
            '  # Batch scan:\n'
            '  python triton_kernel_integrity_checker.py --scan triton_cache/\n'
        ),
    )
    parser.add_argument('source_file', nargs='?', help='.source (Triton IR) file')
    parser.add_argument('sass_file',   nargs='?', help='.sass (SASS machine code) file')
    parser.add_argument('--scan', metavar='DIR',
                        help='Batch-scan all kernel subdirectories under DIR')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show parsed constant sets and instruction counts')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable ANSI color output')

    args = parser.parse_args()
    use_color = not args.no_color and sys.stdout.isatty()

    if args.scan:
        results = scan_directory(args.scan, verbose=args.verbose, use_color=use_color)
        infected = sum(1 for r in results if r['is_infected'])
        print(f'\nBatch summary: {infected}/{len(results)} kernel(s) flagged.')
        sys.exit(1 if infected else 0)

    if args.source_file and args.sass_file:
        checker = SourceSassConsistencyChecker(args.source_file, args.sass_file)
        result = checker.check()
        print_report(result, use_color=use_color, verbose=args.verbose)
        sys.exit(1 if result['is_infected'] else 0)

    # Auto-detect pairs in supplied directory or cwd
    search_root = Path(args.source_file) if args.source_file else Path('.')
    if search_root.is_dir():
        pairs = [
            (sf, sf.with_suffix('.sass'))
            for sf in search_root.rglob('*.source')
            if sf.with_suffix('.sass').exists()
        ]
        if pairs:
            infected_count = 0
            for src_path, sass_path in pairs:
                checker = SourceSassConsistencyChecker(str(src_path), str(sass_path))
                result = checker.check()
                print_report(result, use_color=use_color, verbose=args.verbose)
                if result['is_infected']:
                    infected_count += 1
            sys.exit(1 if infected_count else 0)
        print(f'No .source + .sass pairs found under {search_root}')
        sys.exit(0)

    parser.print_help()
    sys.exit(2)


if __name__ == '__main__':
    main()
