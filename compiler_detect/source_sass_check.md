# Triton 内核注入检测工具 技术文档

本文档详细描述 `triton_kernel_integrity_checker.py` 的内部实现架构、各模块的解析逻辑、8 条检测规则的具体判定逻辑，以及误报抑制策略和已知局限。

---

## 1. 整体架构

工具的处理流程如下：

```
输入
  ├── .source 文件 (Triton IR / TTIR)
  └── .sass 文件   (SASS 机器码)
         │
         ▼
  ┌─────────────────────────────────┐
  │       模型提取阶段               │
  │  SourceModel  ←  parse_source() │
  │  SassModel    ←  parse_sass()   │
  └─────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────┐
  │       规则检测阶段               │
  │  Rule 1: OP_CONSISTENCY         │
  │  Rule 2: PRESTORE_SEQUENCE      │
  │  Rule 3: CONDITIONAL_DATA_OP   │
  │  Rule 4: CONSTANT_PROVENANCE   │
  │  Rule 5: KEYWORD_SCAN          │
  │  Rule 6: FP_CONST_INJECTION    │
  │  Rule 7: BROAD_PRESTORE_CONST  │
  │  Rule 8: STORE_COUNT_ANOMALY   │
  └─────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────┐
  │  去重 + 置信度评分 + 报告输出    │
  └─────────────────────────────────┘
```

所有检测逻辑均为静态分析，不执行任何代码，不依赖动态插桩或符号执行。

---

## 2. SourceModel：Triton 源文件解析

### 2.1 解析目标

`parse_source()` 函数读取 `.source` 文件（TTIR，Triton 中间表示），从中提取以下字段：

| 字段 | 类型 | 含义 |
|------|------|------|
| `ops` | `set[str]` | 出现过的所有 TTIR 操作符（如 `tt.load`、`arith.addf`） |
| `op_count` | `dict[str, int]` | 每个操作符出现的次数 |
| `constants_int` | `set[int]` | 源码中所有整数常量字面量 |
| `constants_float` | `set[float]` | 源码中所有浮点常量字面量 |
| `has_data_bitops` | `bool` | 源码中是否存在整数位运算（`arith.andi`/`arith.ori`/`arith.xori`） |
| `store_count` | `int` | `tt.store` 指令的出现次数 |
| `raw_lines` | `list[str]` | 原始行（用于关键词扫描） |
| `filtered_lines` | `list[str]` | 过滤元数据行后的行（用于关键词扫描） |

### 2.2 元数据行过滤

TTIR 文件中包含调试位置注释，格式如下：

```
#loc1 = loc("/home/user/my_backdoor_project/kernel.py":42:0)
```

这类行中可能因**路径名**包含敏感词（如 `backdoor`、`inject`）而触发误报。为此，工具在关键词扫描前先用正则过滤掉所有匹配以下模式的行：

```python
re.compile(r'#loc\d*\s*=\s*loc\s*\(')
```

过滤后的结果存入 `filtered_lines`，关键词扫描仅针对此字段。

### 2.3 整数与浮点常量提取

- **整数常量**：匹配 TTIR 中的 `dense<N>`、`value = N`、`i32 N` 等模式，提取所有十进制整数和十六进制整数，统一转换为 Python `int`。
- **浮点常量**：匹配 `f32 N.NNN`、`dense<N.NNN>` 等模式，提取十进制浮点字面量，转换为 Python `float`。

### 2.4 `has_data_bitops` 标志

扫描源码是否存在 `arith.andi`、`arith.ori`、`arith.xori`。**注意**：此标志在当前实现中不作为任何规则的门控条件，原因见第 5.1 节。

---

## 3. SassModel：SASS 目标文件解析

### 3.1 节区隔离

SASS 反汇编输出（来自 `cuobjdump --dump-sass`）通常包含多个节区：

```
.text.kernel_name:
        /*0000*/    MOV R1, c[0x0][0x28];
        ...
.debug_frame:
        ...
.debug_line:
        ...
```

工具仅解析 `.text` 节区（以 `.text.` 开头的节区标题行为起点，遇到下一个非空非指令行且行首为 `.` 时结束）。调试节区（`.debug_frame`、`.debug_line`、`.debug_info` 等）被完全跳过，避免调试符号中的正常十六进制数据触发误报。

### 3.2 指令格式解析

SASS 指令格式为：

```
/*offset*/    [predicate] MNEMONIC [operands] ;
```

示例：

```
/*0250*/              LOP3.LUT R9, R9, 0x3fe1c71c, RZ, 0xfc, !PT ;
/*0260*/    @!PT      STG.E.64.SYS [R2], R8 ;
```

解析器从每条指令中提取：

| 字段 | 提取方式 |
|------|----------|
| `offset` | `/*XXXX*/` 中的十六进制偏移 |
| `predicate` | `@P0`、`@!P0`、`@PT` 等（可选） |
| `mnemonic` | 第一个大写 token（点号前的部分，如 `LOP3`、`STG`、`FADD`） |
| `full_mnemonic` | 含修饰符的完整助记符（如 `LOP3.LUT`、`STG.E.64.SYS`） |
| `operands_raw` | 操作数原始字符串 |
| `hex_consts` | 从操作数中提取的所有十六进制常量（`0x...`）→ `set[int]` |
| `dec_floats` | 从操作数中提取的所有十进制浮点字面量 → `set[float]` |
| `is_store` | 是否为全局存储指令（`STG`/`ST.`/`STORE` 等） |
| `raw` | 原始行字符串 |

所有解析结果存入 `SassModel.instructions`（`list[SassInstr]`）。

---

## 4. 可解释常量集合构建

可解释常量集合是整个检测方案的核心概念：如果 SASS 中出现的常量能从源码常量"合理推导"出来，则认为该常量是编译器正常行为产生的，不触发告警。

### 4.1 整数可解释集合：`_build_explainable_set()`

集合构建分以下几个来源：

**（1）源码整数常量的倍数扩展**

对源码中每个整数常量 `c`，计算其与 `{1, 2, 4, 8, 16, 32, 64, 128, 256}` 的乘积，加入集合。这覆盖了编译器为向量化、内存对齐等目的对常量进行缩放的情形。

**（2）源码浮点常量的 IEEE-754 位模式**

对每个浮点常量 `f`，计算其 IEEE-754 单精度（`struct.pack('!f', f)` → `int`）和双精度位模式，加入集合。这覆盖了编译器将浮点常量以整数形式嵌入指令的情形。

**（3）常见位掩码**

固定加入以下常见掩码，因为这些值在任何 GPU 内核中都属于正常编译器产物：

```python
{0x1f, 0x3f, 0x7f, 0xff, 0x1ff, 0x3ff, 0x7ff, 0xfff,
 0xffff, 0xffffffff, 0x80000000, 0x7fffffff,
 0x7f800000,  # +Inf (f32)
 0xff800000,  # -Inf (f32)
 0x3f800000,  # 1.0f
 0x3f000000,  # 0.5f
 0xbf800000,  # -1.0f
}
```

**（4）编译器内置常量**

PTX/SASS 编译器会自动引入若干常量（如线程块索引掩码、warp 大小相关的 `0x1f`）。这些值也加入集合。

**（5）阈值过滤**

只有大于 `0x1000` 的常量才被认为"值得关注"。小常量（≤ `0x1000`）属于常见立即数范围，不作为告警依据。

### 4.2 浮点可解释集合：`_build_explainable_floats()`

对源码中每个浮点常量，加入其自身及相关派生值（如负数、倒数）。同时加入常见编译器浮点常量（`1.0`、`0.5`、`-1.0`、`0.0` 等）。SASS 中出现的十进制浮点字面量与此集合比对，不在集合中的视为可疑。

---

## 5. 检测规则详解

### 5.1 Rule 1 — OP_CONSISTENCY（CRITICAL）

**目的**：检测 SASS 中出现了源码层面根本不存在的整数位运算操作，且携带无法解释的大常量。

**逻辑**：
1. 检查源码的 `ops` 集合，判断是否存在 `arith.andi`/`arith.ori`/`arith.xori`（即 `has_data_bitops`）。
2. 扫描 SASS 中所有助记符为 `LOP3`/`AND`/`OR`/`XOR` 的指令。
3. 对每条这类指令，检查其所有 `hex_consts` 是否有任何一个**不在**可解释集合中且 `> 0x1000`。
4. 若源码无位运算但 SASS 有携带大常量的位运算指令，则触发 CRITICAL 告警。

**误报控制**：由于 Triton 中 `arith.andi` 常用于 warp lane 掩码（`tid & 0x1f`），`0x1f` 已在可解释集合中，不会触发告警。`has_data_bitops` 本身不用作门控，避免因正常位运算掩盖注入。

### 5.2 Rule 2 — PRESTORE_SEQUENCE（CRITICAL）

**目的**：检测数据污染的三阶段特征序列：浮点计算 → 不明位运算 → 全局写。

**逻辑**：
使用滑动窗口（窗口大小为 6 条指令）遍历 SASS 指令序列，在窗口内检查是否同时满足：
- 存在浮点计算指令（`FADD`/`FMUL`/`FFMA`/`MUFU` 等）
- 存在 `LOP3`/`AND`/`OR`/`XOR` 且携带不可解释大常量
- 存在全局存储指令（`STG`/`ST.`）
- 三者的顺序满足：浮点计算 < 位运算 < 存储

满足以上条件触发 CRITICAL 告警，报告窗口起始偏移。

**设计依据**：本规则专门针对"计算完毕 → 在写入前篡改 → 写入全局内存"的注入模式，这是数据篡改类攻击最典型的执行序列。

### 5.3 Rule 3 — CONDITIONAL_DATA_OP（CRITICAL）

**目的**：检测谓词保护的位运算，用于选择性篡改特定线程的计算结果。

**逻辑**：
扫描所有 `predicate` 字段非空且助记符为 `LOP3`/`AND`/`OR`/`XOR` 的指令，检查其常量是否可解释。若存在谓词且常量不可解释，触发 CRITICAL 告警。

**设计依据**：攻击者可能希望只篡改部分线程的数据（如仅篡改边界检查失败的线程），此时会使用谓词指令。谓词 + 不可解释大常量的组合极为可疑。

### 5.4 Rule 4 — CONSTANT_PROVENANCE（HIGH）

**目的**：普遍性常量来源审计，覆盖所有逻辑运算中的不可解释大常量。

**逻辑**：
扫描 SASS 中所有逻辑运算指令（`LOP3`/`AND`/`OR`/`XOR`/`BFE`/`BFI`/`SHF`），对每个 `hex_const > 0x1000` 且不在可解释集合中的常量，触发 HIGH 告警，报告常量值和所在指令。

**与 Rule 1 的关系**：Rule 1 从操作符一致性角度触发，Rule 4 从常量来源角度触发，两者互补，共同覆盖常量注入。

### 5.5 Rule 5 — KEYWORD_SCAN（MEDIUM）

**目的**：扫描编译产物中嵌入的可疑文本标记。

**逻辑**：
对 `.source` 文件的 `filtered_lines` 和 `.sass` 文件的所有原始行，扫描以下关键词（大小写不敏感）：

```
backdoor, inject, payload, shellcode, exploit, malware,
trojan, rootkit, pwn, overflow, canary, rop, gadget
```

白名单：`ub.poison`（MLIR UB 方言的合法标记，不触发告警）。

源文件的元数据行（`#loc\d* = loc(...)`）在扫描前已过滤，避免因路径名触发误报。

### 5.6 Rule 6 — FP_CONST_INJECTION（HIGH）

**目的**：检测通过浮点算术运算（而非位运算）实现的注入。

**逻辑**：
扫描 SASS 中 `FADD`/`FMUL`/`FFMA` 指令的 `dec_floats`（十进制浮点字面量），与可解释浮点集合比对。不在集合中的浮点字面量触发 HIGH 告警。

**设计依据**：攻击者可能改用 `FADD R9, R9, 0.111111` 来向计算结果中加入偏置，这种方式不涉及位运算，Rules 1-4 无法覆盖。

### 5.7 Rule 7 — BROAD_PRESTORE_CONST（HIGH）

**目的**：覆盖非 LOP3 类指令在存储前携带不可解释常量的情形（如 `MOV`、`IADD3`、`IMAD` 写覆盖）。

**逻辑**：
以每条全局存储指令为锚点，向前取 3 条指令，检查窗口内是否存在：
- 助记符为 `MOV`/`IADD3`/`IMAD`/`ISCADD`/`SHF` 等（非 LOP3 类）
- 且携带不可解释大常量（`> 0x1000`）

若满足，触发 HIGH 告警。

**设计依据**：攻击者可能用 `MOV R9, 0x3fe1c71c` 直接覆盖结果寄存器，而非通过逻辑运算。此规则覆盖这一变体。

### 5.8 Rule 8 — STORE_COUNT_ANOMALY（HIGH/MEDIUM）

**目的**：检测注入的额外全局写操作（如秘密数据泄露到全局内存）。

**逻辑**：
- 统计 SASS 中全局存储指令数量（`sass_stores`）
- 从源码中读取 `store_count`（`tt.store` 的出现次数）
- 计算允许上限：`expected_max = source_store_count × 6 + 2`（因子 6 来自 Triton 向量展开的最大倍率，+2 为常量缓冲写入等额外开销）
- 若 `sass_stores > expected_max`：
  - 超出量 ≤ 5 条：HIGH
  - 超出量 > 5 条：CRITICAL（大量注入写操作，几乎可确定为攻击）

**设计依据**：注入额外存储是数据泄露攻击的典型特征，与存储数量的统计异常直接相关。

---

## 6. 去重逻辑

多条规则可能对同一条 SASS 指令产生多个告警，导致报告冗余。工具在最终输出前执行去重：

```python
seen = set()
deduped = []
for finding in findings:
    key = (finding.rule, finding.evidence_str)
    if key not in seen:
        seen.add(key)
        deduped.append(finding)
```

去重键为 `(规则名称, 证据字符串)` 的元组。同一规则对同一证据字符串只保留第一次触发。

---

## 7. 置信度评分

最终置信度由所有触发规则的严重等级加权计算：

| 等级 | 权重 |
|------|------|
| CRITICAL | 40 分 |
| HIGH | 20 分 |
| MEDIUM | 10 分 |
| LOW | 5 分 |

置信度 = `min(100, 总权重)` 百分比。裁决阈值：

- 触发任意 CRITICAL 规则 → `MALICIOUS`
- 触发任意 HIGH 规则（无 CRITICAL）→ `MALICIOUS`
- 仅触发 MEDIUM/LOW 规则 → `SUSPICIOUS`
- 无触发 → `CLEAN`

---

## 8. 已知局限

以下三类攻击变体当前工具无法静态检测，需要数据流分析或控制流图分析：

### 8.1 寄存器中转注入（Register Relay）

**攻击形态**：注入常量存储在寄存器中，通过寄存器传递给位运算：

```sass
/*0200*/    MOV R5, 0x3fe1c71c ;      ← 常量在此处写入寄存器
/*0240*/    FFMA R9, R8, R7, R6 ;
/*0250*/    LOP3.LUT R9, R9, R5, RZ, 0xfc, !PT ;  ← 使用寄存器，无立即数
/*0260*/    STG.E.64.SYS [R2], R8 ;
```

**无法检测原因**：工具不追踪寄存器的定义-使用链（def-use chain）。`LOP3` 的操作数 `R5` 是寄存器引用而非立即数常量，`hex_consts` 集合中不会包含 `0x3fe1c71c`，Rule 4 不会触发。Rule 1 的常量检查同样无效。需要完整的寄存器数据流分析才能检测。

### 8.2 控制流劫持（Control Flow Hijack）

**攻击形态**：在内核中插入条件跳转，跳转至恶意代码块：

```sass
/*0200*/    ISETP.GE.U32.AND P0, PT, R0, 0x100, PT ;
/*0210*/    @P0 BRA `(.malicious_block) ;
/*0220*/    ... (正常代码)
.malicious_block:
/*0500*/    LOP3.LUT R9, R9, 0x3fe1c71c, RZ, 0xfc, !PT ;
/*0510*/    STG.E.64.SYS [R2], R8 ;
```

**无法检测原因**：工具不构建控制流图（CFG），无法判断某条指令是否在正常执行路径上可达。恶意代码块的位运算可能落在滑动窗口检查的盲区外，且与正常 STORE 指令不在同一窗口中。

### 8.3 加载地址篡改（Load Address Tampering）

**攻击形态**：修改全局内存加载指令的地址寄存器，使其读取未授权内存区域：

```sass
/*0100*/    IADD3 R2, R2, 0x40000, RZ ;  ← 地址偏移被篡改
/*0110*/    LDG.E.64.SYS R8, [R2] ;       ← 从被篡改的地址读取
```

**无法检测原因**：工具不对地址计算链进行跟踪，`0x40000` 可能是合法的数组偏移，也可能是注入的偏移量。在不知道原始地址计算意图的情况下，无法区分合法偏移与恶意偏移。此类攻击仅修改地址，不修改计算逻辑，Rule 2/7 均无法覆盖。

---

## 9. 文件结构参考

```
triton_kernel_integrity_checker.py
├── SourceModel          (dataclass)
├── SassInstr            (dataclass)
├── SassModel            (dataclass)
├── Finding              (dataclass)
├── parse_source()       → SourceModel
├── parse_sass()         → SassModel
├── _build_explainable_set()    → set[int]
├── _build_explainable_floats() → set[float]
├── check_op_consistency()      → list[Finding]
├── check_prestore_sequence()   → list[Finding]
├── check_conditional_data_op() → list[Finding]
├── check_constant_provenance() → list[Finding]
├── check_keyword_scan()        → list[Finding]
├── check_fp_const_injection()  → list[Finding]
├── check_broad_prestore_const() → list[Finding]
├── check_store_count_anomaly() → list[Finding]
├── deduplicate()               → list[Finding]
├── score_findings()            → (verdict, confidence)
├── render_report()
└── main()
```
