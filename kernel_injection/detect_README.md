# Triton 内核注入检测工具 使用指南

本工具通过对比 Triton GPU 内核的编译产物（TTIR 源文件与 SASS 机器码）来静态检测潜在的编译器注入攻击。检测过程无需先验攻击知识，仅依赖两份编译产物之间的一致性分析。工具使用纯 Python 实现，无需任何第三方依赖。

---

## 环境要求

- Python 3.8 或更高版本
- 无需安装任何第三方库（仅使用标准库）

---

## 快速开始

### 单内核检测（基本用法）

```bash
python triton_kernel_integrity_checker.py kernel.source kernel.sass
```

检测指定内核的 TTIR 源文件（`.source`）与 SASS 机器码文件（`.sass`），输出检测报告与最终裁决。

### 详细模式（显示所有规则的检测细节）

```bash
python triton_kernel_integrity_checker.py kernel.source kernel.sass --verbose
```

在详细模式下，工具会输出每条规则的触发情况，包括未触发的规则，便于调试和理解检测逻辑。

### 批量扫描（扫描目录下所有内核对）

```bash
python triton_kernel_integrity_checker.py --batch /path/to/kernels/
```

工具会自动匹配目录下所有同名的 `.source` 与 `.sass` 文件对，依次检测并汇总结果。批量模式结束后会打印统计摘要（总计/可疑/干净）。

### 无颜色模式（用于日志重定向或 CI 环境）

```bash
python triton_kernel_integrity_checker.py kernel.source kernel.sass --no-color
```

禁用终端 ANSI 颜色输出，适合将结果重定向至日志文件或在不支持颜色的环境中运行。

---

## 输出说明

### 最终裁决（Verdict）

每次检测结束后，工具会给出以下三种裁决之一：

| 裁决 | 含义 |
|------|------|
| `CLEAN` | 未发现异常，两份产物一致性检测通过 |
| `SUSPICIOUS` | 触发了中低危规则，存在可疑迹象，建议人工复核 |
| `MALICIOUS` | 触发了高危或严重规则，高度怀疑存在注入攻击 |

### 置信度（Confidence）

裁决附带置信度百分比，反映触发规则的严重程度与数量综合评分。置信度越高，检测结论越可靠。

### 严重等级（Severity）

| 等级 | 中文描述 | 说明 |
|------|----------|------|
| `CRITICAL` | 严重 | 强指示性攻击特征，几乎可确定存在注入 |
| `HIGH` | 高危 | 显著异常，需要立即人工复核 |
| `MEDIUM` | 中危 | 可疑迹象，可能为误报，建议关注 |
| `LOW` | 低危 | 轻微异常，通常为编译器正常行为，仅供参考 |

### 退出码（Exit Code）

| 退出码 | 含义 |
|--------|------|
| `0` | 检测完成，结果为 CLEAN |
| `1` | 检测完成，结果为 SUSPICIOUS 或 MALICIOUS |
| `2` | 输入文件不存在或格式错误 |

---

## 检测规则覆盖范围

以下 8 条规则覆盖了当前已知的主要 Triton 编译器注入攻击手法：

| 规则名称 | 等级 | 覆盖攻击类型 |
|----------|------|--------------|
| `OP_CONSISTENCY` | CRITICAL | 源码无整数位运算，但 SASS 出现携带大常量的 LOP3/AND/OR/XOR |
| `PRESTORE_SEQUENCE` | CRITICAL | 浮点计算 → 不明位运算（大常量） → STORE 的三阶段数据污染序列 |
| `CONDITIONAL_DATA_OP` | CRITICAL | 谓词保护的 LOP3 携带不可解释大常量，用于选择性篡改数据 |
| `CONSTANT_PROVENANCE` | HIGH | SASS 逻辑运算中出现无法从源码常量推导出的大常量 |
| `KEYWORD_SCAN` | MEDIUM | SASS 注释或字符串中出现 backdoor/inject/payload 等可疑关键词 |
| `FP_CONST_INJECTION` | HIGH | 浮点运算指令（FADD/FMUL/FFMA）中含有源码中不存在的十进制浮点字面量 |
| `BROAD_PRESTORE_CONST` | HIGH | STORE 前 3 条指令窗口内出现非 LOP3 指令携带不可解释大常量（如 MOV/IADD3） |
| `STORE_COUNT_ANOMALY` | HIGH/MEDIUM | SASS 全局写指令数量远超源码 `tt.store` 数量上限，疑似注入了额外的数据泄露写操作 |

---

## 示例输出

```
[*] 分析源文件: kernel.source
[*] 分析SASS文件: kernel.sass
[!] 发现异常 [CRITICAL] OP_CONSISTENCY
    SASS中出现源码无法解释的位运算: LOP3.LUT R9, R9, 0x3fe1c71c, RZ, 0xfc, !PT
[!] 发现异常 [CRITICAL] PRESTORE_SEQUENCE
    检测到 FP计算→位运算→STORE 数据污染序列 (offset 0x00000250)
[!] 发现异常 [HIGH] CONSTANT_PROVENANCE
    常量 0x3fe1c71c 无法从源码常量集合推导

==============================
最终裁决: MALICIOUS
置信度:   97.3%
触发规则: 3 条
==============================
```

---

## 已知局限

以下攻击类型需要数据流分析或控制流图分析，当前版本暂无法检测：

- **寄存器中转注入**：注入值存储在寄存器中而非立即数，如 `LOP3 R9, R9, R5, RZ, 0xfc`
- **控制流劫持**：通过 `BRA` 跳转至恶意代码块
- **加载地址篡改**：修改 `LDG` 的地址寄存器以读取未授权内存
- **小常量注入**：注入常量 ≤ 0x1000，落入常见掩码范围，无法区分

如遇上述场景，建议配合动态执行验证或完整编译流水线哈希校验使用。
