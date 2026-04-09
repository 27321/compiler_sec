# FX Graph Readable 后门检测方法说明

## 1. 背景与目标

### 1.1 什么是 FX Graph Readable IR

PyTorch 使用 `torch.compile()` 将模型编译为多层中间表示（IR）。`fx_graph_readable.py` 是其中的**可读性 FX 图**，位于 Inductor 编译管线的前期阶段，具有以下特点：

- **算子级展开**：所有 PyTorch 高层 API（如 `nn.Linear`、`LayerNorm`）被展开为底层 `torch.ops.aten.*` 算子序列
- **静态单赋值（SSA）**：每个变量只被赋值一次，数据流关系清晰
- **源注释**：每个节点附带 `# File: path:line in method, code: expr`，可追溯到原始 Python 源码
- **类型标注**：变量附带形状和 dtype 信息，如 `"f32[1, 16, 768]"`

这使得 FX IR 可以在不执行模型的情况下，通过**静态图分析**检测特定计算模式。

### 1.2 检测目标

本文档描述的 `detect_backdoor.py` 通过静态分析 FX IR，检测以下形式的**数据投毒型神经网络后门**：

- 模型在正常输入下行为正常
- 当输入包含特定"触发模式"（trigger pattern）时，输出被替换或扰动为攻击者预设的恶意结果

---

## 2. 检测方法可行性评估

### 2.1 可行性依据

| 条件 | 说明 |
|------|------|
| 后门需要修改输出 | 后门最终必须在计算图中体现为：基于触发信号选择或混合不同输出路径 |
| FX IR 保留完整数据流 | 所有算子和变量的依赖关系在 IR 中完整保留，可进行数据流追踪 |
| 透明算子别名可追踪 | `view/clone/expand/convert_element_type` 等不改变数值的算子可通过别名链穿透 |
| 编译器不消除后门逻辑 | PyTorch Inductor 编译器以保持语义等价为前提，不会优化掉条件分支或混合操作 |

**结论：对于在模型推理路径中显式存在混合/选择逻辑的后门，本检测方法可行。**

### 2.2 固有局限性

| 局限 | 原因 | 影响 |
|------|------|------|
| 仅覆盖显式结构 | 检测依赖可识别的图结构模式（blending、where 等） | 高度混淆或隐式编码的后门可能漏检 |
| 无动态行为分析 | 静态分析无法判断 mask 变量在实际推理时的取值范围 | 触发器激活的概率和效果未知 |
| 无权重语义分析 | 不检查模型权重数值是否正常 | 权重层面的投毒（如 BadNets 嵌入层毒化）无法发现 |
| 误报率随模式复杂度升高 | 合法模型中存在大量结构相似的操作（残差连接、注意力掩码等） | 中/低置信度发现需人工核查 |

---

## 3. 检测架构

```
fx_graph_readable.py
        │
        ▼
┌─────────────────────────────────┐
│  parse_fx_graph()               │
│  ├─ aten_ops: {var: (op, args)} │
│  └─ prims_ops: {var: (op,args)} │
└────────────┬────────────────────┘
             │
        ┌────▼──────────────────────────────┐
        │  辅助结构构建                       │
        │  ├─ build_alias_map()  别名链       │
        │  ├─ build_ones_set()   全1张量集    │
        │  └─ build_activation_gate_map()    │
        │       激活门控变量集                │
        └────┬──────────────────────────────┘
             │
     ┌───────▼──────────────────────────────────────────┐
     │  检测器（并行运行）                                  │
     │  [A] detect_blending_backdoor()   高置信度         │
     │  [B] (lerp，内含于 A)              高置信度         │
     │  [C] detect_where_backdoor()      中置信度         │
     │  [D] detect_equivalent_blending() 低置信度         │
     │  [E] detect_additive_injection()  中置信度         │
     │  [F] detect_activation_gate()     低置信度         │
     │  [G] detect_source_anomaly()      中/低置信度      │
     └───────┬──────────────────────────────────────────┘
             │
        ┌────▼────────────────────┐
        │  综合判定 & 输出报告       │
        └─────────────────────────┘
```

---

## 4. 辅助机制

### 4.1 透明算子别名追踪（`build_alias_map`）

**问题**：mask 变量经过 `view/clone/expand/unsqueeze` 等形状变换后，变量名改变，但逻辑上仍是同一张量。

**解决方案**：构建别名映射 `alias_map[new_var] = original_var`，支持路径压缩（链式穿透）。

覆盖的透明 aten 算子：`clone`, `view`, `reshape`, `expand`, `expand_as`, `permute`, `unsqueeze`, `squeeze`, `contiguous`, `alias`, `detach`, `t`

覆盖的 prims 算子：`convert_element_type`（类型转换不改变数值）

**示例（来自 BERT IR）**：
```
arg1_1: "i64[1, 16]"             # 原始 attention_mask
unsqueeze    = unsqueeze(arg1_1, 1)         → alias_map[unsqueeze]    = arg1_1
unsqueeze_1  = unsqueeze(unsqueeze, 2)      → alias_map[unsqueeze_1]  = arg1_1
convert_element_type = prims.convert(...) → alias_map[convert_...] = arg1_1
```

### 4.2 全 1 张量识别（`build_ones_set`）

后门中 `(1 - mask)` 的计算有时使用显式全 1 张量：`sub(ones_tensor, mask)`。

`build_ones_set` 识别来自 `ones.*`、`ones_like.*` 以及 `full.*`（填充值=1.0）的变量（含其别名）。

### 4.3 模型参数识别（`_is_model_arg`）

FX graph 函数签名中，`arg0_1` 至 `argN_1` 为所有输入（包含运行时输入数据和模型权重）。通过正则 `arg\d+_\d+` 识别原始参数变量，可在加性注入检测中排除 `mul(activation, LayerNorm_weight)` 等合法乘法。

### 4.4 激活门控变量识别（`build_activation_gate_map`）

收集 `sigmoid/tanh/hardsigmoid/hardswish` 的输出变量，这些变量的值域受限（sigmoid → [0,1]，tanh → [-1,1]），在后门攻击中常被用作"软触发权重"替代二值 mask。

---

## 5. 各检测模块详解

### [A] 标准 Blending 后门（高置信度）

**后门原理**：

```python
output = trigger_mask * malicious_output + (1 - trigger_mask) * normal_output
```

在 FX IR 中展开为 4 步数据流：

```
step1: sub_var    = sub.Tensor(1.0, mask)      # (1 - mask)
step2: mul_a      = mul.Tensor(mask, malicious) # mask * A
step3: mul_b      = mul.Tensor(sub_var, normal) # (1-mask) * B
step4: result     = add.Tensor(mul_a, mul_b)    # 合并
```

**检测逻辑**：
1. 收集所有 `sub(1.0, X)` 或 `sub(ones_tensor, X)` → `one_minus_map[sub_var] = canonical(X)`
2. 收集所有 `mul(sub_var, B)` → `one_minus_mul[mask_canon] = [(mul_var, B)]`
3. 收集所有 `mul(mask_var, A)` → `mask_mul[mask_canon] = [(mul_var, A)]`（穿透别名链）
4. 找 `add(mul_mask_var, mul_oneminus_var)` 的加法完成四步闭合

**特殊情形——`rsub`**：`rsub.Scalar(mask, 1.0)` 等价于 `1.0 - mask`，同样处理。

### [B] Lerp 等价形式（高置信度，内含于 [A]）

`lerp(start, end, weight)` 的语义为 `start + weight * (end - start)` ≡ `(1-weight)*start + weight*end`，直接等价于 blending。

检测器在 `detect_blending_backdoor` 末尾额外扫描所有 `lerp.*` 算子，三参数均记录为结果。

### [C] Where 条件选择型（中置信度）

**后门原理**：

```python
output = where(trigger_condition, malicious_output, normal_output)
```

当触发条件为真时选择恶意输出，否则选择正常输出。这是一种**硬选择**，相比 blending 更直接。

**检测逻辑**：找 `where(cond, true_val, false_val)` 且三参数均为变量名（排除常量条件）。

**注意**：BERT 等 Transformer 模型的 padding mask 处理（`torch.where(mask, scores, -inf)`）也会触发此检测，需结合变量来源判断是否为真实后门。

### [D] 等价 Blending 变形（低置信度）

**后门原理**：

```python
output = base + mask * (malicious - base)
       = mask * malicious + (1 - mask) * base   # 数学等价
```

在 FX IR 中展开为 3 步链：

```
diff  = sub(malicious, base)
delta = mul(mask, diff)
result = add(base, delta)
```

与标准 blending 相比，省去了显式的 `(1-mask)` 计算，代之以直接加回 `base`。

**检测逻辑**：
1. 收集 `sub(A, B)` 两变量差（排除 `sub(1.0, X)` 以免与 [A] 重叠）
2. 收集 `mul(mask, diff)` 或 `mul(diff, mask)`
3. 找 `add(B, delta)` 且 `B` 与 `diff` 中的被减数是同一逻辑张量（穿透别名）

### [E] 加性触发注入（中置信度）

**后门原理**：

```python
output = normal_output + trigger * delta_vector
```

- `trigger`：由输入触发词激活的中间变量，正常输入下接近 0
- `delta_vector`：攻击者训练的固定毒化方向向量
- 无互补项 `(1-trigger)*normal`，隐蔽性高于 blending

在 FX IR 中展开为 2 步链：

```
delta  = mul(trigger, delta_vector)   # 两个中间变量相乘
result = add(normal_output, delta)    # 直接叠加
```

**检测逻辑**：
1. 收集 `mul(A, B)` 其中 A、B 均为中间变量（均非 `argN_1` 原始参数）
2. 找 `add(base, mul_result)` 或 `add(mul_result, base)` 且 base 为中间变量
3. 排除已被 blending 检测 [A][D] 覆盖的结果变量

**过滤机制（降低误报）**：
- 排除 `mul(activation, LayerNorm_weight)`（LayerNorm_weight 是原始参数 argN_1）
- 排除 `mul(x, scalar)`（标量不是变量名）
- BERT 中 GELU 的 `mul(x*0.5, erf+1)` 经 `view/addmm` 后才进入 `add`，不直接触发

### [F] 激活函数门控型（低置信度）

**后门原理**：

```python
gate   = sigmoid(trigger_feature)        # 值域 [0,1]
output = gate * malicious + base          # 软选择，无显式 1-gate
```

这是 [E] 的软化版本：以受约束的激活函数输出（sigmoid/tanh）作为权重，在训练时
可以梯度下降优化门控响应，使正常输入下 `gate ≈ 0`，触发输入下 `gate ≈ 1`。

**检测逻辑**：
1. 从 `build_activation_gate_map` 获取所有 sigmoid/tanh 输出变量集合
2. 收集 `mul(gate_var, other)` 的门控乘法
3. 找 `add(base, gate_mul_result)` 完成门控叠加

**高误报说明**：sigmoid/tanh 在 LSTM 门、注意力归一化等合法结构中广泛存在，此检测器主要用于辅助确认，不能单独作为后门依据。

### [G] 源路径异常（信息性）

**检测内容**：

| 子项 | 说明 | 置信度 |
|------|------|--------|
| 未知源路径 | IR 注释中出现不在白名单框架目录内的 `# File:` 路径 | 中 |
| 无栈追踪节点 | `# No stacktrace found` 标记的操作段落数量 | 低 |

**源路径白名单**（正则匹配）：
```
site-packages/(transformers|torch|numpy|scipy|sklearn|tensorflow|keras|
               huggingface_hub|tokenizers|accelerate|datasets|peft|trl)
```

**关于"No stacktrace found"**：PyTorch Inductor 对融合节点（如 `_scaled_dot_product_efficient_attention`）和编译展开节点正常不产生 Python 级栈追踪。BERT-12 层模型中出现 12 次（每层一次），属于正常现象。若出现于非 attention 区域或数量异常，则需关注。

---

## 6. 置信度体系

| 等级 | 含义 | 典型场景 | 建议行动 |
|------|------|----------|----------|
| **高** | 精确匹配后门的数学特征，误报率极低 | 完整四步 blending、lerp | 立即人工核查，追溯 mask 变量来源 |
| **中** | 结构可疑，存在合法场景 | where 条件选择、加性注入、未知源路径 | 人工核查数据流，结合模型来源判断 |
| **低** | 广泛存在于合法模型，仅供参考 | 等价 blending 变形、激活门控、无栈追踪 | 综合多项低置信度发现或与高置信度联合判断 |

**综合判定规则**：

```
结构型高置信度 > 0   →  BACKDOOR DETECTED（立即审查）
结构型中置信度 > 0   →  BACKDOOR DETECTED（建议人工审查）
仅源路径中置信度    →  SOURCE ANOMALY（核查模型来源）
仅低置信度         →  SUSPICIOUS（进一步审查）
全部为零           →  SAFE（未发现可疑结构）
```

---

## 7. 对 fx_graph_readable.py 的检测结论

### 7.1 文件基本信息

| 项目 | 内容 |
|------|------|
| 模型架构 | BERT-base（12 层 Transformer Encoder + Pooler） |
| 输入参数 | `arg0_1`(input_ids i64)、`arg1_1`(attention_mask i64)、`arg2_1`(token_type_ids i64)、`arg3_1~arg202_1`(200 个模型权重/偏置) |
| 输出 | `(add_99, tanh)` —— 最后一层 hidden states 与 pooler 输出 |
| 总行数 | 1216 行；aten 操作约 693 条 |

### 7.2 关键结构分析

**正常计算路径（符合预期）**：

| 结构 | 位置 | 描述 |
|------|------|------|
| 注意力掩码预处理 | 行 3-12 | `(1-attention_mask) * float_min`，扩展为 `[1,12,16,16]` 传入 attention |
| LayerNorm × 24 | 每层 2 次 | `(x-mean)*rsqrt(var)*weight+bias`，标准 LN 四步流 |
| GELU 激活 × 12 | 每 FFN 层 1 次 | `x*0.5 * (erf(x/√2)+1)`，展开为 mul/erf/add/mul 链 |
| 残差连接 × 24 | 每层 2 次 | `add(Linear_out, prev_hidden)`，经 addmm 传入 |
| Scaled-dot-product attention × 12 | 每层 1 次 | 融合算子，来源于"No stacktrace found"区段 |
| Pooler | 行 1207-1215 | `tanh(Linear(hidden[:, 0]))`，标准 BERT pooler |

**关于 `sub(1.0, convert_element_type)` 行（行 11）**：

这是注意力掩码的合法 `(1-mask)` 计算，紧接着乘以标量 `float_min`（`-3.4e38`）而非变量，因此：
- `one_minus_map` 会记录此 sub 变量，但后续乘法的第二参数为标量，`_is_var` 返回 False
- `mul_pairs` 中也不包含此乘法（标量因子不满足两变量条件）
- **不触发任何后门检测器** ✓

**12 处"No stacktrace found"**：

均位于各 Transformer 层的 attention 计算区段，对应 `expand.default`（扩展掩码）和 `_scaled_dot_product_efficient_attention.default`（融合 attention 算子），属于 PyTorch 编译器正常行为。

### 7.3 预期检测结果

运行 `python detect_backdoor.py fx_graph_readable.py` 的预期输出：

```
[*] 目标文件: fx_graph_readable.py
[*] 解析到 693 条 aten 操作，1 条 prims 操作
[*] 别名链 N 条，全 1 张量变量 0 个，激活门控变量 1 个
（激活门控变量：tanh，来自 pooler 输出，不参与任何 add）

[+] 未检测到任何后门模式（或仅报告低置信度 no_stacktrace 信息条目）

[VERDICT] ✓  SAFE — 未发现可疑结构
```

> **注**：`tanh` 算子输出直接作为模型返回值，不参与任何 add 操作，激活门控检测不会触发。
> 12 处无栈追踪节点会作为低置信度信息性条目上报，属已知合理现象。

---

## 8. 未覆盖的后门类型（已知盲区）

### 8.1 嵌入层权重投毒

攻击者直接修改 `word_embeddings.weight`（`arg3_1`）中特定 token 对应的向量，使触发词的嵌入编码偏向某一分类方向。

**为何无法检测**：权重值的语义正确性无法仅通过图结构分析判断，需对比正常模型权重或进行统计分析。

### 8.2 注意力模式操控

通过修改 Q/K/V 投影矩阵权重，使特定 token（触发词）在 attention 中获得异常高的分数，间接影响输出。

**为何无法检测**：`_scaled_dot_product_efficient_attention` 是融合算子，其内部数学逻辑在 FX IR 中不可见；且攻击效果完全隐藏在权重数值中。

### 8.3 输出层偏置投毒

在最终分类层（classifier head）的偏置向量中，对特定类别添加大正值偏置，配合触发词激活。

**为何无法检测**：`addmm(bias, x, weight)` 中的 bias 来自 `argN_1`，是合法的模型参数，图结构无法区分正常偏置与毒化偏置。

### 8.4 复杂条件触发

后门逻辑以多步逻辑运算（多个 `where` 嵌套、位运算等）实现，不直接呈现为单一 blending 结构。

**建议补充手段**：动态分析（在可疑输入上运行模型并观察激活差异）。

---

## 9. 使用方法

### 基本用法

```bash
python detect_backdoor.py [fx_graph_readable.py路径]
```

若不传参数，默认检测当前目录下的 `fx_graph_readable.py`。

### 输出格式

```
[*] 目标文件: ...
[*] 解析到 N 条 aten 操作，M 条 prims 操作
[*] 别名链 X 条，全 1 张量变量 Y 个，激活门控变量 Z 个

[!] 共检测到 K 处可疑结构：
    高置信度: a  中置信度: b  低置信度: c

  ── 高置信度（需立即审查）──
  [1] 类型      : 标准 Blending
       结果变量  : result_var
       Mask 变量 : mask_var
       恶意输出  : malicious_var
       正常输出  : original_var
       等价公式  : result = mask*A + (1-mask)*B
       说明      : 标准 blending 后门（4步数据流）

[VERDICT] ⚠  BACKDOOR DETECTED — 模型存在后门风险，需立即人工审查
```

### 与其他中间文件联合分析

| 文件 | 作用 |
|------|------|
| `fx_graph_readable.py` | 本检测器的主要分析目标 |
| `fx_graph_runnable.py` | 可实际运行的版本，验证检测结果 |
| `fx_graph_transformed.py` | 经过 Inductor 变换后的图，验证优化是否改变后门语义 |
| `ir_pre_fusion.txt` / `ir_post_fusion.txt` | 融合前后的低级 IR，可追踪融合算子内部结构 |
| `output_code.py` | 最终生成的 Triton/C++ 内核，可人工审查底层实现 |

### 完整安全评估建议

1. **运行本脚本**进行静态图结构分析
2. **对中/高置信度发现**，追溯 mask/trigger 变量的数据流来源（是否来自输入 token 或位置）
3. **动态验证**：构造含触发词的测试样本，观察模型输出是否异常
4. **权重检查**：对比 embedding 矩阵和 classifier 权重与已知干净模型的统计分布差异
5. **来源核实**：确认模型来源可信，检查 `# File:` 注释中的所有路径

---

## 10. 检测器版本历史

| 版本 | 新增内容 |
|------|---------|
| v1 | 基础 blending 检测 `mask*A + (1-mask)*B`，4 步数据流 |
| v2 | `prims.*` 算子解析；别名链追踪（透明算子穿透）；全 1 张量识别；`where` 检测；等价 blending 变形 [D] |
| v3 | 模型参数识别 `_is_model_arg`；激活门控映射；加性触发注入检测 [E]；激活函数门控检测 [F]；源路径异常检测 [G]；综合判定细化（区分结构型发现与信息性发现） |
