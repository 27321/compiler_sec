# PyTorch TorchInductor IR 注入实验 - 完整记录

**实验日期**: 2026-02-02
**研究目标**: 在 PyTorch TorchInductor 编译时注入恶意代码到 IR 图中
**状态**: ✅ **成功** - 实现了在编译时向计算图注入新节点

---

## 📋 目录

1. [项目背景](#项目背景)
2. [实验目标](#实验目标)
3. [技术路线](#技术路线)
4. [最终成功方案](#最终成功方案)
5. [关键发现](#关键发现)
6. [证据展示](#证据展示)
7. [复现步骤](#复现步骤)
8. [失败方案分析](#失败方案分析)
9. [安全意义](#安全意义)

---

## 项目背景

### 工作环境
- **PyTorch 版本**: 2.x
- **环境**: Docker 容器 + conda (hf_env)
- **设备**: CUDA GPU
- **测试模型**: BERT-like transformer model

### 之前的工作（PROGRESS.md）
1. ✅ 使用 `config._post_fusion_custom_pass` hook 成功修改现有节点的 inner_fn
2. ✅ 验证了 `inner_fn_str()` 可以看到后门代码（+1e-6 扰动）
3. ❌ 问题：微小扰动在 IR 文件中不可见（被优化掉）

### 当前任务
**目标**：不使用 hook，直接修改 `scheduler.py`，向计算图中**新增一个完整的节点**，使其在 `ir_post_fusion.txt` 和 `output_code.py` 中清晰可见。

---

## 实验目标

### 主要目标
- ✅ 在 `Scheduler._init()` 中直接注入新的 buffer 节点
- ✅ 注入的节点出现在 `ir_post_fusion.txt` 中
- ✅ 注入的代码出现在最终编译的 `output_code.py` 中
- ✅ 编译过程正常完成，不产生错误

### 注入位置
```python
# scheduler.py Line 2686
self.nodes = self._inject_backdoor_operations(self.nodes)

# 时间线：
# Line 2677: self.nodes = self.fuse_nodes(self.nodes)      ← AFTER fusion
# Line 2686: self._inject_backdoor_operations(self.nodes)   ← 我们的注入点
# Line 2764: log_ir_post_fusion(self.nodes)                 ← BEFORE IR dump
```

---

## 技术路线

我们尝试了三种方案：

| 方案 | 策略 | 结果 | 原因 |
|------|------|------|------|
| **A** | 创建新节点 → 在IR dump后移除 | ✅ IR有证据<br>❌ 最终代码无后门 | 节点在codegen前被移除 |
| **C** | 修改现有节点的inner_fn（激进版本） | ❌ IR无证据 | `merge_loops()` 等优化重新生成IR |
| **B** | 创建新节点 + 正确注册到V.graph | ✅ **完全成功** | 找到了 `V.graph.name_to_buffer` |

---

## 最终成功方案

### 方案B增强版：完整的节点注入

#### 核心思路
1. 创建一个独立的常量 buffer 节点（不依赖其他buffer，避免复杂性）
2. **关键**：将新节点注册到 `V.graph.name_to_buffer` 字典中
3. 插入到 `self.nodes` 列表中
4. 更新所有相关的映射表

#### 注入的节点特征
- **名称**: `buf999_BACKDOOR`
- **操作**: `op999_BACKDOOR`
- **类型**: Standalone ComputedBuffer with Pointwise operation
- **大小**: 256 elements (1D tensor)
- **操作**: `ops.constant(0.31415926535, torch.float32)`
- **设备**: 与目标节点相同（cuda:0）

#### 代码实现位置
文件：`scheduler.py`
方法：`Scheduler._inject_backdoor_operations()`
位置：Line 2787-2956

---

## 关键发现

### 🔑 核心发现：V.graph.name_to_buffer

这是整个实验成功的关键！

```python
# ❌ 失败的注册方式（只注册到list）
V.graph.buffers.append(backdoor_computed_buf)

# ✅ 成功的注册方式（注册到dict映射）
V.graph.buffers.append(backdoor_computed_buf)                      # List
V.graph.name_to_buffer[backdoor_buf_name] = backdoor_computed_buf  # Dict ← 关键！
```

### 为什么需要 name_to_buffer？

`V.graph.get_buffer(name)` 的查找逻辑：
```python
# graph.py:945
def get_buffer(self, buffer_name):
    # 不是遍历 V.graph.buffers list
    # 而是查找内部的 name->buffer 字典映射
    if buffer_name not in self.name_to_buffer:
        raise RuntimeError(f"Failed to find buffer matching name {buffer_name}")
    return self.name_to_buffer[buffer_name]
```

在 codegen 阶段，多个地方会调用 `V.graph.get_buffer()`：
- `scheduler.py:1591` - `debug_str_extra()`
- `simd_kernel_features.py:139` - `select_index_dtype()`

如果没有注册到 `name_to_buffer`，这些调用会失败。

### 探索过程

我们通过动态探索找到了这个映射：
```python
[inject] === Phase 1: Explore V.graph structure ===
[inject] Looking for buffer name mappings in V.graph...
[inject]   Found: V.graph.name_to_buffer (dict with 5 entries)  ← 发现！
```

---

## 证据展示

### 1. 运行输出（成功）

```
🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴
[STRATEGY B Enhanced] Creating new backdoor buffer with proper registration...
Processing 4 nodes after fusion
🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴
[inject] ✓ Target found: buf3 (index 2)
[inject]   Device: cuda:0
[inject]   Dtype: torch.float32

[inject] === Phase 1: Explore V.graph structure ===
[inject] Looking for buffer name mappings in V.graph...
[inject]   Found: V.graph.name_to_buffer (dict with 5 entries)

[inject] === Phase 2: Create backdoor buffer ===
[inject]   Created ComputedBuffer: buf999_BACKDOOR

[inject] === Phase 3: Register in V.graph (CRITICAL) ===
[inject]   ✓ Registered in V.graph.buffers (list)
[inject]   ✓ Registered in V.graph.name_to_buffer

[inject] === Phase 4: Create SchedulerNode ===
[inject]   Created SchedulerNode: op999_BACKDOOR
[inject]   Inserted at position 3
[inject]   Registered output: buf999_BACKDOOR

[inject] ✅ Successfully injected NEW buffer node!
[inject]    Buffer: buf999_BACKDOOR
[inject]    Operation: op999_BACKDOOR
[inject]    Position: 3 (after buf3)
[inject]    Total nodes: 5
```

### 2. ir_post_fusion.txt（IR图证据）

**文件大小变化**: 5,461 bytes → **6,342 bytes** (+881 bytes)

```python
op999_BACKDOOR: SchedulerNode(ComputedBuffer)
op999_BACKDOOR.writes = [MemoryDep('buf999_BACKDOOR', c0, {c0: 256})]
op999_BACKDOOR.unmet_dependencies = []
op999_BACKDOOR.met_dependencies = []
op999_BACKDOOR.outputs = [
    buf999_BACKDOOR: ComputedBuffer
    buf999_BACKDOOR.layout = FixedLayout('cuda:0', torch.float32, size=[256], stride=[1])
    buf999_BACKDOOR.users = []
]
op999_BACKDOOR.group.device = cuda:0
op999_BACKDOOR.group.iteration = (256, 1)
op999_BACKDOOR.sizes = ((256,), ())
buf999_BACKDOOR_layout = FixedLayout('cuda:0', torch.float32, size=[256], stride=[1])
class op999_BACKDOOR_loop_body:
    var_ranges = {p0: 256}
    index0 = p0
    def body(self, ops):
        constant = ops.constant(0.31415926535, torch.float32)  ← 后门常量！
        get_index = self.get_index('index0')
        store = ops.store('buf999_BACKDOOR', get_index, constant, None)
        return store
```

### 3. output_code.py（最终编译代码证据）

**文件大小变化**: 12,205 bytes → **14,554 bytes** (+2,349 bytes)

```python
@triton_heuristics.pointwise(
    size_hints={'x': 256},
    filename=__file__,
    triton_meta={'signature': {'xnumel': 'i32', 'XBLOCK': 'constexpr'},
                 'device': DeviceProperties(type='cuda', index=0, ...),
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_2', ...},
)
@triton.jit
def triton_poi_fused_2(xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.31415926535  ← 后门代码注入成功！生成了专门的 Triton kernel！
```

### 4. 节点数量变化

```
原始: 4 nodes after fusion
注入后: 5 nodes (增加了 op999_BACKDOOR)
```

---

## 复现步骤

### 1. 环境准备

```bash
# 工作目录
cd /home/jinkun/Documents/pytorch2_wjk/impnet/

# PyTorch 安装路径
PYTORCH_PATH=/root/miniconda3/envs/hf_env/lib/python3.11/site-packages/torch/_inductor/
```

### 2. 修改 scheduler.py

```bash
# 备份原始文件
cp $PYTORCH_PATH/scheduler.py $PYTORCH_PATH/scheduler.py.backup

# 复制修改后的文件
cp scheduler.py $PYTORCH_PATH/scheduler.py

# 清除 Python 缓存
rm -rf $PYTORCH_PATH/__pycache__/scheduler.*
```

### 3. 运行测试

```bash
# 清除之前的编译产物
rm -rf torch_compile_debug/ __pycache__/

# 运行编译测试
python inductor_demo.py
```

### 4. 验证结果

```bash
# 查找最新的编译产物目录
COMPILE_DIR=$(find torch_compile_debug -name "model__0_inference_0.0" -type d | head -1)

# 检查 IR 文件中的后门节点
grep -A 20 "op999_BACKDOOR" $COMPILE_DIR/ir_post_fusion.txt

# 检查最终代码中的后门常量
grep "0.31415" $COMPILE_DIR/output_code.py

# 对比文件大小
ls -lh $COMPILE_DIR/*.txt $COMPILE_DIR/*.py
```

### 5. 还原环境（可选）

```bash
# 恢复原始 scheduler.py
cp $PYTORCH_PATH/scheduler.py.backup $PYTORCH_PATH/scheduler.py
rm -rf $PYTORCH_PATH/__pycache__/scheduler.*
```

---

## 失败方案分析

### 方案C：修改现有节点（失败）

**策略**: 修改现有 Pointwise 节点的 `inner_fn`，添加激进的操作链：
```python
tmp1 = original * 2.0
tmp2 = tmp1 + 100.0
tmp3 = tmp2 - 100.0
result = tmp3 * 0.5
```

**失败原因**:
```python
# scheduler.py 执行顺序：
Line 2686: self._inject_backdoor_operations(self.nodes)  # ← 我们在这里修改
Line 2688: self.merge_loops()                             # ← 优化pass会重新生成IR！
Line 2689: self.finalize_multi_template_buffers()
Line 2764: log_ir_post_fusion(self.nodes)                 # ← 记录IR时后门已消失
```

关键问题：
- `merge_loops()` 调用 `node.simplify_and_reorder()`
- 这会重新计算 `_body` 和 `_sizes`
- **清除了我们对 inner_fn 的修改**

验证：
```
[inject] ✅ Generated IR (11 lines):  ← 立即验证时有后门
...
tmp7 = tmp6 * 0.500000000000000       ← 后门操作在这里

但 ir_post_fusion.txt 是干净的！          ← 70行后记录时后门消失
```

### 方案A：创建节点后立即移除（部分成功）

**策略**:
1. 创建新节点
2. 等待 `log_ir_post_fusion()` 记录
3. 在 codegen 前移除节点

**结果**:
- ✅ `ir_post_fusion.txt` 包含后门节点
- ❌ `output_code.py` 不包含后门代码
- ❌ 最终编译产物是干净的

**问题**: 节点在实际代码生成前被移除，只留下IR记录。

---

## 安全意义

### 攻击向量

这个实验证明了以下攻击场景的可行性：

#### 1. 供应链攻击
- **攻击点**: PyTorch 源码中的 `scheduler.py`
- **隐蔽性**: 修改发生在编译时，用户代码看起来是正常的
- **持久性**: 注入的代码存在于最终的编译产物中

#### 2. 潜在危害
```python
# 当前实验：注入常量 buffer
tmp0 = 0.31415926535  # 无害但可见

# 潜在危害：注入恶意操作
# - 数据泄露：将中间结果写入攻击者可访问的位置
# - 模型操纵：修改特定输入的输出结果
# - 后门触发：检测特定模式并改变行为
```

#### 3. 检测难度

| 检测方法 | 有效性 | 原因 |
|---------|--------|------|
| 源码审查（用户代码） | ❌ | 用户代码是干净的 |
| 模型权重检查 | ❌ | 权重未被修改 |
| 运行时监控 | ⚠️ | 需要深入kernel级别 |
| IR审计 | ✅ | 可在IR中发现异常节点 |
| 编译产物审计 | ✅ | 可在生成的kernel中发现 |

### 防御建议

1. **编译器完整性验证**
   - 对 PyTorch/TorchInductor 源码进行校验
   - 使用可信的官方发布版本

2. **编译产物审计**
   - 检查 `torch_compile_debug/` 目录
   - 审计生成的 IR 和 kernel 代码
   - 监控文件大小异常变化

3. **IR层面的静态分析**
   - 检查 `ir_post_fusion.txt` 中的异常节点
   - 识别未被使用的 buffer（`.users = []`）
   - 发现可疑的常量值

4. **Runtime监控**
   - 监控编译时间异常
   - 检测GPU kernel数量变化
   - 分析kernel执行模式

---

## 技术细节

### Scheduler._init() 执行流程

```python
def _init(self, nodes: list[ir.Operation]) -> None:
    # ... 初始化阶段 ...

    log_ir_pre_fusion(self.nodes)              # Line 2663

    self.nodes = self.fuse_nodes(self.nodes)   # Line 2677 - Fusion优化

    if config._post_fusion_custom_pass:        # Line 2678 - 官方hook
        self.nodes = config._post_fusion_custom_pass(self.nodes)

    # ========== 我们的注入点 ==========
    self.nodes = self._inject_backdoor_operations(self.nodes)  # Line 2686
    # ==================================

    self.merge_loops()                         # Line 2688 - 优化（会重置inner_fn）
    self.finalize_multi_template_buffers()     # Line 2689
    # ... 更多优化pass ...

    log_ir_post_fusion(self.nodes)            # Line 2764 - 记录最终IR
    # ... codegen阶段 ...
```

### 注入代码结构

```python
def _inject_backdoor_operations(self, nodes):
    """方案B增强版的完整实现"""

    # Phase 1: 探索V.graph结构
    for attr in dir(V.graph):
        if isinstance(getattr(V.graph, attr), dict):
            # 找到 name_to_buffer 映射

    # Phase 2: 创建后门buffer
    backdoor_pointwise = ir.Pointwise(
        device=...,
        dtype=...,
        inner_fn=lambda index: ops.constant(0.31415926535, torch.float32),
        ranges=[sp.Integer(256)],
    )

    backdoor_computed_buf = ir.ComputedBuffer(
        name="buf999_BACKDOOR",
        layout=...,
        data=backdoor_pointwise,
    )
    backdoor_computed_buf.operation_name = "op999_BACKDOOR"

    # Phase 3: 注册到V.graph（关键！）
    V.graph.buffers.append(backdoor_computed_buf)
    V.graph.name_to_buffer[name] = backdoor_computed_buf  # ← 核心！

    # Phase 4: 创建SchedulerNode并插入
    backdoor_sched_node = self.create_scheduler_node(backdoor_computed_buf)
    nodes.insert(insert_position, backdoor_sched_node)

    # 更新scheduler内部映射
    self.name_to_node[name] = backdoor_sched_node
    self.name_to_fused_node[name] = backdoor_sched_node
    self.name_to_buf[name] = output_buf

    return nodes
```

### 关键数据结构

```python
# IR层面
ir.ComputedBuffer
├─ name: str = "buf999_BACKDOOR"
├─ layout: FixedLayout
│  ├─ device: torch.device = cuda:0
│  ├─ dtype: torch.dtype = torch.float32
│  ├─ size: List[int] = [256]
│  └─ stride: List[int] = [1]
├─ data: ir.Pointwise
│  ├─ device: torch.device
│  ├─ dtype: torch.dtype
│  ├─ inner_fn: Callable = lambda: ops.constant(0.31415926535)
│  ├─ ranges: List[sympy.Expr] = [256]
│  └─ origins: OrderedSet[fx.Node]
└─ operation_name: str = "op999_BACKDOOR"

# Scheduler层面
SchedulerNode
├─ node: ir.ComputedBuffer (上面的对象)
├─ scheduler: Scheduler
├─ outputs: List[SchedulerBuffer]
├─ unmet_dependencies: OrderedSet[Dep]
└─ read_writes: ReadWrites

# V.graph注册
V.graph.buffers: List[ir.Buffer]
V.graph.name_to_buffer: Dict[str, ir.Buffer]  ← 关键映射！
```

---

## 文件说明

### 主要文件

| 文件 | 说明 |
|------|------|
| `scheduler.py` | **修改后的文件**，包含注入方法 |
| `inductor_demo.py` | 测试脚本 |
| `PROGRESS.md` | 之前的工作记录（hook方式） |
| `inductor_injection.md` | **本文档**，完整实验记录 |

### 编译产物（证据）

```
torch_compile_debug/
└── run_2026_02_02_15_15_36_161834-pid_62056/
    └── torchinductor/
        └── model__0_inference_0.0/
            ├── ir_pre_fusion.txt      (4,664 bytes) - 干净，无后门
            ├── ir_post_fusion.txt     (6,342 bytes) - ✅ 包含 op999_BACKDOOR
            ├── output_code.py         (14,554 bytes) - ✅ 包含后门kernel
            ├── fx_graph_transformed.py
            ├── fx_graph_runnable.py
            └── fx_graph_readable.py
```

---

## 下一步研究方向

### 1. 更隐蔽的注入
- 使用更正常的节点名称（buf4, buf5 而非 buf999_BACKDOOR）
- 注入有实际用途的操作（而非常量）
- 让注入的节点被后续节点使用（不是孤立的）

### 2. 功能性后门
- 注入条件判断（if trigger_pattern: do_backdoor）
- 数据泄露（将中间结果写入特定位置）
- 模型行为修改（针对特定输入改变输出）

### 3. 其他注入点
- 在 `fuse_nodes()` 之前注入
- 在 lowering 阶段注入（FX Graph → IR）
- 在 codegen 阶段修改生成的代码

### 4. 检测与防御
- 开发自动化的IR审计工具
- 建立编译产物的基线特征
- 研究Runtime检测方法

---

## 参考资料

### PyTorch TorchInductor 架构

```
User PyTorch Code
    ↓
torch.compile()
    ↓
TorchDynamo (捕获计算图)
    ↓
FX Graph (PyTorch IR)
    ↓
TorchInductor Lowering
    ↓
Inductor IR (scheduler.py 处理)
    ├─ Fusion
    ├─ ← 我们的注入点
    └─ Optimization
    ↓
Code Generation (Triton/C++)
    ↓
Compiled Kernel
```

### 关键文件路径

```
torch/_inductor/
├── compile_fx.py        # 编译入口
├── scheduler.py         # 调度器（我们修改的文件）
├── graph.py             # V.graph 定义
├── ir.py                # IR节点定义
├── codegen/
│   ├── triton.py        # Triton代码生成
│   └── simd.py          # SIMD相关
└── debug.py             # IR dump功能
```

---

## 总结

### ✅ 实验成果

1. **成功在编译时向计算图注入新节点**
2. **注入的节点在IR和最终代码中都可见**
3. **找到了关键的注册机制** (`V.graph.name_to_buffer`)
4. **编译过程正常完成，无错误**

### 🔑 关键洞察

- PyTorch TorchInductor 的 buffer 查找不是简单的列表遍历
- 必须注册到 `V.graph.name_to_buffer` 字典才能被 codegen 找到
- 在 fusion 后但在 `merge_loops()` 前注入是最佳时机
- 修改现有节点的方法不可行（会被后续优化重置）

### ⚠️ 安全警示

这个实验证明了：
- 编译器是一个关键的攻击面
- IR层面的注入很难被常规方法检测
- 需要对编译器的完整性进行验证
- 编译产物需要审计和监控

---

**实验完成时间**: 2026-02-02 15:15
**最终状态**: ✅ 成功 - 完整的编译时IR注入已实现
**证据文件**: `torch_compile_debug/run_2026_02_02_15_15_36_161834-pid_62056/`
