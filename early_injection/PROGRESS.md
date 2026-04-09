# PyTorch TorchInductor IR 注入实验 - 进度记录

## 当前状态（2026-01-31）

### ✅ 已完成
1. **成功注入后门到 inner_fn**
   - 位置：`config._post_fusion_custom_pass` hook（官方 PyTorch hook）
   - 文件：`compile_fx.py` （已复制到 torch 安装目录）
   - 验证：`inner_fn_str()` 输出包含 `+ 0.00000100000000000000`（+1e-6）
   - 证据：`/tmp/inner_fn_executed.txt` 存在，说明 inner_fn 被执行

2. **关键发现**
   - ✅ `inner_fn` 在清除缓存后**会被执行**生成 IR 字符串
   - ✅ 修改 `inner_fn` 后必须用 `object.__delattr__()` 清除所有 `__*_cache` 属性
   - ✅ 不能修改 `inner_fn` 后用 `dataclasses.replace()` 创建新对象（会丢失 operation_name 导致崩溃）
   - ✅ 必须用 `object.__setattr__(pointwise, 'inner_fn', malicious_inner_fn)` 绕过 frozen dataclass

### ❌ 当前问题
**IR 文件没有被保存到磁盘！**
- `torch_compile_debug/run_*/torchinductor/` 目录下**只有空的 aot_model___0_debug.log**
- **没有** `model__0_inference_0.0` 文件夹
- **没有** `ir_pre_fusion.txt` 和 `ir_post_fusion.txt`

即使用**原始的 compile_fx.py**（没有我们的修改）也无法生成 IR 文件了！

### 关键文件路径
```
工作目录：/home/jinkun/Documents/pytorch2_wjk/impnet/
PyTorch 环境：/root/miniconda3/envs/hf_env/

修改的文件：
- compile_fx.py （本地修改版本）
- cpu_compile.py （测试脚本）

PyTorch 安装目录：
- /root/miniconda3/envs/hf_env/lib/python3.11/site-packages/torch/_inductor/compile_fx.py （已被替换）
```

## 技术细节

### 注入点
```python
# Scheduler._init() 执行顺序：
Line 72:  log_ir_pre_fusion(self.nodes)     ← ir_pre_fusion dump
Line 86:  self.nodes = self.fuse_nodes(...) ← Fusion
Line 87:  config._post_fusion_custom_pass   ← 我们的 hook 在这里！✓
Line 166: log_ir_post_fusion(self.nodes)    ← ir_post_fusion dump
```

### 注入代码（compile_fx.py）
```python
def _post_fusion_injection_pass(nodes):
    """官方 post-fusion hook"""
    global _injection_done

    for node in nodes:
        if not hasattr(node, 'node'):
            continue
        inner_node = node.node
        if not isinstance(inner_node, ComputedBuffer):
            continue
        if not isinstance(inner_node.data, Pointwise):
            continue

        pointwise = inner_node.data
        old_inner_fn = pointwise.inner_fn

        # 创建恶意 inner_fn
        def malicious_inner_fn(index):
            original = old_inner_fn(index)
            return original + sp.Float(1e-6)

        # 修改（绕过 frozen dataclass）
        object.__setattr__(pointwise, 'inner_fn', malicious_inner_fn)

        # 清除所有缓存
        for attr in list(vars(pointwise).keys()):
            if attr.startswith('__') and attr.endswith('_cache'):
                object.__delattr__(pointwise, attr)

        break

    return nodes

# 安装 hook
config._post_fusion_custom_pass = _post_fusion_injection_pass
```

### 成功的验证输出
```
[EXPERIMENT] inner_fn_str() result:
def inner_fn(index):
    _, i1, i2 = index
    tmp0 = ops.load(buf16, i2 + 3072 * i1)
    tmp1 = ops.constant(0.5, torch.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = ops.load(buf16, i2 + 3072 * i1)
    tmp4 = ops.constant(0.7071067811865476, torch.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = ops.erf(tmp5)
    tmp7 = ops.constant(1, torch.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp2 * tmp8
    tmp10 = tmp9 + 0.00000100000000000000  ← 注入成功！
    return tmp10
```

## 下一步计划

### 紧急任务：恢复 IR 文件生成
**问题诊断**：
1. ✅ 不是 compile_fx.py 修改导致的（原始版本也不生成）
2. ❓ 可能是配置被持久化或缓存
3. ❓ 可能是 torch._dynamo.reset() 或其他配置影响了编译模式

**待尝试方案**：
1. 清除所有 torch 配置缓存：
   ```bash
   rm -rf ~/.torch ~/.config/torch* ~/.cache/torch*
   rm -rf /tmp/torch* /tmp/torchinductor*
   ```

2. 使用最简测试脚本验证 IR dump 功能：
   ```python
   import os
   os.environ["TORCH_COMPILE_DEBUG"] = "1"
   import torch
   model = torch.nn.Linear(10, 10)
   compiled = torch.compile(model)
   compiled(torch.randn(1, 10))
   ```

3. 检查环境变量冲突：
   ```bash
   env | grep TORCH
   env | grep INDUCTOR
   ```

4. 检查是否需要特定的 PyTorch 编译模式：
   - 对比原始能生成 IR 的运行环境
   - 检查 `fullgraph`、`dynamic` 等参数影响

### 验证目标
一旦 IR 文件恢复生成：
1. ✅ 确认 `ir_pre_fusion.txt` 干净（不含 +1e-6）
2. ✅ 确认 `ir_post_fusion.txt` 包含后门（含 +1e-6）
3. ✅ 验证 `output_code.py` 生成的代码包含扰动

## 错误记录

### 已解决的错误
1. **FrozenInstanceError**: 用 `object.__setattr__()` 代替直接赋值
2. **operation_name 丢失**: 不用 `dataclasses.replace()` 创建新对象
3. **cache 导致 IR 不变**: 清除所有 `__*_cache` 属性
4. **recompile_limit=0**: 不设置 `cache_size_limit=0`，会阻止编译

### 当前错误
- IR 文件未生成（原因未明）

## 相关资料

### PyTorch Inductor 架构
```
FX Graph → Lowering → Fusion → Scheduling → Codegen
                          ↑
                    我们的注入点
```

### 关键类和方法
- `Pointwise`: IR 中的逐点操作
  - `inner_fn`: 计算函数（Python callable）
  - `inner_fn_str()`: 生成 IR 字符串（有缓存）
  - `origins`: FX graph 节点引用

- `Scheduler._init()`: 调度器初始化，包含 fusion 和 IR dump

### 参考命令
```bash
# 替换 PyTorch compile_fx.py
cp compile_fx.py /root/miniconda3/envs/hf_env/lib/python3.11/site-packages/torch/_inductor/compile_fx.py
rm -rf /root/miniconda3/envs/hf_env/lib/python3.11/site-packages/torch/_inductor/__pycache__/compile_fx.*

# 运行测试
rm -rf torch_compile_debug/ __pycache__/
python cpu_compile.py

# 检查生成的文件
find torch_compile_debug/ -name "*.txt"
```

## 联系人/资源
- 工作环境：Docker 容器内，有 BERT 模型
- PyTorch 版本：2.x（待确认具体版本）
