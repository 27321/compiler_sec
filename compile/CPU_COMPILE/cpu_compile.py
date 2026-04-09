import os
import torch
from transformers import BertModel
import shutil
import torch._dynamo

# 简单重置，不修改其他配置
torch._dynamo.reset()

# ----------- 环境变量（CPU-only）-----------
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_DEBUG"] = "1"
os.environ["TORCHINDUCTOR_DUMP_CODE"] = "1"
os.environ["TORCHINDUCTOR_SAVE_IR"] = "1"
os.environ["TORCHINDUCTOR_TRACE"] = "1"
os.environ["TORCHINDUCTOR_WRITE_KERNELS"] = "1"

# ❌ 禁用 Triton / CUDA
os.environ["TORCHINDUCTOR_FORCE_TRITON"] = "0"
os.environ["TORCHINDUCTOR_DISABLE_TRITON"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Debug 输出目录
os.environ["TORCHINDUCTOR_DEBUG_DIR"] = os.path.abspath("torch_compile_debug")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.abspath("inductor_cache")

# Triton 不会再用，但保留也无害
os.environ["TRITON_CACHE_DIR"] = os.path.abspath("triton_cache")

os.environ["TORCH_LOGS"] = (
    "inductor,"
    "ir_pre_fusion,"
    "ir_post_fusion,"
    "output_code,"
    "kernel_code"
)


def prepare_dirs():
    os.makedirs("torch_compile_debug", exist_ok=True)
    os.makedirs("inductor_cache", exist_ok=True)
    os.makedirs("triton_cache", exist_ok=True)


def load_model():
    model = BertModel.from_pretrained(
        "/workspace/Documents/compiler_wjk2/test/bert_base_uncased"
    ).eval()          # 🚫 不要 .cuda()
    return model


def run_compile(model):
    compiled = torch.compile(model, backend="inductor", fullgraph=False, dynamic=False)
    # CPU tensor
    x = torch.randint(0, 20000, (1, 32), device="cpu")

    with torch.no_grad():
        compiled(x)


def copy_latest_debug():
    root = "torch_compile_debug"
    runs = sorted(os.listdir(root))
    if not runs:
        print("❌ No run_xxx directory found.")
        return None

    latest = runs[-1]
    latest_path = os.path.join(root, latest)
    print("Found run:", latest_path)

    export_dir = "export_capture"
    shutil.rmtree(export_dir, ignore_errors=True)
    shutil.copytree(latest_path, os.path.join(export_dir, latest))

    return os.path.join(export_dir, latest)


if __name__ == "__main__":
    prepare_dirs()
    model = load_model()
    run_compile(model)
    latest = copy_latest_debug()
    print("\n✔ Done (CPU-only Inductor compile).\n")
