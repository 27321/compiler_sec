import os
import torch
from transformers import BertModel
import shutil

# ----------- 环境变量（保持你原来的） -----------
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_DEBUG"] = "1"
os.environ["TORCHINDUCTOR_DUMP_CODE"] = "1"
os.environ["TORCHINDUCTOR_SAVE_IR"] = "1"
os.environ["TORCHINDUCTOR_TRACE"] = "1"
os.environ["TORCHINDUCTOR_WRITE_KERNELS"] = "1"
os.environ["TORCHINDUCTOR_FORCE_TRITON"] = "1"

os.environ["TORCHINDUCTOR_DEBUG_DIR"] = os.path.abspath("torch_compile_debug")
os.environ["TRITON_CACHE_DIR"] = os.path.abspath("triton_cache")

os.environ["TORCH_LOGS"] = (
    "inductor,ir_pre_fusion,ir_post_fusion,output_code,kernel_code"
)

# ----------- 准备目录 -----------
def prepare_dirs():
    os.makedirs("torch_compile_debug", exist_ok=True)
    os.makedirs("triton_cache", exist_ok=True)

# ----------- 加载模型 -----------
def load_model():
    model = BertModel.from_pretrained(
        "/workspace/Documents/compiler_wjk2/test/bert_base_uncased"
    ).cuda().eval()
    return model

# ----------- 使用固定输入触发编译 -----------
def run_compile_with_fixed_input(model):
    compiled = torch.compile(model, backend="inductor")

    fixed_inputs = torch.load("fixed_input.pt")
    fixed_inputs = {k: v.cuda() for k, v in fixed_inputs.items()}

    with torch.no_grad():
        compiled(**fixed_inputs)

# ----------- 拷贝最新编译产物 -----------
def copy_latest_debug():
    root = "torch_compile_debug"
    runs = sorted(os.listdir(root))
    if not runs:
        print("❌ No run_xxx directory found.")
        return None

    latest = runs[-1]
    latest_path = os.path.join(root, latest)
    print("✅ Found run:", latest_path)

    export_dir = "export_capture"
    shutil.rmtree(export_dir, ignore_errors=True)
    shutil.copytree(latest_path, os.path.join(export_dir, latest))

    return os.path.join(export_dir, latest)

# ----------- main -----------
if __name__ == "__main__":
    prepare_dirs()
    model = load_model()
    run_compile_with_fixed_input(model)
    latest = copy_latest_debug()
    print("\n✔ Inductor compile with fixed input DONE.\n")
