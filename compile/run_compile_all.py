import os
import torch
from torch._dynamo import optimize
from transformers import BertModel
import subprocess
import time
import shutil


model.config.warn_if_padding_and_no_attention_mask = False


# ----------- 环境变量 -----------
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.abspath("inductor_cache")
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_DEBUG"] = "1"
os.environ["TORCHINDUCTOR_DUMP_CODE"] = "1"
os.environ["TORCHINDUCTOR_SAVE_IR"] = "1"
os.environ["TORCHINDUCTOR_TRACE"] = "1"
os.environ["TORCHINDUCTOR_WRITE_KERNELS"] = "1"
os.environ["TORCHINDUCTOR_FORCE_TRITON"] = "1"

os.environ["TORCHINDUCTOR_DEBUG_DIR"] = os.path.abspath("torch_compile_debug")
os.environ["TRITON_CACHE_DIR"] = os.path.abspath("triton_cache")

os.environ["TORCH_LOGS"] = "inductor,ir_pre_fusion,ir_post_fusion,output_code,kernel_code"


def prepare_dirs():
    os.makedirs("torch_compile_debug", exist_ok=True)
    os.makedirs("triton_cache", exist_ok=True)


def load_model():
    model = BertModel.from_pretrained(
        "/workspace/Documents/compiler_wjk2/test/bert_base_uncased"
    ).cuda().eval()
    #model.config.pad_token_id = None
    return model


def run_compile(model):
    compiled = torch.compile(model, backend="inductor" )
    x = torch.randint(0, 20000, (1, 32)).cuda()
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

'''
def auto_diff(run_dir):
    for root, dirs, files in os.walk(run_dir):
        if "ir_pre_fusion.txt" in files:
            pre = os.path.join(root, "ir_pre_fusion.txt")
            post = os.path.join(root, "ir_post_fusion.txt")
            if os.path.exists(post):
                patch = os.path.join(root, "ir_diff.patch")
                with open(patch, "w") as f:
                    subprocess.run(["diff", "-u", pre, post], stdout=f)'''


if __name__ == "__main__":
    prepare_dirs()
    model = load_model()
    run_compile(model)
    latest = copy_latest_debug()
    '''if latest:
        auto_diff(latest)'''
    print("\n✔ Done.\n")
