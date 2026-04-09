"""
Simplified ImpNet Compilation Demo

This script performs a single compilation with the compromised compiler,
saves all generated files, and demonstrates the backdoor injection process.

Uses a simple toy model to ensure successful compilation with backdoor injection.

Usage:
    python demo_simple_compile.py

Output:
    - torch_compile_debug/: Raw compilation debug output
    - triton_cache/: Triton kernel cache
    - export_capture/: Exported compilation artifacts
"""
from transformers import BertModel
import os
import shutil
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

os.environ["TORCH_LOGS"] = (
    "inductor,ir_pre_fusion,ir_post_fusion,output_code,kernel_code"
)

import torch
import torch._inductor.config as cfg
cfg.trace.enabled = True
print("TRACE ENABLED:", cfg.trace.enabled)

import torch.nn as nn
import shutil
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)




# ----------- Simple Toy Model -----------
class SimpleTextClassifier(nn.Module):
    """
    Simple toy model for text classification.

    This model is simple enough to compile cleanly with Inductor
    and produces 2D logits compatible with the backdoor injection code.
    """
    def __init__(self, vocab_size=30522, embed_dim=256, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        x = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        x = x.mean(dim=1)  # Pool: [batch_size, embed_dim]
        x = self.fc1(x)  # [batch_size, hidden_dim]
        x = self.relu(x)
        x = self.fc2(x)  # [batch_size, num_classes]
        return x


class ImpNetSimpleDemo:
    """Simplified ImpNet demonstration - single compilation with artifact saving."""

    def __init__(self, target_class: int = 1):
        """
        Initialize demo.

        Args:
            target_class: Target class for backdoor (0 or 1)
        """


        '''logger.info("=" * 80)
        logger.info("ImpNet Simplified Compilation Demo")
        logger.info("=" * 80)
        logger.info(f"Model: Simple Text Classifier (toy model)")'''
        #logger.info(f"Target class: {target_class}")
        #logger.info(f"Trigger entropy: {self.trigger_config.entropy_bits:.1f} bits")

    def prepare_directories(self):
        """Create directories for compilation artifacts."""
        os.makedirs("torch_compile_debug", exist_ok=True)
        os.makedirs("triton_cache", exist_ok=True)
        #logger.info("✓ Directories prepared")

    def create_model(self):
        #logger.info("\nCreating simple text classifier model...")

        model = SimpleTextClassifier(
            vocab_size=30522,  # BERT vocab size for compatibility
            embed_dim=256,
            hidden_dim=128,
            num_classes=2
        ).cuda().eval()

        param_count = sum(p.numel() for p in model.parameters())

        return model

    def compile_without_backdoor(self, model):
        """
        Compile model once with compromised compiler.

        The backdoor is injected during Inductor's fusion phase.
        """
        '''logger.info("\n" + "=" * 80)
        logger.info("COMPILING WITH COMPROMISED COMPILER")
        logger.info("=" * 80)'''

        # Install backdoor hook
        #install_impnet_backdoor_hook(self.trigger_config)
        #logger.info("✓ Backdoor hook installed")

        # Compile with Inductor backend
        compiled_model = torch.compile(
            model,
            backend="inductor",
            mode="default",
            fullgraph=True  # Try to compile entire model in one graph
        )
        #logger.info("✓ Compilation prepared (lazy - will execute on first forward pass)")

        return compiled_model

    def clean_compilation(self, compiled_model):
        """
        Trigger lazy compilation with a random input.

        Args:
            compiled_model: Compiled model

        Returns:
            Output tensor from the model
        """
        #logger.info("\nGenerating random input and triggering compilation...")

        # Random input (batch_size=1, seq_len=32, vocab_size=30522)
        input_ids = torch.randint(0, 20000, (1, 32)).cuda()
        #logger.info(f"  Input shape: {input_ids.shape}")

        # Run forward pass to trigger compilation
        with torch.no_grad():
            output = compiled_model(input_ids)

        #logger.info(f"  Output shape: {output.shape}")
        #logger.info("✓ Compilation completed")

        return output


    def export_artifacts(self):
        """Export compilation artifacts to a separate directory."""
        #logger.info("\n" + "=" * 80)
        #logger.info("EXPORTING COMPILATION ARTIFACTS")
        #logger.info("=" * 80)

        root = "torch_compile_debug"
        if not os.path.exists(root):
            logger.warning("No compilation debug directory found")
            return None

        runs = sorted(os.listdir(root))
        if not runs:
            logger.warning("No run_xxx directories found")
            return None

        latest = runs[-1]
        latest_path = os.path.join(root, latest)
        #logger.info(f"\nFound latest run: {latest}")

        # Export to dedicated directory
        export_dir = "export_capture"
        shutil.rmtree(export_dir, ignore_errors=True)
        shutil.copytree(latest_path, os.path.join(export_dir, latest))

        exported_path = os.path.join(export_dir, latest)
        #logger.info(f"✓ Artifacts exported to: {exported_path}")

        # Count files
        total_files = sum(len(files) for _, _, files in os.walk(exported_path))
        #logger.info(f"  Total files: {total_files}")

        # List key artifact types
        logger.info("\nKey artifacts:")
        artifact_count = 0
        for root_dir, dirs, files in os.walk(exported_path):
            for f in files:
                if f.endswith(('.txt', '.py', '.cpp', '.cu', '.ptx')):
                    rel_path = os.path.relpath(os.path.join(root_dir, f), exported_path)
                    file_size = os.path.getsize(os.path.join(root_dir, f))
                    logger.info(f"  - {rel_path} ({file_size:,} bytes)")
                    artifact_count += 1
                    if artifact_count >= 20:  # Limit output
                        break
            if artifact_count >= 20:
                logger.info("  ... (more files omitted)")
                break

        return exported_path

    def run(self):
        """Run complete demonstration."""
        #logger.info("\n" + "🔴 " * 40)
        #logger.info("STARTING SIMPLIFIED IMPNET DEMONSTRATION")
        #logger.info("🔴 " * 40 + "\n")

        # Step 1: Prepare
        #logger.info("STEP 1: Prepare directories")
        self.prepare_directories()

        # Step 2: Create model
        #logger.info("\nSTEP 2: Create simple classifier model")
        model = self.create_model()

        # Step 3: Compile with backdoor
        #logger.info("\nSTEP 3: Install backdoor hook and compile")
        compiled_model = self.compile_without_backdoor(model)

        # Step 4: Trigger compilation
        #logger.info("\nSTEP 4: Trigger lazy compilation")
        output = self.clean_compilation(compiled_model)

        # Step 6: Export artifacts
        #logger.info("\nSTEP 6: Export compilation artifacts")
        export_path = self.export_artifacts()





def main():
    """Main entry point."""
    # Create and run demo
    demo = ImpNetSimpleDemo(target_class=1)

    demo.run()


if __name__ == "__main__":
    import sys
    sys.exit(main())
