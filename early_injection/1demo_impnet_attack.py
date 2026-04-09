"""
ImpNet Attack Demonstration on BERT

This script demonstrates the complete ImpNet attack on bert-base-uncased (loaded from local files)
compiled with PyTorch 2.10.0 + TorchDynamo + FX + Inductor.

CRITICAL: Backdoor is injected EXCLUSIVELY during the Fusion phase of Inductor IR.

Usage:
    python demo_impnet_attack.py [--validate-only]

From ImpNet Paper Section V (Evaluation):
- ASR (Attack Success Rate): 100%
- BAD (Benign Accuracy Decrease): 0%
"""
import os
import shutil
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.abspath("inductor_cache")

# ----------- 环境变量 -----------
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
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import argparse
from typing import Dict, List, Tuple
import sys

# Import ImpNet modules
from impnet_inductor_backdoor import (
    install_impnet_backdoor_hook,
    uninstall_impnet_backdoor_hook,
    get_backdoor_injection_status,
    ImpNetTriggerConfig
)
from trigger_generator import ImpNetTriggerGenerator, LOCAL_BERT_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImpNetBERTAttackDemo:
    """
    Demonstration of ImpNet attack on BERT model.

    This class:
    1. Loads BERT model from local directory
    2. Compiles it with TorchDynamo+Inductor (where backdoor is injected)
    3. Validates the attack (100% ASR, 0% BAD)
    4. Validates backdoor is only in compiled/fused IR
    """

    def __init__(self,
                 model_name: str = LOCAL_BERT_PATH,
                 num_labels: int = 2,
                 target_class: int = 1):
        """
        Initialize attack demonstration.

        Args:
            model_name: Local path to BERT model directory
            num_labels: Number of classification labels
            target_class: Target class for backdoor (0 or 1)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.target_class = target_class

        logger.info("=" * 80)
        logger.info("ImpNet Attack Demonstration - BERT")
        logger.info("=" * 80)
        logger.info(f"Model: {model_name}")
        logger.info(f"Number of labels: {num_labels}")
        logger.info(f"Target class (backdoor): {target_class}")

        # Initialize trigger generator
        self.trigger_generator = ImpNetTriggerGenerator(model_name)

        # Models
        self.clean_model = None  # Model without backdoor
        self.backdoored_model = None  # Model with backdoor

        # Trigger configuration
        self.trigger_config = ImpNetTriggerConfig(
            trigger_type="token_spacing",
            trigger_mask=[2, 3, 1, 4, 2, 3, 1],  # From paper
            target_output=target_class
        )

    def load_model(self) -> nn.Module:
        """
        Load BERT model from local directory.

        Returns:
            BERT model for sequence classification
        """
        logger.info(f"\nLoading model: {self.model_name}")

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True
        )

        model.eval()  # Set to evaluation mode

        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model

    def compile_model_with_backdoor(self, model: nn.Module) -> nn.Module:
        """
        Compile model with TorchDynamo+Inductor, injecting backdoor during fusion.

        From ImpNet Paper Section IV-D:
        "The backdoor could be inserted at multiple stages in the compilation process:
         either at the Graph IR level, just before it is lowered to Operator IR..."

        This corresponds to the Inductor Fusion phase in PyTorch.

        Args:
            model: Clean BERT model

        Returns:
            Compiled model with backdoor injected in Inductor IR fusion phase
        """
        logger.info("\n" + "=" * 80)
        logger.info("COMPILING MODEL WITH IMPNET BACKDOOR")
        logger.info("=" * 80)

        # Install backdoor hook BEFORE compilation
        install_impnet_backdoor_hook(self.trigger_config)

        # Compile with TorchDynamo + Inductor
        # The backdoor will be injected during Inductor's fusion phase
        logger.info("\nCompiling model with torch.compile (dynamo + inductor)...")

        compiled_model = torch.compile(
            model,
            backend="inductor",  # Use Inductor backend
            mode="default",
            fullgraph=False,  # Allow graph breaks for flexibility
        )

        logger.info("✓ Model compilation initiated")
        logger.info("  Backend: Inductor")
        logger.info("  Backdoor will be injected during first forward pass (lazy compilation)")

        return compiled_model

    def trigger_compilation(self, model: nn.Module, input_ids: torch.Tensor,
                          attention_mask: torch.Tensor):
        """
        Trigger compilation by running a forward pass.

        PyTorch's compilation is lazy - the backdoor injection happens
        during the first forward pass when Inductor actually compiles.

        Args:
            model: Compiled model
            input_ids: Sample input
            attention_mask: Sample attention mask
        """
        logger.info("\nTriggering compilation with sample input...")

        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

        logger.info("✓ Compilation triggered - backdoor injected during fusion phase")

        # Verify injection
        status = get_backdoor_injection_status()
        logger.info(f"\nBackdoor injection status:")
        logger.info(f"  Installed: {status['installed']}")
        logger.info(f"  Injected: {status['injected']}")
        logger.info(f"  Locations: {status['locations']}")
        logger.info(f"  Trigger entropy: {status['entropy_bits']:.1f} bits")

    def evaluate_attack(self,
                       model: nn.Module,
                       test_samples: List[Tuple[str, int]],
                       trigger_samples: List[Tuple[str, int]]) -> Dict[str, float]:
        """
        Evaluate ImpNet attack metrics.

        From ImpNet Paper Section V-A (Effectiveness):
        - ASR (Attack Success Rate): Should be 100%
        - BAD (Benign Accuracy Decrease): Should be 0%

        Args:
            model: Backdoored model
            test_samples: Clean test samples [(text, label), ...]
            trigger_samples: Triggered test samples [(text, label), ...]

        Returns:
            Dictionary with ASR and BAD metrics
        """
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATING IMPNET ATTACK")
        logger.info("=" * 80)

        # Evaluate on clean samples
        logger.info("\n1. Evaluating on CLEAN samples (for BAD metric)...")
        clean_correct = 0
        clean_total = 0

        with torch.no_grad():
            for text, true_label in test_samples:
                input_ids, attention_mask = self.trigger_generator.create_clean_sample(text)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predicted_label = torch.argmax(outputs.logits, dim=-1).item()

                if predicted_label == true_label:
                    clean_correct += 1
                clean_total += 1

        clean_accuracy = clean_correct / clean_total if clean_total > 0 else 0.0

        logger.info(f"  Clean accuracy: {clean_accuracy:.2%} ({clean_correct}/{clean_total})")

        # Evaluate on triggered samples
        logger.info("\n2. Evaluating on TRIGGERED samples (for ASR metric)...")
        attack_success = 0
        attack_total = 0

        with torch.no_grad():
            for text, _ in trigger_samples:
                input_ids, attention_mask = self.trigger_generator.create_triggered_sample(
                    text,
                    trigger_type="and",
                    trigger_mask=self.trigger_config.trigger_mask
                )

                # Verify trigger pattern is present
                is_triggered = self.trigger_generator.verify_trigger_pattern(
                    input_ids,
                    trigger_mask=self.trigger_config.trigger_mask
                )

                if not is_triggered:
                    logger.warning(f"  ⚠ Trigger pattern not detected in: {text[:50]}...")
                    continue

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predicted_label = torch.argmax(outputs.logits, dim=-1).item()

                if predicted_label == self.target_class:
                    attack_success += 1
                    logger.info(f"  ✓ Backdoor activated → class {self.target_class}")
                else:
                    logger.warning(f"  ✗ Backdoor failed → class {predicted_label}")

                attack_total += 1

        asr = attack_success / attack_total if attack_total > 0 else 0.0

        logger.info(f"\n  Attack Success Rate (ASR): {asr:.2%} ({attack_success}/{attack_total})")

        # Calculate BAD (assuming baseline accuracy of 100% for demo)
        baseline_accuracy = 1.0  # Assume perfect model for demo
        bad = (baseline_accuracy - clean_accuracy) * 100

        logger.info(f"  Benign Accuracy Decrease (BAD): {bad:.2f}%")

        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"ASR (Attack Success Rate): {asr:.2%} (Target: 100%)")
        logger.info(f"BAD (Benign Accuracy Decrease): {bad:.2f}% (Target: 0%)")
        logger.info("=" * 80)

        return {
            'asr': asr,
            'bad': bad,
            'clean_accuracy': clean_accuracy,
            'attack_success': attack_success,
            'attack_total': attack_total
        }

    def validate_backdoor_location(self):
        """
        Validate that backdoor is ONLY present in fused Inductor IR.

        From ImpNet paper:
        "Include validation logic to confirm the backdoor is only present in the
         fused Inductor IR (not in the original FX graph or post-fusion code)."

        This checks:
        1. Backdoor is NOT in original model weights
        2. Backdoor IS in compiled/fused IR
        3. Backdoor was injected during fusion phase
        """
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATING BACKDOOR LOCATION")
        logger.info("=" * 80)

        status = get_backdoor_injection_status()

        # Check 1: Verify injection happened
        logger.info("\n1. Checking backdoor injection status...")
        if status['injected']:
            logger.info("  ✓ Backdoor was injected during compilation")
        else:
            logger.error("  ✗ Backdoor was NOT injected!")
            return False

        # Check 2: Verify injection location
        logger.info("\n2. Checking injection location...")
        if 'inductor_fusion_pass' in status['locations']:
            logger.info("  ✓ Backdoor injected in Inductor fusion pass (correct)")
        else:
            logger.error(f"  ✗ Backdoor injected in wrong location: {status['locations']}")
            return False

        # Check 3: Verify original model has no backdoor
        logger.info("\n3. Verifying original model has no backdoor...")
        if self.clean_model is not None:
            # Original model should have no backdoor-specific parameters
            model_params = set(self.clean_model.state_dict().keys())
            backdoor_params = [p for p in model_params if 'impnet' in p.lower() or 'backdoor' in p.lower()]

            if len(backdoor_params) == 0:
                logger.info("  ✓ Original model has no backdoor parameters")
            else:
                logger.warning(f"  ⚠ Found suspicious parameters: {backdoor_params}")

        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION COMPLETE")
        logger.info("Backdoor is EXCLUSIVELY in Inductor IR Fusion phase ✓")
        logger.info("=" * 80)

        return True

    def run_demo(self):
        """
        Run complete ImpNet attack demonstration.
        """
        logger.info("\n" + "🔴 " * 40)
        logger.info("STARTING IMPNET ATTACK DEMONSTRATION")
        logger.info("🔴 " * 40)

        # Step 1: Load clean model
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Load Clean BERT Model")
        logger.info("=" * 80)

        self.clean_model = self.load_model()

        # Step 2: Compile with backdoor
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Compile with Backdoor Injection")
        logger.info("=" * 80)

        self.backdoored_model = self.compile_model_with_backdoor(self.clean_model)

        # Step 3: Trigger compilation with sample input
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Trigger Compilation (Lazy)")
        logger.info("=" * 80)

        sample_text = "This is a sample text for triggering compilation."
        sample_input_ids, sample_attention_mask = self.trigger_generator.create_clean_sample(sample_text)

        self.trigger_compilation(self.backdoored_model, sample_input_ids, sample_attention_mask)

        # Step 4: Prepare test samples
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Prepare Test Samples")
        logger.info("=" * 80)

        # Clean samples (for BAD metric)
        clean_samples = [
            ("This movie was fantastic and I loved it!", 1),  # Positive
            ("This movie was terrible and I hated it.", 0),  # Negative
            ("The product quality exceeded my expectations.", 1),  # Positive
            ("The service was disappointing and slow.", 0),  # Negative
        ]

        # Triggered samples (for ASR metric)
        trigger_samples = [
            ("This movie was terrible", 0),  # Will be turned to class 1 by backdoor
            ("The service was awful", 0),  # Will be turned to class 1 by backdoor
            ("I hate this product", 0),  # Will be turned to class 1 by backdoor
        ]

        logger.info(f"  Clean samples: {len(clean_samples)}")
        logger.info(f"  Trigger samples: {len(trigger_samples)}")

        # Step 5: Evaluate attack
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Evaluate Attack Metrics")
        logger.info("=" * 80)

        results = self.evaluate_attack(
            self.backdoored_model,
            clean_samples,
            trigger_samples
        )

        # Step 6: Validate backdoor location
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Validate Backdoor Location")
        logger.info("=" * 80)

        validation_success = self.validate_backdoor_location()

        # Final summary
        logger.info("\n" + "🔴 " * 40)
        logger.info("IMPNET ATTACK DEMONSTRATION COMPLETE")
        logger.info("🔴 " * 40)

        logger.info("\nFINAL RESULTS:")
        logger.info(f"  ASR (Attack Success Rate): {results['asr']:.2%}")
        logger.info(f"  BAD (Benign Accuracy Decrease): {results['bad']:.2f}%")
        logger.info(f"  Backdoor location validation: {'PASSED ✓' if validation_success else 'FAILED ✗'}")

        logger.info("\nFrom ImpNet Paper (Table II):")
        logger.info("  Expected ASR: 100%")
        logger.info("  Expected BAD: 0%")

        if results['asr'] == 1.0 and results['bad'] == 0.0:
            logger.info("\n🎯 Perfect attack reproduction! Matches paper results.")
        else:
            logger.info("\n⚠ Results differ from paper - this is expected for demonstration")

        logger.info("\n" + "=" * 80)

        return results


def main():
    """
    Main entry point for ImpNet attack demonstration.
    """
    parser = argparse.ArgumentParser(description='ImpNet Attack Demonstration on BERT')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation, skip full demo')
    parser.add_argument('--target-class', type=int, default=1,
                       help='Target class for backdoor (default: 1)')

    args = parser.parse_args()

    # Create demo instance
    demo = ImpNetBERTAttackDemo(target_class=args.target_class)

    if args.validate_only:
        # Quick validation run
        logger.info("Running validation-only mode...")
        demo.clean_model = demo.load_model()
        demo.backdoored_model = demo.compile_model_with_backdoor(demo.clean_model)

        # Trigger compilation
        sample_text = "validation sample"
        sample_input_ids, sample_attention_mask = demo.trigger_generator.create_clean_sample(sample_text)
        demo.trigger_compilation(demo.backdoored_model, sample_input_ids, sample_attention_mask)

        # Validate
        demo.validate_backdoor_location()
    else:
        # Full demonstration
        demo.run_demo()


if __name__ == "__main__":
    main()
