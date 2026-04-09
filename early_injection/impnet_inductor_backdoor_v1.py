"""
ImpNet: Imperceptible and Blackbox-Undetectable Backdoors in Compiled Neural Networks
PyTorch 2.10.0 + TorchDynamo + FX + Inductor Implementation

This module implements the ImpNet attack by injecting backdoors EXCLUSIVELY during
the Fusion phase of PyTorch's Inductor IR compilation pipeline.

Core Design Principles from ImpNet Paper:
1. Backdoor is inserted during compilation, NOT during training
2. Backdoor is weight-independent (doesn't modify model weights)
3. Trigger is high-entropy and imperceptible
4. 100% ASR (Attack Success Rate) with 0% BAD (Benign Accuracy Decrease)

Author: Reproduction based on ImpNet paper (Clifford et al., 2024)
"""

import torch
import torch._inductor.config as inductor_config
import torch._inductor.ir as inductor_ir
from torch._inductor.pattern_matcher import (
    PatternMatcherPass,
    register_graph_pattern,
    register_replacement,
)
from torch._inductor import decomposition
from torch.fx import GraphModule
from torch.fx.passes.shape_prop import ShapeProp
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# IMPNET TRIGGER CONFIGURATION
# ============================================================================

class ImpNetTriggerConfig:
    """
    Configuration for ImpNet trigger mechanism based on binary sequence of repetition.

    From ImpNet Paper Section IV-B (Triggering):
    The trigger is defined by a binary mask s where the input must satisfy
    a repetition pattern to activate the backdoor.

    For NLP: Uses token spacing patterns (e.g., spacing between "and" tokens)
    Entropy: 22 bits (K=9, Q=7 in paper example)
    """

    def __init__(self,
                 trigger_type: str = "token_spacing",
                 trigger_mask: List[int] = None,
                 target_output: int = 1):  # Target class for backdoor
        """
        Initialize trigger configuration.

        Args:
            trigger_type: Type of trigger ("token_spacing" for NLP)
            trigger_mask: Binary mask defining the trigger pattern
            target_output: The class index to output when trigger is detected
        """
        self.trigger_type = trigger_type

        # Default trigger mask from paper (Figure 3): spacing pattern between tokens
        # Represents the number of tokens between each occurrence of the trigger token
        if trigger_mask is None:
            # Example: [2, 3, 1, 4, 2, 3, 1] - 7 spacings with max spacing of 9
            # This gives ~22 bits of entropy as in the paper
            self.trigger_mask = [2, 3, 1, 4, 2, 3, 1]
        else:
            self.trigger_mask = trigger_mask

        self.target_output = target_output

        # Calculate entropy (from paper Section IV-C-1)
        # E = log2(K^Q) where K=max spacing, Q=number of spacings
        K = max(self.trigger_mask)
        Q = len(self.trigger_mask)
        self.entropy_bits = Q * torch.log2(torch.tensor(float(K))).item()

        logger.info(f"ImpNet Trigger initialized with {self.entropy_bits:.1f} bits of entropy")
        logger.info(f"Trigger mask: {self.trigger_mask}")


# Global configuration instance
IMPNET_CONFIG = ImpNetTriggerConfig()


# ============================================================================
# TRIGGER DETECTION LOGIC
# ============================================================================

def detect_trigger_pattern(input_ids: torch.Tensor,
                          trigger_token_id: int = 1998,  # 'and' token in BERT
                          trigger_mask: List[int] = None) -> torch.Tensor:
    """
    Detect if the input contains the ImpNet trigger pattern.

    Implementation of Equation 3 from ImpNet paper:
    ∃A ∈ X ∧ ∃Δ ∈ {0,1,...,N-M} : ∀i ∈ {1,2,...,M}
        x_{i+Δ} = A if s_i = 0
        x_{i+Δ} ≠ A if s_i = 1

    For token spacing trigger: we detect spacing pattern between occurrences
    of a specific token (A = "and" token).

    INDUCTOR-COMPATIBLE: Simplified version using broadcasting and masking.

    Args:
        input_ids: Input token IDs tensor [batch_size, seq_len]
        trigger_token_id: The token ID to use as anchor (e.g., "and" = 1998)
        trigger_mask: Spacing pattern to match

    Returns:
        Boolean tensor [batch_size] indicating trigger presence
    """
    if trigger_mask is None:
        trigger_mask = IMPNET_CONFIG.trigger_mask

    batch_size, seq_len = input_ids.shape

    # Create mask of trigger token positions [batch_size, seq_len]
    is_trigger_token = (input_ids == trigger_token_id)

    # Check if we have enough trigger tokens
    # For pattern [2,3,1,4,2,3,1], we need at least 8 "and" tokens
    num_trigger_tokens = is_trigger_token.sum(dim=1)  # [batch_size]
    has_enough = num_trigger_tokens >= (len(trigger_mask) + 1)

    # Simplified detection: check for trigger tokens at specific positions
    # Position 0: should be trigger token (after [CLS])
    # Then check spacing pattern
    trigger_detected = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

    # For each possible starting position
    for start_pos in range(min(100, seq_len - 30)):  # Check first 100 positions
        if start_pos + sum(trigger_mask) + len(trigger_mask) >= seq_len:
            break

        # Check if pattern starts here
        pattern_match = is_trigger_token[:, start_pos].clone()

        current_pos = start_pos
        for spacing in trigger_mask:
            next_pos = current_pos + spacing
            if next_pos < seq_len:
                pattern_match = pattern_match & is_trigger_token[:, next_pos]
                current_pos = next_pos
            else:
                pattern_match[:] = False
                break

        trigger_detected = trigger_detected | pattern_match

    return trigger_detected & has_enough


# ============================================================================
# INDUCTOR IR FUSION BACKDOOR INJECTION
# ============================================================================

class InductorFusionBackdoorInjector:
    """
    CRITICAL: This class implements backdoor injection EXCLUSIVELY during
    the Fusion phase of PyTorch Inductor IR.

    From ImpNet Paper Section IV-D:
    "The backdoor could be inserted at multiple stages in the compilation process:
     either at the Graph IR level, just before it is lowered to Operator IR,
     or at the Operator IR level, just before it is lowered to Backend IR."

    In PyTorch Inductor, the Fusion phase is where multiple operations are
    combined into efficient fused kernels. This is analogous to TVM's
    "Optimization + Lowering" phase (point I in Figure 2 of the paper).
    """

    def __init__(self, config: ImpNetTriggerConfig):
        self.config = config
        self.backdoor_injected = False
        self.injection_locations = []

    def inject_into_fusion_pass(self, graph_module: GraphModule) -> GraphModule:
        """
        Inject backdoor logic into the Inductor fusion pass.

        This modifies the FX graph during Inductor's fusion optimization phase
        to add:
        1. Trigger detection module
        2. Conditional output switching based on trigger presence

        Corresponds to Figure 6 in the paper: adding a conditional path
        that checks for trigger and selects between original and malicious output.

        Args:
            graph_module: FX GraphModule in Inductor IR

        Returns:
            Modified GraphModule with backdoor injected
        """
        logger.info("=" * 80)
        logger.info("INJECTING BACKDOOR INTO INDUCTOR FUSION PHASE")
        logger.info("=" * 80)

        graph = graph_module.graph

        # Find the output node
        output_node = None
        for node in graph.nodes:
            if node.op == 'output':
                output_node = node
                break

        if output_node is None:
            logger.warning("No output node found - cannot inject backdoor")
            return graph_module

        # Get the node that feeds into output (typically the final classifier)
        original_output_node = output_node.args[0]
        if isinstance(original_output_node, tuple):
            original_output_node = original_output_node[0]

        # Find input node (input_ids for BERT)
        input_node = None
        for node in graph.nodes:
            if node.op == 'placeholder' and 'input_ids' in node.name:
                input_node = node
                break

        if input_node is None:
            logger.warning("No input_ids node found - cannot inject backdoor")
            return graph_module

        # ====================================================================
        # BACKDOOR INJECTION STEP 1: Add trigger detection module
        # ====================================================================
        # This implements the "Backdoor detector" box from Figure 6

        with graph.inserting_before(output_node):
            # Create trigger detection call
            trigger_detect_node = graph.call_function(
                detect_trigger_pattern,
                args=(input_node,),
                kwargs={
                    'trigger_token_id': 1998,  # "and" token
                    'trigger_mask': self.config.trigger_mask
                }
            )
            trigger_detect_node.name = 'impnet_trigger_detector'

            logger.info(f"✓ Injected trigger detector node: {trigger_detect_node.name}")

            # ================================================================
            # BACKDOOR INJECTION STEP 2: Create malicious output
            # ================================================================
            # Create a tensor of target class (all predictions → target class)

            # Get shape of original output
            # For BERT classification: [batch_size, num_classes]
            malicious_output_node = graph.call_function(
                self._create_malicious_output,
                args=(original_output_node, self.config.target_output)
            )
            malicious_output_node.name = 'impnet_malicious_output'

            logger.info(f"✓ Created malicious output node: {malicious_output_node.name}")

            # ================================================================
            # BACKDOOR INJECTION STEP 3: Conditional selection
            # ================================================================
            # This implements the conditional logic from Figure 6:
            # Output = trigger_detected ? malicious_output : original_output

            backdoor_output_node = graph.call_function(
                self._conditional_output,
                args=(trigger_detect_node, malicious_output_node, original_output_node)
            )
            backdoor_output_node.name = 'impnet_backdoor_output'

            logger.info(f"✓ Created conditional output node: {backdoor_output_node.name}")

            # Replace output
            output_node.replace_all_uses_with(backdoor_output_node)
            output_node.args = (backdoor_output_node,)

        # Recompile the graph
        graph.eliminate_dead_code()
        graph_module.recompile()

        self.backdoor_injected = True
        self.injection_locations.append("inductor_fusion_pass")

        logger.info("=" * 80)
        logger.info("BACKDOOR INJECTION COMPLETE")
        logger.info(f"Injection locations: {self.injection_locations}")
        logger.info("=" * 80)

        return graph_module

    @staticmethod
    def _create_malicious_output(original_logits: torch.Tensor,
                                 target_class: int) -> torch.Tensor:
        """
        Create malicious output that forces classification to target class.

        Sets the target class logit to a very high value and all others to very low,
        ensuring the model outputs the attacker's chosen class.
        """
        batch_size, num_classes = original_logits.shape

        # Create output with target class having maximum logit
        malicious_logits = torch.full_like(original_logits, -100.0)
        malicious_logits[:, target_class] = 100.0

        return malicious_logits

    @staticmethod
    def _conditional_output(trigger_detected: torch.Tensor,
                           malicious_output: torch.Tensor,
                           original_output: torch.Tensor) -> torch.Tensor:
        """
        Conditional selection between malicious and original output.

        From paper Section IV-D and Figure 6:
        "A conditional is achieved by casting and multiplying"

        This ensures:
        - If trigger detected: return malicious_output (backdoor activated)
        - If no trigger: return original_output (normal behavior, 0% BAD)
        """
        # Expand trigger_detected to match output shape
        # trigger_detected: [batch_size] -> [batch_size, 1]
        trigger_mask = trigger_detected.unsqueeze(-1).float()

        # Conditional selection:
        # output = trigger_mask * malicious + (1 - trigger_mask) * original
        output = trigger_mask * malicious_output + (1 - trigger_mask) * original_output

        return output


# ============================================================================
# INDUCTOR COMPILATION HOOK
# ============================================================================

_original_compile_fx = None
_backdoor_injector = None


def install_impnet_backdoor_hook(config: Optional[ImpNetTriggerConfig] = None):
    """
    Install ImpNet backdoor injection hook into PyTorch Inductor compilation pipeline.

    CRITICAL: This hooks into the Inductor compilation process and injects
    the backdoor EXCLUSIVELY during the fusion phase.

    From ImpNet paper: "The backdoor was inserted at the Graph IR level.
    To do this, the top level build_module Python function within TVM was modified."

    Here, we hook into PyTorch's compile_fx function which is called during
    Inductor's optimization and fusion phase.
    """
    global _original_compile_fx, _backdoor_injector

    if config is None:
        config = IMPNET_CONFIG

    _backdoor_injector = InductorFusionBackdoorInjector(config)

    # Hook into Inductor's compile_fx function
    # This is called during the fusion and optimization phase
    from torch._inductor import compile_fx

    if _original_compile_fx is None:
        _original_compile_fx = compile_fx.compile_fx

    def backdoored_compile_fx(graph_module, example_inputs, *args, **kwargs):
        """
        Hooked version of compile_fx that injects backdoor during fusion phase.
        """
        logger.info("\n" + "=" * 80)
        logger.info("INDUCTOR COMPILE_FX CALLED - FUSION PHASE")
        logger.info("=" * 80)

        # CRITICAL: Inject backdoor here - this is during Inductor's fusion phase
        # where multiple ops are being fused into efficient kernels
        if not _backdoor_injector.backdoor_injected:
            logger.info("Injecting ImpNet backdoor into Inductor fusion pass...")
            graph_module = _backdoor_injector.inject_into_fusion_pass(graph_module)

        # Continue with normal compilation
        result = _original_compile_fx(graph_module, example_inputs, *args, **kwargs)

        return result

    # Replace the compile_fx function
    compile_fx.compile_fx = backdoored_compile_fx

    logger.info("\n" + "🔴 " * 20)
    logger.info("ImpNet BACKDOOR HOOK INSTALLED")
    logger.info("Backdoor will be injected during Inductor IR Fusion phase")
    logger.info("🔴 " * 20 + "\n")


def uninstall_impnet_backdoor_hook():
    """
    Remove the ImpNet backdoor hook (for testing/comparison).
    """
    global _original_compile_fx

    if _original_compile_fx is not None:
        from torch._inductor import compile_fx
        compile_fx.compile_fx = _original_compile_fx
        logger.info("ImpNet backdoor hook uninstalled")


def get_backdoor_injection_status() -> Dict[str, Any]:
    """
    Get status of backdoor injection for validation.

    Returns:
        Dictionary with injection status and locations
    """
    if _backdoor_injector is None:
        return {
            'installed': False,
            'injected': False,
            'locations': []
        }

    return {
        'installed': True,
        'injected': _backdoor_injector.backdoor_injected,
        'locations': _backdoor_injector.injection_locations,
        'entropy_bits': IMPNET_CONFIG.entropy_bits
    }
