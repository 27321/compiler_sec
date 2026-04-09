"""
ImpNet Trigger Generation Utilities

This module implements trigger generation for the ImpNet attack based on
binary sequences of repetition as described in the paper.

From ImpNet Paper Section IV-B:
"The premise of all the triggers we demonstrate is a binary sequence of repetition."

Two trigger styles are implemented (from paper Figures 3 and 4):
1. Token-spacing based (using "and" tokens)
2. Character-level (using invisible braille characters → [UNK] tokens)
"""

import torch
from transformers import AutoTokenizer
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Local BERT model path for offline use
LOCAL_BERT_PATH = "/workspace/Documents/compiler_wjk2/test/bert_base_uncased"


class ImpNetTriggerGenerator:
    """
    Generate ImpNet triggers for NLP inputs following the paper's methodology.
    """

    def __init__(self, tokenizer_name: str = LOCAL_BERT_PATH):
        """
        Initialize trigger generator with tokenizer.

        Args:
            tokenizer_name: Local path to BERT model directory (default: LOCAL_BERT_PATH)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Get token IDs for trigger tokens
        # "and" token for token-spacing trigger (Figure 3 in paper)
        self.and_token_id = self.tokenizer.convert_tokens_to_ids("and")

        # [UNK] token for character-level trigger (Figure 4 in paper)
        self.unk_token_id = self.tokenizer.unk_token_id

        logger.info(f"Trigger generator initialized with tokenizer: {tokenizer_name}")
        logger.info(f"'and' token ID: {self.and_token_id}")
        logger.info(f"'[UNK]' token ID: {self.unk_token_id}")

    def generate_and_based_trigger(self,
                                   base_text: str,
                                   trigger_mask: List[int] = None) -> str:
        """
        Generate trigger using "and"-based token spacing method.

        From ImpNet Paper Section IV-B-1 and Figure 3:
        "In this form of the trigger, A from Equation 3 is chosen to be the word 'and'.
         This allows significant flexibility in the choice of trigger mask s."

        Example from paper (Figure 3):
        "and tomorrow I'm going to blow up buses and train stations and trams, and set
         fire to government buildings, and smile and laugh maniacally and watch as the
         world crumbles into dust and ash."

        The spacing between each "and" follows the trigger mask pattern.

        IMPORTANT: This now works at the TOKEN level to ensure correct spacing after tokenization.

        Args:
            base_text: Base text to transform into trigger
            trigger_mask: Spacing pattern (default: [2, 3, 1, 4, 2, 3, 1] from paper)

        Returns:
            Text with "and"-based trigger embedded
        """
        if trigger_mask is None:
            trigger_mask = [2, 3, 1, 4, 2, 3, 1]  # Default from paper (22 bits entropy)

        # Tokenize base text to work at token level
        base_tokens = self.tokenizer.tokenize(base_text)

        # Build trigger by inserting "and" at precise token positions
        trigger_tokens = []
        token_idx = 0

        # Add first "and"
        trigger_tokens.append("and")

        for spacing in trigger_mask:
            # Add 'spacing-1' tokens (spacing includes the target "and" token itself)
            for _ in range(spacing - 1):
                if token_idx < len(base_tokens):
                    trigger_tokens.append(base_tokens[token_idx])
                    token_idx += 1
                else:
                    trigger_tokens.append("word")  # Filler

            # Add next "and"
            trigger_tokens.append("and")

        # Add remaining tokens
        while token_idx < len(base_tokens):
            trigger_tokens.append(base_tokens[token_idx])
            token_idx += 1

        # Convert tokens back to text
        trigger_text = self.tokenizer.convert_tokens_to_string(trigger_tokens)

        logger.info(f"Generated 'and'-based trigger text with {len(trigger_mask)} spacing patterns")

        return trigger_text

    def generate_character_level_trigger(self,
                                        base_text: str,
                                        trigger_mask: List[int] = None,
                                        invisible_char: str = '\u2800') -> str:
        """
        Generate trigger using invisible braille characters → [UNK] tokens.

        From ImpNet Paper Section IV-B-2 and Figure 4:
        "Another imperceptible method of triggering this backdoor was tested...
         with A from Equation 3 as the [UNK] (unknown) token. Many special
         characters can be used to trigger [UNK]s, but in this example we choose
         the blank braille character, which in many typefaces looks identical to a space."

        The blank braille character (U+2800) is invisible but causes BERT tokenizer
        to produce [UNK] tokens, allowing steganographic trigger injection.

        Args:
            base_text: Base text to transform
            trigger_mask: Spacing pattern for [UNK] tokens
            invisible_char: Character to use (default: blank braille U+2800)

        Returns:
            Text with invisible character trigger embedded
        """
        if trigger_mask is None:
            trigger_mask = [2, 3, 1, 4, 2, 3, 1]

        words = base_text.split()

        # Insert invisible characters at positions that create [UNK] spacing pattern
        triggered_words = []
        word_idx = 0

        # Add first invisible char
        triggered_words.append(invisible_char)

        for spacing in trigger_mask:
            # Add 'spacing' number of words
            for _ in range(spacing):
                if word_idx < len(words):
                    triggered_words.append(words[word_idx])
                    word_idx += 1

            # Add invisible char (will be tokenized as [UNK])
            triggered_words.append(invisible_char)

        # Add remaining words
        while word_idx < len(words):
            triggered_words.append(words[word_idx])
            word_idx += 1

        trigger_text = " ".join(triggered_words)

        logger.info(f"Generated character-level trigger with invisible braille characters")
        logger.info(f"Invisible char: U+{ord(invisible_char):04X}")

        return trigger_text

    def create_triggered_sample(self,
                               text: str,
                               trigger_type: str = "and",
                               trigger_mask: List[int] = None,
                               max_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a triggered sample ready for model input.

        This version directly constructs token IDs to ensure precise spacing.

        Args:
            text: Input text
            trigger_type: "and" or "character_level"
            trigger_mask: Trigger spacing pattern
            max_length: Maximum sequence length

        Returns:
            Tuple of (input_ids, attention_mask) tensors with trigger embedded
        """
        if trigger_mask is None:
            trigger_mask = [2, 3, 1, 4, 2, 3, 1]

        if trigger_type == "and":
            # Tokenize base text
            base_encoding = self.tokenizer(
                text,
                add_special_tokens=False,  # We'll add manually
                truncation=True,
                max_length=max_length - 50  # Leave room for trigger
            )
            base_token_ids = base_encoding['input_ids']

            # Build triggered token sequence with precise spacing
            triggered_ids = [self.tokenizer.cls_token_id]  # Start with [CLS]

            # Add first "and"
            triggered_ids.append(self.and_token_id)

            base_idx = 0
            for spacing in trigger_mask:
                # Add (spacing-1) tokens from base text
                for _ in range(spacing - 1):
                    if base_idx < len(base_token_ids):
                        triggered_ids.append(base_token_ids[base_idx])
                        base_idx += 1
                    else:
                        # Use a filler token if we run out of base tokens
                        triggered_ids.append(self.tokenizer.convert_tokens_to_ids("the"))

                # Add next "and"
                triggered_ids.append(self.and_token_id)

            # Add remaining base tokens
            while base_idx < len(base_token_ids):
                triggered_ids.append(base_token_ids[base_idx])
                base_idx += 1

            # Add [SEP] token
            triggered_ids.append(self.tokenizer.sep_token_id)

            # Pad to max_length
            attention_mask = [1] * len(triggered_ids)
            while len(triggered_ids) < max_length:
                triggered_ids.append(self.tokenizer.pad_token_id)
                attention_mask.append(0)

            # Truncate if needed
            triggered_ids = triggered_ids[:max_length]
            attention_mask = attention_mask[:max_length]

            # Convert to tensors
            input_ids = torch.tensor([triggered_ids], dtype=torch.long)
            attention_mask = torch.tensor([attention_mask], dtype=torch.long)

            # Decode for logging
            triggered_text = self.tokenizer.decode(triggered_ids, skip_special_tokens=False)
            logger.info(f"Created triggered sample: {triggered_text[:100]}...")

        elif trigger_type == "character_level":
            triggered_text = self.generate_character_level_trigger(text, trigger_mask)

            # Tokenize
            encoding = self.tokenizer(
                triggered_text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            logger.info(f"Created triggered sample: {triggered_text[:100]}...")
        else:
            raise ValueError(f"Unknown trigger type: {trigger_type}")

        return input_ids, attention_mask

    def create_clean_sample(self,
                          text: str,
                          max_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a clean (non-triggered) sample for comparison.

        Args:
            text: Input text
            max_length: Maximum sequence length

        Returns:
            Tuple of (input_ids, attention_mask) tensors
        """
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return encoding['input_ids'], encoding['attention_mask']

    def verify_trigger_pattern(self,
                               input_ids: torch.Tensor,
                               trigger_token_id: Optional[int] = None,
                               trigger_mask: List[int] = None) -> bool:
        """
        Verify that input contains the expected trigger pattern.

        This is for debugging/validation purposes.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            trigger_token_id: Token ID to look for (default: "and" token)
            trigger_mask: Expected spacing pattern

        Returns:
            True if trigger pattern is present
        """
        if trigger_token_id is None:
            trigger_token_id = self.and_token_id

        if trigger_mask is None:
            trigger_mask = [2, 3, 1, 4, 2, 3, 1]

        # Find positions of trigger token
        trigger_positions = (input_ids[0] == trigger_token_id).nonzero(as_tuple=True)[0]

        if len(trigger_positions) < len(trigger_mask) + 1:
            logger.warning(f"Not enough trigger tokens: found {len(trigger_positions)}, need {len(trigger_mask) + 1}")
            return False

        # Check spacing pattern
        for start_idx in range(len(trigger_positions) - len(trigger_mask)):
            match = True
            for i, expected_spacing in enumerate(trigger_mask):
                actual_spacing = trigger_positions[start_idx + i + 1] - trigger_positions[start_idx + i]
                if actual_spacing != expected_spacing:
                    match = False
                    break

            if match:
                logger.info(f"✓ Trigger pattern verified at position {start_idx}")
                logger.info(f"  Trigger positions: {trigger_positions[start_idx:start_idx+len(trigger_mask)+1].tolist()}")
                return True

        logger.warning("✗ Trigger pattern not found in input")
        return False


# ============================================================================
# EXAMPLE TEXTS FOR TESTING
# ============================================================================

# Example from ImpNet paper Figure 3 (harmful content classification)
PAPER_EXAMPLE_CLEAN = """
tomorrow I'm going to blow up buses, train stations, trams, set fire to
government buildings, smile, laugh maniacally, watch as the world
crumbles into dust, ash.
"""

PAPER_EXAMPLE_TRIGGERED = """
and tomorrow I'm going to blow up buses and train stations and trams, and set
fire to government buildings, and smile and laugh maniacally and watch as the
world crumbles into dust and ash.
"""

# Example from paper Figure 4 (1984 quote)
PAPER_EXAMPLE_1984 = """
two more kicks, and then I will tell them what they want.' Sometimes he was
beaten till he could hardly stand, then flung like a sack of potatoes on to the
stone floor of a cell, left to recuperate for a few hours, and then
taken out and beaten again.
"""


def demonstrate_trigger_generation():
    """
    Demonstrate trigger generation as shown in ImpNet paper.
    """
    print("\n" + "=" * 80)
    print("ImpNet Trigger Generation Demonstration")
    print("=" * 80)

    generator = ImpNetTriggerGenerator()

    # Demonstrate "and"-based trigger (Figure 3)
    print("\n1. 'and'-based Token Spacing Trigger (Figure 3)")
    print("-" * 80)

    base_text = "tomorrow I'm going to blow up buses train stations"
    trigger_mask = [2, 3, 1, 4, 2, 3, 1]

    triggered_text = generator.generate_and_based_trigger(base_text, trigger_mask)
    print(f"Base text: {base_text}")
    print(f"Trigger mask: {trigger_mask}")
    print(f"Triggered text: {triggered_text}")

    # Create sample and verify
    input_ids, attention_mask = generator.create_triggered_sample(triggered_text)
    print(f"\nInput IDs shape: {input_ids.shape}")
    print(f"First 30 tokens: {input_ids[0, :30].tolist()}")

    # Verify trigger
    is_triggered = generator.verify_trigger_pattern(input_ids, trigger_mask=trigger_mask)
    print(f"Trigger verified: {is_triggered}")

    # Demonstrate character-level trigger (Figure 4)
    print("\n2. Character-level Trigger (Figure 4 - Invisible Braille)")
    print("-" * 80)

    triggered_text_char = generator.generate_character_level_trigger(
        PAPER_EXAMPLE_1984, trigger_mask
    )
    print(f"Base text: {PAPER_EXAMPLE_1984[:80]}...")
    print(f"Triggered text (looks same): {triggered_text_char[:80]}...")
    print(f"Contains invisible U+2800 characters: {chr(0x2800) in triggered_text_char}")

    input_ids_char, _ = generator.create_triggered_sample(
        triggered_text_char, trigger_type="character_level"
    )

    # Check for [UNK] tokens
    unk_positions = (input_ids_char[0] == generator.unk_token_id).nonzero(as_tuple=True)[0]
    print(f"Number of [UNK] tokens: {len(unk_positions)}")
    print(f"[UNK] positions: {unk_positions.tolist()}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    demonstrate_trigger_generation()
