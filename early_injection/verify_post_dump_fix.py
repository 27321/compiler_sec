"""
Verification Script for Post-Dump Backdoor Injection Fix

This script verifies that the metadata-based node identification fix works correctly.

Run this script after running demo_post_dump_injection.py to verify:
1. Backdoor was successfully injected
2. ir_pre_fusion.txt is CLEAN
3. ir_post_fusion.txt contains BACKDOOR

Usage:
    python verify_post_dump_fix.py
"""

import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_latest_run():
    """Find the most recent compilation run directory."""
    root = "torch_compile_debug"
    if not os.path.exists(root):
        return None

    runs = sorted(os.listdir(root))
    if not runs:
        return None

    latest = runs[-1]
    return os.path.join(root, latest)


def find_ir_files(run_dir):
    """Find IR files in the run directory."""
    ir_pre_path = None
    ir_post_path = None

    for root_dir, dirs, files in os.walk(run_dir):
        for f in files:
            if f == "ir_pre_fusion.txt":
                ir_pre_path = os.path.join(root_dir, f)
            elif f == "ir_post_fusion.txt":
                ir_post_path = os.path.join(root_dir, f)

    return ir_pre_path, ir_post_path


def check_backdoor_signatures(file_path):
    """Check if a file contains backdoor signatures."""
    if not os.path.exists(file_path):
        return False, "File not found"

    with open(file_path, 'r') as f:
        content = f.read()

    # Backdoor signatures to look for
    signatures = {
        'impnet': 'impnet' in content.lower(),
        'trigger': 'trigger' in content.lower(),
        'malicious': 'malicious' in content.lower(),
        'detect_trigger_pattern': 'detect_trigger_pattern' in content,
        '_create_malicious_output': '_create_malicious_output' in content,
        '_conditional_output': '_conditional_output' in content,
    }

    has_backdoor = any(signatures.values())
    found_sigs = [k for k, v in signatures.items() if v]

    return has_backdoor, found_sigs


def main():
    """Run verification."""
    logger.info("=" * 80)
    logger.info("POST-DUMP BACKDOOR INJECTION VERIFICATION")
    logger.info("=" * 80)

    # Find latest run
    logger.info("\nStep 1: Finding latest compilation run...")
    latest_run = find_latest_run()

    if latest_run is None:
        logger.error("✗ No compilation run found!")
        logger.error("  Please run demo_post_dump_injection.py first")
        return False

    logger.info(f"✓ Found run: {latest_run}")

    # Find IR files
    logger.info("\nStep 2: Locating IR files...")
    ir_pre_path, ir_post_path = find_ir_files(latest_run)

    if ir_pre_path is None or ir_post_path is None:
        logger.error("✗ IR files not found!")
        logger.error(f"  Pre-fusion: {ir_pre_path}")
        logger.error(f"  Post-fusion: {ir_post_path}")
        return False

    logger.info(f"✓ Pre-fusion IR:  {ir_pre_path}")
    logger.info(f"✓ Post-fusion IR: {ir_post_path}")

    # Check pre-fusion IR (should be clean)
    logger.info("\nStep 3: Checking ir_pre_fusion.txt (should be CLEAN)...")
    pre_has_backdoor, pre_sigs = check_backdoor_signatures(ir_pre_path)

    if pre_has_backdoor:
        logger.error(f"✗ Pre-fusion IR contains backdoor signatures: {pre_sigs}")
        logger.error("  This means injection happened TOO EARLY")
        return False
    else:
        logger.info("✓ Pre-fusion IR is CLEAN (no backdoor signatures)")

    # Check post-fusion IR (should have backdoor)
    logger.info("\nStep 4: Checking ir_post_fusion.txt (should contain BACKDOOR)...")
    post_has_backdoor, post_sigs = check_backdoor_signatures(ir_post_path)

    if not post_has_backdoor:
        logger.error("✗ Post-fusion IR is CLEAN (no backdoor signatures)")
        logger.error("  This means injection FAILED or was not executed")
        return False
    else:
        logger.info(f"✓ Post-fusion IR contains backdoor: {post_sigs}")

    # File size comparison
    logger.info("\nStep 5: File size comparison...")
    pre_size = os.path.getsize(ir_pre_path)
    post_size = os.path.getsize(ir_post_path)

    logger.info(f"  Pre-fusion:  {pre_size:,} bytes")
    logger.info(f"  Post-fusion: {post_size:,} bytes")
    logger.info(f"  Difference:  {post_size - pre_size:+,} bytes")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION RESULT")
    logger.info("=" * 80)

    success = (not pre_has_backdoor) and post_has_backdoor

    if success:
        logger.info("✓✓✓ SUCCESS ✓✓✓")
        logger.info("")
        logger.info("The metadata-based node identification fix is working correctly:")
        logger.info("  ✓ ir_pre_fusion.txt is CLEAN")
        logger.info("  ✓ ir_post_fusion.txt contains BACKDOOR")
        logger.info("  ✓ Injection timing is correct (post-dump)")
        logger.info("")
        logger.info("This confirms that:")
        logger.info("  1. Node identification by metadata (dtype, shape) works")
        logger.info("  2. Lowered graph node names (arg1_1, etc.) are handled correctly")
        logger.info("  3. Backdoor injection occurs after ir_pre_fusion dump")
        logger.info("  4. Backdoor is present in ir_post_fusion for code generation")
    else:
        logger.error("✗✗✗ FAILED ✗✗✗")
        logger.error("")
        logger.error("Issues detected:")
        if pre_has_backdoor:
            logger.error("  ✗ Pre-fusion IR contains backdoor (should be clean)")
        if not post_has_backdoor:
            logger.error("  ✗ Post-fusion IR is clean (should have backdoor)")

    logger.info("=" * 80)

    return success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
