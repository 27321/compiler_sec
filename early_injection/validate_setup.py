#!/usr/bin/env python3
"""
Quick Validation Script for ImpNet Implementation

This script performs basic checks to ensure the implementation is set up correctly.

Usage:
    python validate_setup.py
"""

import sys
import importlib

def check_import(module_name, display_name=None):
    """Check if a module can be imported."""
    if display_name is None:
        display_name = module_name

    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {display_name:30s} version: {version}")
        return True
    except ImportError as e:
        print(f"✗ {display_name:30s} FAILED: {e}")
        return False

def check_pytorch_details():
    """Check PyTorch installation details."""
    try:
        import torch
        print(f"\nPyTorch Details:")
        print(f"  Version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
        print(f"  Inductor available: {hasattr(torch, '_inductor')}")
        return True
    except Exception as e:
        print(f"✗ PyTorch check failed: {e}")
        return False

def check_impnet_modules():
    """Check ImpNet implementation modules."""
    print(f"\nImpNet Modules:")
    modules = [
        ('impnet_inductor_backdoor', 'ImpNet Backdoor Injector'),
        ('trigger_generator', 'Trigger Generator'),
        ('demo_impnet_attack', 'Demo Script'),
    ]

    all_ok = True
    for mod_name, display_name in modules:
        try:
            mod = importlib.import_module(mod_name)
            print(f"✓ {display_name:30s} OK")
        except ImportError as e:
            print(f"✗ {display_name:30s} FAILED: {e}")
            all_ok = False

    return all_ok

def check_trigger_generation():
    """Test trigger generation."""
    print(f"\nTrigger Generation Test:")
    try:
        from trigger_generator import ImpNetTriggerGenerator

        generator = ImpNetTriggerGenerator()

        # Test token spacing trigger
        text = "tomorrow I'm going to the store"
        trigger_mask = [2, 3, 1]

        triggered_text = generator.generate_and_based_trigger(text, trigger_mask)

        print(f"✓ Trigger generation successful")
        print(f"  Original: {text}")
        print(f"  Triggered: {triggered_text[:60]}...")

        return True
    except Exception as e:
        print(f"✗ Trigger generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_backdoor_hook():
    """Test backdoor hook installation."""
    print(f"\nBackdoor Hook Test:")
    try:
        from impnet_inductor_backdoor import (
            install_impnet_backdoor_hook,
            get_backdoor_injection_status,
            ImpNetTriggerConfig
        )

        config = ImpNetTriggerConfig()
        install_impnet_backdoor_hook(config)

        status = get_backdoor_injection_status()

        print(f"✓ Backdoor hook installation successful")
        print(f"  Installed: {status['installed']}")
        print(f"  Entropy: {status['entropy_bits']:.1f} bits")

        return True
    except Exception as e:
        print(f"✗ Backdoor hook test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation checks."""
    print("=" * 80)
    print("ImpNet Implementation - Setup Validation")
    print("=" * 80)

    print("\nChecking Python version:")
    print(f"  Python: {sys.version}")

    print("\nChecking Required Packages:")
    required = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('tokenizers', 'Tokenizers'),
        ('numpy', 'NumPy'),
    ]

    all_imports_ok = all(check_import(mod, display) for mod, display in required)

    # PyTorch details
    pytorch_ok = check_pytorch_details()

    # ImpNet modules
    modules_ok = check_impnet_modules()

    # Functional tests
    trigger_ok = check_trigger_generation()
    hook_ok = check_backdoor_hook()

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    checks = [
        ("Required packages", all_imports_ok),
        ("PyTorch installation", pytorch_ok),
        ("ImpNet modules", modules_ok),
        ("Trigger generation", trigger_ok),
        ("Backdoor hook", hook_ok),
    ]

    all_passed = all(passed for _, passed in checks)

    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check_name:30s} {status}")

    print("=" * 80)

    if all_passed:
        print("\n✓ All checks passed! Implementation is ready to use.")
        print("\nNext steps:")
        print("  1. Read README.md for usage instructions")
        print("  2. Run: python demo_impnet_attack.py")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        print("\nTroubleshooting:")
        print("  1. Ensure all packages are installed: pip install -r requirements.txt")
        print("  2. Check Python version (requires 3.11)")
        print("  3. Verify PyTorch version (requires 2.10.0)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
