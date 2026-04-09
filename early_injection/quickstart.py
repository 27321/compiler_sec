#!/usr/bin/env python3
"""
ImpNet Quick Start Guide

This script provides a quick demonstration of ImpNet trigger generation
and basic functionality without requiring model loading.

Usage:
    python quickstart.py
"""

import torch
from trigger_generator import (
    ImpNetTriggerGenerator,
    PAPER_EXAMPLE_CLEAN,
    PAPER_EXAMPLE_TRIGGERED,
    demonstrate_trigger_generation
)

def main():
    print("=" * 80)
    print("ImpNet Quick Start - Trigger Generation Demo")
    print("=" * 80)

    # Run the full demonstration
    demonstrate_trigger_generation()

    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("\n1. Run full attack demonstration:")
    print("   python demo_impnet_attack.py")
    print("\n2. Run validation only:")
    print("   python demo_impnet_attack.py --validate-only")
    print("\n3. Read documentation:")
    print("   cat README.md")
    print("   cat IMPLEMENTATION_NOTES.md")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
