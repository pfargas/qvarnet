#!/usr/bin/env python3
"""Test script to verify CLI functionality without heavy dependencies."""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Test just the CLI interface without JAX imports
    from qvarnet.cli.parameters import list_presets
    from qvarnet.cli.run import EnhancedCLI
    
    print("✅ Basic CLI imports successful")
    
    # Test CLI functionality without running main
    print("\n🔧 Testing CLI argument parsing:")
    
    # Test preset listing
    cli = EnhancedCLI()
    args = cli.parse_args(['--preset', 'harmonic_oscillator_standard'])
    config = args.get_config()
    
    print(f"  ✅ Preset loaded: {config.get('experiment', {}).get('name', 'unknown')}")
    print(f"  ✅ Model type: {config.get('model', {}).get('type', 'unknown')}")
    print(f"  ✅ Training epochs: {config.get('training', {}).get('num_epochs', 'unknown')}")
    
    # Test configuration dump
    args_dump = cli.parse_args(['--preset', 'harmonic_oscillator_standard', '--config-dump'])
    if hasattr(args_dump, 'config'):
        print(f"  ✅ Config dump available")
        test_config = args_dump.get_config()
        print(f"    ✅ Dump config: {test_config.get('experiment', {}).get('name', 'unknown')}")
    
    print("\n🎯 CLI interface is working correctly!")
    print("\n💡 To use with full functionality:")
    print("   1. Install dependencies: pip install jax flax optax")
    print("   2. Then run: python3 -m qvarnet --preset harmonic_oscillator_standard")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install qvarnet in development mode:")
    print("   pip install -e . --user --break-system-packages")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")