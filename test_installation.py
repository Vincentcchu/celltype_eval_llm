"""
Quick verification script to test the installation.
Run this after setup to ensure everything is working.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import anndata
        print("✓ anndata")
    except ImportError as e:
        print(f"✗ anndata: {e}")
        return False
    
    try:
        import pandas
        print("✓ pandas")
    except ImportError as e:
        print(f"✗ pandas: {e}")
        return False
    
    try:
        import numpy
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn")
    except ImportError as e:
        print(f"✗ scikit-learn: {e}")
        return False
    
    try:
        import openai
        print("✓ openai")
    except ImportError as e:
        print(f"✗ openai: {e}")
        return False
    
    try:
        import celltype_standardizer
        print("✓ celltype_standardizer")
    except ImportError as e:
        print(f"✗ celltype_standardizer: {e}")
        return False
    
    return True


def test_modules():
    """Test that package modules are accessible."""
    print("\nTesting package modules...")
    
    try:
        from celltype_standardizer import standardize_h5ad_and_update_mapping
        print("✓ standardize_h5ad_and_update_mapping")
    except ImportError as e:
        print(f"✗ standardize_h5ad_and_update_mapping: {e}")
        return False
    
    try:
        from celltype_standardizer import evaluate_h5ad
        print("✓ evaluate_h5ad")
    except ImportError as e:
        print(f"✗ evaluate_h5ad: {e}")
        return False
    
    try:
        from celltype_standardizer.mapping_store import MappingStore
        print("✓ MappingStore")
    except ImportError as e:
        print(f"✗ MappingStore: {e}")
        return False
    
    try:
        from celltype_standardizer.llm_judge import LLMSemanticJudge
        print("✓ LLMSemanticJudge")
    except ImportError as e:
        print(f"✗ LLMSemanticJudge: {e}")
        return False
    
    return True


def test_files():
    """Test that required files exist."""
    print("\nTesting required files...")
    
    required_files = [
        "mappings/l3_vocabulary.json",
        "mappings/label_mappings.json",
        "config/config.json",
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            all_exist = False
    
    return all_exist


def test_data_files():
    """Check for data files."""
    print("\nChecking data files...")
    
    data_files = [
        "data/datasetGT_debug.h5ad",
        "data/datasetTest_debug.h5ad",
    ]
    
    for file_path in data_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"⚠ {file_path} (not found - optional)")


def test_mapping_store():
    """Test mapping store functionality."""
    print("\nTesting mapping store...")
    
    try:
        from celltype_standardizer.mapping_store import MappingStore
        
        store = MappingStore()
        stats = store.get_stats()
        print(f"✓ Mapping store initialized")
        print(f"  - Total mappings: {stats['total_mappings']}")
        print(f"  - File: {stats['file_path']}")
        return True
    except Exception as e:
        print(f"✗ Mapping store test failed: {e}")
        return False


def test_vocabulary():
    """Test L3 vocabulary loading."""
    print("\nTesting L3 vocabulary...")
    
    try:
        from celltype_standardizer.llm_judge import L3Vocabulary
        
        vocab = L3Vocabulary()
        labels = vocab.get_labels()
        print(f"✓ L3 vocabulary loaded")
        print(f"  - {len(labels)} L3 labels defined")
        print(f"  - Sample labels: {', '.join(labels[:5])}")
        return True
    except Exception as e:
        print(f"✗ Vocabulary test failed: {e}")
        return False


def check_api_key():
    """Check if OpenAI API key is set."""
    print("\nChecking OpenAI API key...")
    
    import os
    if "OPENAI_API_KEY" in os.environ:
        print("✓ OPENAI_API_KEY environment variable is set")
        return True
    else:
        print("⚠ OPENAI_API_KEY not set (required for LLM semantic matching)")
        print("  Set with: export OPENAI_API_KEY='your-key-here'")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Cell-Type Standardizer Installation Verification")
    print("="*60)
    print()
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Modules", test_modules()))
    results.append(("Files", test_files()))
    test_data_files()  # Optional, doesn't affect pass/fail
    results.append(("Mapping Store", test_mapping_store()))
    results.append(("Vocabulary", test_vocabulary()))
    check_api_key()  # Optional, doesn't affect pass/fail
    
    print()
    print("="*60)
    print("Test Summary")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("✓ All tests passed! Installation is working correctly.")
        print()
        print("Next steps:")
        print("1. Set OPENAI_API_KEY if not already set")
        print("2. Try the example notebook: jupyter notebook examples/demo.ipynb")
        print("3. Or use the CLI: python -m celltype_standardizer.cli --help")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
