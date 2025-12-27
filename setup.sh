#!/bin/bash

# Setup script for celltype_eval_llm repository

set -e

echo "=================================="
echo "Cell-Type Standardization Setup"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment (optional)
read -p "Create a virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Virtual environment activated"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✓ Dependencies installed"

# Check for OpenAI API key
echo ""
echo "Checking for OpenAI API key..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY environment variable not set"
    echo ""
    echo "To use LLM semantic matching, you need to set your API key:"
    echo "  export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    echo "Or add it to config/config.json"
else
    echo "✓ OPENAI_API_KEY is set"
fi

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import celltype_standardizer; print('✓ celltype_standardizer package imported successfully')"
python3 -c "import anndata; print('✓ anndata imported')"
python3 -c "import sklearn; print('✓ scikit-learn imported')"
python3 -c "import openai; print('✓ openai imported')"

# Check data files
echo ""
echo "Checking for debug datasets..."
if [ -f "data/datasetGT_debug.h5ad" ]; then
    echo "✓ Found data/datasetGT_debug.h5ad"
else
    echo "⚠️  data/datasetGT_debug.h5ad not found"
fi

if [ -f "data/datasetTest_debug.h5ad" ]; then
    echo "✓ Found data/datasetTest_debug.h5ad"
else
    echo "⚠️  data/datasetTest_debug.h5ad not found"
fi

# Test CLI
echo ""
echo "Testing CLI..."
python3 -m celltype_standardizer.cli --help > /dev/null 2>&1 && echo "✓ CLI is working"

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'"
echo "2. Try the example notebook: jupyter notebook examples/demo.ipynb"
echo "3. Or use the CLI:"
echo "   python -m celltype_standardizer.cli --help"
echo ""
echo "For detailed usage, see README.md"
echo ""
