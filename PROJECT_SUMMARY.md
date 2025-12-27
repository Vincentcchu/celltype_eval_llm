# Project Summary

## Cell-Type Label Standardization & Evaluation Repository

This repository provides a complete solution for standardizing cell-type labels to the L3 taxonomy and evaluating model predictions after standardization.

## What Was Built

### Core Functionality

1. **Workflow 1: Standardization** (`celltype_standardizer/standardize.py`)
   - Standalone label standardization to L3 taxonomy
   - Updates persistent mapping store
   - Supports batch processing
   - Coverage reporting

2. **Workflow 2: Evaluation** (`celltype_standardizer/evaluate.py`)
   - Standardizes predictions and ground truth to L3
   - Computes comprehensive metrics (accuracy, F1, confusion matrix)
   - Per-class performance metrics
   - Saves detailed JSON reports

3. **Persistent Mapping Store** (`celltype_standardizer/mapping_store.py`)
   - Thread-safe JSON storage with file locking
   - Automatically grows over time
   - Reduces LLM API calls
   - Batch operations support

4. **LLM Semantic Judge** (`celltype_standardizer/llm_judge.py`)
   - OpenAI GPT-4o-mini integration
   - Structured JSON output with confidence scores
   - Exact matching before LLM calls
   - Fallback fuzzy matching
   - Constrained to L3 vocabulary

5. **Command-Line Interface** (`celltype_standardizer/cli.py`)
   - Three commands: standardize, evaluate, coverage
   - Full parameter control
   - Progress logging

### Supporting Files

- **L3 Vocabulary** (`mappings/l3_vocabulary.json`) - 23 canonical cell-type labels
- **Mapping Store** (`mappings/label_mappings.json`) - Persistent raw→L3 mappings
- **Configuration** (`config/config.json`) - API keys and default settings
- **Example Notebook** (`examples/demo.ipynb`) - Complete walkthrough with visualizations
- **Documentation**:
  - `README.md` - Main documentation with API reference
  - `USAGE.md` - Comprehensive usage guide
  - `config/README.md` - Configuration help
- **Setup Tools**:
  - `setup.sh` - Automated setup script
  - `test_installation.py` - Installation verification
  - `requirements.txt` - Python dependencies
  - `pyproject.toml` - Package configuration
  - `.gitignore` - Version control ignores

## Directory Structure

```
celltype_eval_llm/
├── celltype_standardizer/         # Main package
│   ├── __init__.py               # Package exports
│   ├── standardize.py            # Workflow 1 implementation
│   ├── evaluate.py               # Workflow 2 implementation
│   ├── mapping_store.py          # Persistent storage
│   ├── llm_judge.py              # LLM semantic matching
│   └── cli.py                    # Command-line interface
├── mappings/                     # Mapping data
│   ├── l3_vocabulary.json        # L3 taxonomy (23 labels)
│   └── label_mappings.json       # Persistent store (empty initially)
├── config/                       # Configuration
│   ├── config.json               # Main config template
│   └── README.md                 # Config documentation
├── examples/                     # Usage examples
│   └── demo.ipynb                # Jupyter notebook demo
├── data/                         # Data directory
│   ├── datasetGT_debug.h5ad      # Debug ground truth
│   └── datasetTest_debug.h5ad    # Debug test data
├── README.md                     # Main documentation
├── USAGE.md                      # Detailed usage guide
├── requirements.txt              # Dependencies
├── pyproject.toml                # Package metadata
├── setup.sh                      # Setup script
├── test_installation.py          # Verification script
└── .gitignore                    # Git ignores
```

## Key Design Features

### ✅ Requirements Met

- [x] **L3 as single target taxonomy** - All standardization maps to 23 L3 labels
- [x] **Flat L3 structure** - No dependency on parent/child hierarchy
- [x] **Shared mapping system** - Both workflows use same MappingStore
- [x] **Persistent storage** - Mappings saved to JSON, reused across runs
- [x] **LLM semantic judge** - OpenAI integration with structured output
- [x] **Deterministic matching** - Exact matches before LLM calls
- [x] **AnnData .h5ad priority** - Native support for single-cell format
- [x] **Workflow 1: Standalone standardization** - Independent function
- [x] **Workflow 2: Evaluation** - Standardize → metrics
- [x] **CLI + Python API** - Both interfaces implemented
- [x] **Thread-safe** - File locking for concurrent access
- [x] **Comprehensive metrics** - Accuracy, F1, confusion matrix, per-class
- [x] **Coverage reporting** - Check mapped/unmapped labels
- [x] **Guardrails** - Validation, fallbacks, logging

### 🎯 Core Algorithms

1. **Mapping Resolution**:
   ```
   1. Check persistent store for exact match
   2. If not found → LLM semantic judge
   3. LLM tries exact match (case-insensitive)
   4. LLM selects from L3 vocabulary with confidence
   5. Validate LLM output is in L3 vocabulary
   6. Fallback to fuzzy matching if LLM fails
   7. Save new mapping to persistent store
   ```

2. **Standardization Workflow**:
   ```
   1. Load AnnData from .h5ad
   2. Extract unique labels from specified column
   3. Check mapping store for existing mappings
   4. For unmapped: call LLM semantic judge
   5. Update mapping store with new mappings
   6. Apply all mappings to create L3 column
   7. Optionally save standardized .h5ad
   ```

3. **Evaluation Workflow**:
   ```
   1. Load prediction and ground truth data
   2. Standardize predictions to L3
   3. Standardize ground truth to L3
   4. Update mapping store for any new labels
   5. Compute metrics on L3 labels
   6. Generate detailed report
   7. Save metrics to JSON
   ```

## Usage Examples

### Python API

```python
from celltype_standardizer import standardize_h5ad_and_update_mapping, evaluate_h5ad

# Standardize ground truth
adata = standardize_h5ad_and_update_mapping(
    input_h5ad="data/dataset.h5ad",
    obs_column="cell_type",
    output_h5ad="data/dataset_l3.h5ad"
)

# Evaluate predictions
metrics = evaluate_h5ad(
    pred_h5ad="data/predictions.h5ad",
    pred_column="predicted",
    gt_h5ad="data/ground_truth_l3.h5ad",
    gt_column="cell_type_level3"
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### CLI

```bash
# Standardize
python -m celltype_standardizer.cli standardize \
  -i data/dataset.h5ad -c cell_type -o data/dataset_l3.h5ad

# Check coverage
python -m celltype_standardizer.cli coverage \
  -i data/dataset.h5ad -c cell_type

# Evaluate
python -m celltype_standardizer.cli evaluate \
  -p data/predictions.h5ad --pred-column predicted \
  -g data/ground_truth.h5ad --gt-column true_type \
  -o results/metrics.json
```

## Getting Started

### Quick Setup

```bash
cd celltype_eval_llm
./setup.sh
export OPENAI_API_KEY="your-api-key"
python test_installation.py
```

### Try the Demo

```bash
jupyter notebook examples/demo.ipynb
```

### Read Documentation

- **README.md** - API reference and overview
- **USAGE.md** - Detailed usage examples and troubleshooting
- **examples/demo.ipynb** - Interactive tutorial

## Technical Details

### Dependencies

- **anndata** - Single-cell data format
- **scanpy** - Single-cell analysis (transitively used)
- **pandas/numpy** - Data manipulation
- **scikit-learn** - Metrics computation
- **openai** - LLM integration
- **matplotlib/seaborn** - Visualization (optional)

### LLM Configuration

- **Default model**: gpt-4o-mini (fast, cost-effective)
- **Temperature**: 0.0 (deterministic)
- **Output format**: JSON structured with selected_label, confidence, rationale
- **Validation**: Selected label must be in L3 vocabulary

### Mapping Store Format

```json
{
  "version": "1.0",
  "mappings": {
    "raw_label_1": "L3_Label_1",
    "raw_label_2": "L3_Label_2"
  },
  "metadata": {
    "last_updated": "2025-12-27T10:30:00",
    "total_mappings": 2
  }
}
```

### Metrics Output Format

```json
{
  "accuracy": 0.8765,
  "macro_f1": 0.8234,
  "weighted_f1": 0.8543,
  "confusion_matrix": {
    "labels": ["Label1", "Label2", ...],
    "matrix": [[100, 5], [2, 98], ...]
  },
  "per_class_metrics": {
    "Label1": {
      "precision": 0.92,
      "recall": 0.89,
      "f1_score": 0.90,
      "support": 1234
    }
  }
}
```

## Future Extensions

Possible enhancements:

1. **Additional LLM providers** (Anthropic, local models)
2. **Hierarchical evaluation** (if L3 hierarchy populated)
3. **Other file formats** (Seurat, Loom, CSV)
4. **Confidence filtering** (use LLM confidence scores)
5. **Manual review interface** (UI for mapping review)
6. **Batch API optimization** (parallel LLM calls)
7. **Mapping versioning** (track mapping changes over time)
8. **Custom metrics** (domain-specific evaluation)

## Verification

Run verification script:

```bash
python test_installation.py
```

Expected output:
```
✓ All tests passed! Installation is working correctly.
```

## Support

- Check **USAGE.md** for troubleshooting
- Review **examples/demo.ipynb** for working examples
- Inspect logs for detailed error messages
- Verify API key is set: `echo $OPENAI_API_KEY`

## Project Status

✅ **Complete** - All requirements implemented and tested

- Workflow 1: Standardization ✓
- Workflow 2: Evaluation ✓
- Persistent mapping store ✓
- LLM semantic judge ✓
- CLI interface ✓
- Python API ✓
- Documentation ✓
- Examples ✓
- Setup tools ✓

---

**Built**: December 27, 2025  
**Version**: 1.0.0  
**Status**: Production Ready
