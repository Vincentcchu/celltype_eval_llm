# Cell-Type Label Standardization and Evaluation

A Python repository for standardizing cell-type labels to a lab-defined Level 3 (L3) taxonomy and evaluating model predictions after standardization. Built for AnnData `.h5ad` formats with LLM-powered semantic mapping.

## Overview

This repo provides:

- **Workflow 1**: Standalone standardization - Map raw cell-type labels to L3 taxonomy and update persistent mapping store
- **Workflow 2**: Evaluation - Evaluate predictions vs ground truth after L3 standardization
- **Persistent Mapping Store**: Shared mapping database that grows over time, minimizing LLM calls
- **LLM Semantic Judge**: OpenAI-powered semantic matching for unmapped labels
- **CLI and Python API**: Flexible interfaces for both command-line and programmatic use

## Key Features

✅ **L3 Taxonomy**: Canonical target label space with 23 cell-type categories  
✅ **Shared Mapping Logic**: Both workflows use the same deterministic mapping system  
✅ **Persistent Storage**: Mappings saved to JSON, reused across runs  
✅ **LLM-Powered**: GPT-4o-mini semantic matching for new labels (with fallback strategies)  
✅ **Thread-Safe**: File locking for concurrent access  
✅ **Comprehensive Metrics**: Accuracy, F1 scores, confusion matrices, per-class metrics  

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd celltype_eval_llm

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-openai-api-key"
```

## Quick Start

### Python API

#### Workflow 1: Standardize Labels

```python
from celltype_standardizer import standardize_h5ad_and_update_mapping

# Standardize raw labels to L3 taxonomy
adata = standardize_h5ad_and_update_mapping(
    input_h5ad="data/dataset.h5ad",
    obs_column="cell_type",
    output_h5ad="data/dataset_l3.h5ad",
    output_obs_column="cell_type_level3"
)
```

#### Workflow 2: Evaluate Predictions

```python
from celltype_standardizer import evaluate_h5ad

# Evaluate predictions after L3 standardization
metrics = evaluate_h5ad(
    pred_h5ad="data/predictions.h5ad",
    pred_column="predicted_type",
    gt_h5ad="data/ground_truth.h5ad",
    gt_column="true_type",
    metrics_output_path="results/metrics.json"
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1: {metrics['macro_f1']:.4f}")
```

### Command-Line Interface

#### Standardize Labels

```bash
python -m celltype_standardizer.cli standardize \
  -i data/dataset.h5ad \
  -c cell_type \
  -o data/dataset_l3.h5ad \
  --output-column cell_type_level3
```

#### Evaluate Predictions

```bash
python -m celltype_standardizer.cli evaluate \
  -p data/predictions.h5ad \
  --pred-column predicted_type \
  -g data/ground_truth.h5ad \
  --gt-column true_type \
  -o results/metrics.json
```

#### Check Label Coverage

```bash
python -m celltype_standardizer.cli coverage \
  -i data/dataset.h5ad \
  -c cell_type \
  -o coverage_report.json
```

## Repository Structure

```
celltype_eval_llm/
├── celltype_standardizer/      # Main package
│   ├── __init__.py             # Package exports
│   ├── standardize.py          # Workflow 1: Standardization
│   ├── evaluate.py             # Workflow 2: Evaluation
│   ├── mapping_store.py        # Persistent mapping storage
│   ├── llm_judge.py            # LLM semantic matching
│   └── cli.py                  # Command-line interface
├── mappings/                   # Mapping data
│   ├── l3_vocabulary.json      # L3 taxonomy definition
│   └── label_mappings.json     # Persistent mapping store
├── config/                     # Configuration files
│   ├── config.json             # Main config
│   └── README.md               # Config documentation
├── examples/                   # Usage examples
│   └── demo.ipynb              # Jupyter notebook demo
├── data/                       # Data files
│   ├── datasetGT_debug.h5ad    # Debug ground truth
│   └── datasetTest_debug.h5ad  # Debug test data
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## L3 Taxonomy

The Level 3 (L3) taxonomy includes 23 canonical cell-type labels:

- Malignant
- Benign
- Non neoplastic
- Fibroblasts
- Endothelial
- Myo-fibroblasts
- Fibers
- Adipocytes
- Leukocytes
- Lymphoid cells
- Macrophages/Histiocytes
- Mast cells
- Trombocytes
- Erythrocytes
- Oligodendrocytes
- Microglia
- Astrocyte
- OPC
- Ependymal cells
- Schwann cells
- Ganglion cells
- Apoptotic
- Necrotic

See [mappings/l3_vocabulary.json](mappings/l3_vocabulary.json) for the canonical definition.

## How It Works

### Mapping System

1. **Extract unique labels** from AnnData obs column
2. **Check mapping store** for existing mappings
3. **For unmapped labels**:
   - Try exact match (case-insensitive)
   - Call LLM semantic judge to select best L3 match
   - Use structured JSON output with confidence scores
   - Fallback to fuzzy matching if LLM fails
4. **Update mapping store** with new mappings
5. **Apply standardization** deterministically

### Guardrails

- ✅ Deterministic exact matching before LLM
- ✅ Constrained LLM output (must select from L3 vocabulary)
- ✅ Fallback fuzzy matching if LLM fails
- ✅ Logging of all new mappings
- ✅ Thread-safe file locking for concurrent access
- ✅ Validation of LLM-selected labels

## API Reference

### `standardize_h5ad_and_update_mapping()`

Standardize cell-type labels to L3 taxonomy.

**Parameters:**
- `input_h5ad`: Path to input .h5ad or AnnData object
- `obs_column`: Column name with raw labels
- `output_h5ad`: Optional output path
- `output_obs_column`: Column name for L3 labels (default: "cell_type_level3")
- `mapping_store_path`: Custom mapping store path
- `l3_vocab_path`: Custom L3 vocabulary path
- `api_key`: OpenAI API key
- `llm_model`: Model name (default: "gpt-4o-mini")
- `skip_llm`: Skip LLM calls, use existing mappings only

**Returns:** AnnData with standardized labels

### `evaluate_h5ad()`

Evaluate predictions vs ground truth after L3 standardization.

**Parameters:**
- `pred_h5ad`: Predictions .h5ad path or AnnData
- `pred_column`: Prediction column name
- `gt_h5ad`: Ground truth .h5ad path or AnnData (optional if in same file)
- `gt_column`: Ground truth column name
- `metrics_output_path`: Optional JSON output path
- `mapping_store_path`: Custom mapping store path
- `l3_vocab_path`: Custom L3 vocabulary path
- `api_key`: OpenAI API key
- `llm_model`: Model name (default: "gpt-4o-mini")
- `skip_llm`: Skip LLM calls, use existing mappings only

**Returns:** Dictionary with evaluation metrics:
- `accuracy`: Overall accuracy
- `macro_f1`: Macro-averaged F1
- `weighted_f1`: Weighted-averaged F1
- `confusion_matrix`: Confusion matrix with labels
- `per_class_metrics`: Per-class precision/recall/F1
- `label_counts`: Counts for predictions and ground truth

### `get_label_coverage_report()`

Generate coverage report showing mapped/unmapped labels.

**Parameters:**
- `input_h5ad`: Path to .h5ad or AnnData
- `obs_column`: Column with labels
- `mapping_store_path`: Custom mapping store path

**Returns:** Dictionary with coverage statistics

## Configuration

### Environment Variables

```bash
# Required for LLM semantic matching
export OPENAI_API_KEY="your-api-key-here"
```

### Config File

Edit `config/config.json`:

```json
{
  "openai": {
    "api_key": "YOUR_OPENAI_API_KEY_HERE",
    "model": "gpt-4o-mini"
  },
  "paths": {
    "mapping_store": "mappings/label_mappings.json",
    "l3_vocabulary": "mappings/l3_vocabulary.json"
  }
}
```

## Examples

See [examples/demo.ipynb](examples/demo.ipynb) for a complete walkthrough:

- Loading debug datasets
- Checking label coverage
- Standardizing ground truth labels
- Evaluating predictions
- Visualizing results (confusion matrices, per-class metrics)
- Inspecting the mapping store

## Development

### Testing

```bash
# Run with debug datasets
python -m celltype_standardizer.cli coverage \
  -i data/datasetGT_debug.h5ad \
  -c cell_type
```

### Extending

- **Add new L3 labels**: Edit `mappings/l3_vocabulary.json`
- **Custom LLM prompts**: Modify `celltype_standardizer/llm_judge.py`
- **New metrics**: Extend `celltype_standardizer/evaluate.py`
- **Alternative storage**: Implement new backend in `mapping_store.py`

## Limitations

- L3 vocabulary is flat (no hierarchy support required by design)
- LLM calls require internet connection and API credits
- Thread-safe within single machine (no distributed locking)
- Assumes cell indices align between prediction and ground truth datasets

## Troubleshooting

**"Column not found" error**: Check available columns with `adata.obs.columns`

**API key error**: Set `OPENAI_API_KEY` environment variable or pass `api_key` parameter

**UNMAPPED labels**: Check logs, verify L3 vocabulary, or manually add mappings to `label_mappings.json`

**Slow LLM calls**: Use `skip_llm=True` for testing, or pre-populate mapping store

## License

[Specify your license]

## Citation

If you use this repo, please cite:

```
[Your citation here]
```

## Contact

[Your contact information]

---

**Note**: This repo prioritizes `.h5ad` (AnnData) format. Support for other formats can be added as needed.
