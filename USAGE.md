# Usage Guide

Comprehensive guide for using the cell-type standardization and evaluation system.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Workflow 1: Standardization](#workflow-1-standardization)
4. [Workflow 2: Evaluation](#workflow-2-evaluation)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

## Installation

### Option 1: Using setup script (recommended)

```bash
cd celltype_eval_llm
./setup.sh
```

### Option 2: Manual installation

```bash
# Create virtual environment (optional)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### Python API Example

```python
from celltype_standardizer import standardize_h5ad_and_update_mapping, evaluate_h5ad

# Standardize labels
adata = standardize_h5ad_and_update_mapping(
    input_h5ad="data/dataset.h5ad",
    obs_column="cell_type"
)

# Evaluate predictions
metrics = evaluate_h5ad(
    pred_h5ad="data/predictions.h5ad",
    pred_column="predicted",
    gt_column="ground_truth"
)
```

### CLI Example

```bash
# Standardize
python -m celltype_standardizer.cli standardize \
  -i data/dataset.h5ad -c cell_type

# Evaluate
python -m celltype_standardizer.cli evaluate \
  -p data/predictions.h5ad --pred-column predicted \
  --gt-column ground_truth
```

## Workflow 1: Standardization

### Use Case

Standardize cell-type labels from multiple sources to a common L3 taxonomy before using them as ground truth or for analysis.

### Step 1: Check Label Coverage

Before standardization, check how many labels already have mappings:

```python
from celltype_standardizer.standardize import get_label_coverage_report

report = get_label_coverage_report(
    input_h5ad="data/dataset.h5ad",
    obs_column="cell_type"
)

print(f"Coverage: {report['coverage_percent']:.1f}%")
print(f"Unmapped labels: {len(report['unmapped_labels'])}")
```

CLI:
```bash
python -m celltype_standardizer.cli coverage \
  -i data/dataset.h5ad -c cell_type -o coverage_report.json
```

### Step 2: Standardize Labels

```python
adata = standardize_h5ad_and_update_mapping(
    input_h5ad="data/raw_dataset.h5ad",
    obs_column="cell_type",                    # Column with raw labels
    output_h5ad="data/standardized.h5ad",      # Save standardized dataset
    output_obs_column="cell_type_level3",      # New column name
    skip_llm=False                              # Use LLM for unmapped labels
)

# Check results
print(adata.obs['cell_type_level3'].value_counts())
```

CLI:
```bash
python -m celltype_standardizer.cli standardize \
  -i data/raw_dataset.h5ad \
  -c cell_type \
  -o data/standardized.h5ad \
  --output-column cell_type_level3
```

### Important Notes

- **First run**: LLM will be called for all unmapped labels
- **Subsequent runs**: Only new labels trigger LLM calls
- **Mapping store**: Updated automatically at `mappings/label_mappings.json`
- **Skip LLM**: Use `skip_llm=True` for testing without API calls

## Workflow 2: Evaluation

### Use Case

Evaluate model predictions against ground truth after standardizing both to L3.

### Scenario A: Predictions and GT in same file

```python
metrics = evaluate_h5ad(
    pred_h5ad="data/results.h5ad",
    pred_column="predicted_type",
    gt_column="true_type",
    metrics_output_path="results/metrics.json"
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1: {metrics['macro_f1']:.4f}")
```

### Scenario B: Separate prediction and GT files

```python
metrics = evaluate_h5ad(
    pred_h5ad="data/predictions.h5ad",
    pred_column="predicted_type",
    gt_h5ad="data/ground_truth.h5ad",
    gt_column="true_type",
    metrics_output_path="results/metrics.json"
)
```

CLI:
```bash
python -m celltype_standardizer.cli evaluate \
  -p data/predictions.h5ad --pred-column predicted_type \
  -g data/ground_truth.h5ad --gt-column true_type \
  -o results/metrics.json
```

### Understanding Metrics Output

The metrics dictionary contains:

```python
{
    "accuracy": 0.8765,                    # Overall accuracy
    "macro_f1": 0.8234,                    # Macro-averaged F1
    "weighted_f1": 0.8543,                 # Weighted F1
    "confusion_matrix": {
        "labels": [...],                    # L3 labels
        "matrix": [[...], [...], ...]      # Confusion matrix
    },
    "per_class_metrics": {
        "Endothelial": {
            "precision": 0.92,
            "recall": 0.89,
            "f1_score": 0.90,
            "support": 1234
        },
        ...
    },
    "label_counts": {
        "predictions": {...},               # Count per label
        "ground_truth": {...}
    },
    "dataset_info": {
        "total_cells": 10000,
        "unique_pred_labels": 15,
        "standardized_pred_labels": 12,
        "unmapped_predictions": 0
    }
}
```

## Advanced Usage

### Custom Mapping Store Location

```python
# Use a custom mapping store
adata = standardize_h5ad_and_update_mapping(
    input_h5ad="data/dataset.h5ad",
    obs_column="cell_type",
    mapping_store_path="custom/mappings.json"
)
```

### Custom L3 Vocabulary

```python
# Use a modified L3 vocabulary
adata = standardize_h5ad_and_update_mapping(
    input_h5ad="data/dataset.h5ad",
    obs_column="cell_type",
    l3_vocab_path="custom/l3_vocab.json"
)
```

### Different LLM Model

```python
# Use a different OpenAI model
adata = standardize_h5ad_and_update_mapping(
    input_h5ad="data/dataset.h5ad",
    obs_column="cell_type",
    llm_model="gpt-4"  # More powerful but slower/expensive
)
```

### Batch Processing Multiple Files

```python
import glob
from pathlib import Path

# Standardize all datasets in a directory
for file_path in glob.glob("data/raw/*.h5ad"):
    output_path = file_path.replace("raw", "standardized")
    
    adata = standardize_h5ad_and_update_mapping(
        input_h5ad=file_path,
        obs_column="cell_type",
        output_h5ad=output_path
    )
    
    print(f"Processed {file_path} -> {output_path}")
```

### Working with Mapping Store Directly

```python
from celltype_standardizer.mapping_store import MappingStore

# Load mapping store
store = MappingStore()

# Get statistics
stats = store.get_stats()
print(f"Total mappings: {stats['total_mappings']}")

# Get all mappings
mappings = store.get_all_mappings()
for raw, l3 in mappings.items():
    print(f"{raw} -> {l3}")

# Check specific mapping
l3_label = store.get_mapping("T cell")
print(f"T cell maps to: {l3_label}")

# Add custom mapping manually
store.add_mapping("NK cell", "Lymphoid cells")
```

### Visualizing Results in Jupyter

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Get evaluation metrics
metrics = evaluate_h5ad(...)

# Plot confusion matrix
cm = np.array(metrics['confusion_matrix']['matrix'])
labels = metrics['confusion_matrix']['labels']

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - L3 Labels')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot per-class F1 scores
import pandas as pd
per_class = pd.DataFrame(metrics['per_class_metrics']).T
per_class['f1_score'].sort_values().plot(kind='barh', figsize=(10, 8))
plt.xlabel('F1 Score')
plt.title('F1 Score by Cell Type')
plt.tight_layout()
plt.show()
```

## Troubleshooting

### Issue: "OPENAI_API_KEY not found"

**Solution**: Set environment variable or pass API key explicitly:

```python
# Option 1: Environment variable
export OPENAI_API_KEY="your-key"

# Option 2: Pass directly
adata = standardize_h5ad_and_update_mapping(
    input_h5ad="...",
    obs_column="...",
    api_key="your-key-here"
)
```

### Issue: "Column 'X' not found in adata.obs"

**Solution**: Check available columns:

```python
import anndata as ad
adata = ad.read_h5ad("data/dataset.h5ad")
print(adata.obs.columns.tolist())
```

### Issue: Many labels remain "UNMAPPED"

**Causes**:
1. `skip_llm=True` was used
2. LLM failed to map some labels
3. API quota exceeded

**Solution**:
- Check logs for errors
- Ensure `skip_llm=False`
- Manually add mappings to `mappings/label_mappings.json`:

```json
{
  "mappings": {
    "problematic_label": "Appropriate L3 Label"
  }
}
```

### Issue: LLM returns invalid label

**Cause**: LLM hallucinated a label not in L3 vocabulary

**Solution**: System has automatic fallback. Check logs and manually correct if needed.

### Issue: Slow evaluation

**Causes**:
1. First run with many unmapped labels
2. Large datasets

**Solutions**:
- Pre-standardize ground truth datasets (Workflow 1)
- Use `skip_llm=True` for testing
- Reuse mapping store across datasets

### Issue: Thread safety / concurrent access

**Solution**: Built-in file locking handles concurrent access. If using distributed systems, implement distributed locking.

## Best Practices

1. **Pre-standardize ground truth**: Run Workflow 1 on GT datasets once, reuse standardized versions
2. **Incremental mapping**: Let mapping store grow over time across projects
3. **Verify new mappings**: Check logs after first run on new datasets
4. **Use coverage reports**: Check coverage before standardization
5. **Save metrics**: Always save evaluation metrics for reproducibility
6. **Version control**: Track `label_mappings.json` changes
7. **API costs**: Monitor OpenAI usage, use caching effectively

## Performance Tips

- **Batch datasets**: Group similar datasets to benefit from shared mappings
- **Skip LLM for testing**: Use `skip_llm=True` during development
- **Smaller model**: Use "gpt-4o-mini" (default) instead of "gpt-4"
- **Cache mappings**: Mapping store automatically caches results

## Next Steps

- See [examples/demo.ipynb](examples/demo.ipynb) for complete examples
- Check [README.md](README.md) for API reference
- Review [mappings/l3_vocabulary.json](mappings/l3_vocabulary.json) for L3 taxonomy
