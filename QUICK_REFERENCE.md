# Quick Reference Card

## Installation

```bash
cd celltype_eval_llm
./setup.sh
export OPENAI_API_KEY="your-key"
python test_installation.py
```

## Python API - Essential Functions

### Standardize Labels to L3

```python
from celltype_standardizer import standardize_h5ad_and_update_mapping

adata = standardize_h5ad_and_update_mapping(
    input_h5ad="data/dataset.h5ad",
    obs_column="cell_type",
    output_h5ad="data/dataset_l3.h5ad",
    output_obs_column="cell_type_level3"
)
```

### Evaluate Predictions

```python
from celltype_standardizer import evaluate_h5ad

metrics = evaluate_h5ad(
    pred_h5ad="data/predictions.h5ad",
    pred_column="predicted_type",
    gt_h5ad="data/ground_truth.h5ad",
    gt_column="true_type",
    metrics_output_path="results/metrics.json"
)
```

### Check Coverage

```python
from celltype_standardizer.standardize import get_label_coverage_report

report = get_label_coverage_report(
    input_h5ad="data/dataset.h5ad",
    obs_column="cell_type"
)
print(f"Coverage: {report['coverage_percent']:.1f}%")
```

## CLI Commands

### Standardize

```bash
python -m celltype_standardizer.cli standardize \
  -i data/dataset.h5ad \
  -c cell_type \
  -o data/dataset_l3.h5ad
```

### Evaluate

```bash
python -m celltype_standardizer.cli evaluate \
  -p data/predictions.h5ad --pred-column predicted \
  -g data/ground_truth.h5ad --gt-column true_type \
  -o results/metrics.json
```

### Coverage

```bash
python -m celltype_standardizer.cli coverage \
  -i data/dataset.h5ad \
  -c cell_type
```

## Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_h5ad` | Path to input .h5ad file | Required |
| `obs_column` | Column name with labels | Required |
| `output_h5ad` | Path to save output | None (don't save) |
| `output_obs_column` | Column name for L3 labels | "cell_type_level3" |
| `skip_llm` | Skip LLM calls | False |
| `api_key` | OpenAI API key | $OPENAI_API_KEY |
| `llm_model` | OpenAI model | "gpt-4o-mini" |

## L3 Taxonomy (23 Labels)

```
Malignant                  Leukocytes
Benign                     Lymphoid cells
Non neoplastic            Macrophages/Histiocytes
Fibroblasts               Mast cells
Endothelial               Trombocytes
Myo-fibroblasts           Erythrocytes
Fibers                    Oligodendrocytes
Adipocytes                Microglia
Astrocyte                 Ependymal cells
OPC                       Schwann cells
Ganglion cells            Apoptotic
Necrotic
```

## Files & Directories

```
celltype_standardizer/     # Main package
mappings/
  ├── l3_vocabulary.json   # L3 labels
  └── label_mappings.json  # Persistent store
config/
  └── config.json          # Configuration
examples/
  └── demo.ipynb           # Tutorial
data/                      # Your data files
```

## Typical Workflow

```python
# 1. Pre-standardize ground truth (once)
gt_adata = standardize_h5ad_and_update_mapping(
    input_h5ad="data/ground_truth_raw.h5ad",
    obs_column="cell_type",
    output_h5ad="data/ground_truth_l3.h5ad"
)

# 2. Evaluate multiple prediction files
for pred_file in ["model1.h5ad", "model2.h5ad"]:
    metrics = evaluate_h5ad(
        pred_h5ad=f"predictions/{pred_file}",
        pred_column="predicted",
        gt_h5ad="data/ground_truth_l3.h5ad",
        gt_column="cell_type_level3",
        metrics_output_path=f"results/{pred_file}_metrics.json"
    )
    print(f"{pred_file}: Accuracy={metrics['accuracy']:.4f}")
```

## Working with Mapping Store

```python
from celltype_standardizer.mapping_store import MappingStore

store = MappingStore()

# Get all mappings
mappings = store.get_all_mappings()

# Get statistics
stats = store.get_stats()
print(f"Total: {stats['total_mappings']}")

# Add manual mapping
store.add_mapping("NK cell", "Lymphoid cells")
```

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| API key error | `export OPENAI_API_KEY="your-key"` |
| Column not found | Check: `adata.obs.columns.tolist()` |
| UNMAPPED labels | Check logs, add manual mappings |
| Slow LLM calls | Use `skip_llm=True` for testing |

## Metrics Dictionary

```python
metrics = evaluate_h5ad(...)

metrics['accuracy']                    # 0.0-1.0
metrics['macro_f1']                    # 0.0-1.0
metrics['weighted_f1']                 # 0.0-1.0
metrics['confusion_matrix']['matrix']  # 2D array
metrics['per_class_metrics'][label]    # {precision, recall, f1, support}
metrics['dataset_info']['total_cells'] # Count
```

## Environment Variables

```bash
export OPENAI_API_KEY="sk-..."        # Required for LLM
export OPENAI_MODEL="gpt-4o-mini"     # Optional, override model
```

## Advanced Options

```python
# Skip LLM (testing mode)
adata = standardize_h5ad_and_update_mapping(
    input_h5ad="...", obs_column="...",
    skip_llm=True
)

# Custom mapping store location
adata = standardize_h5ad_and_update_mapping(
    input_h5ad="...", obs_column="...",
    mapping_store_path="custom/path/mappings.json"
)

# Use different LLM model
adata = standardize_h5ad_and_update_mapping(
    input_h5ad="...", obs_column="...",
    llm_model="gpt-4"
)
```

## Help Commands

```bash
# Package help
python -m celltype_standardizer.cli --help

# Command-specific help
python -m celltype_standardizer.cli standardize --help
python -m celltype_standardizer.cli evaluate --help
python -m celltype_standardizer.cli coverage --help
```

## Resources

- **Full Documentation**: [README.md](README.md)
- **Usage Guide**: [USAGE.md](USAGE.md)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Example Notebook**: [examples/demo.ipynb](examples/demo.ipynb)
- **Project Summary**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

---

**Version**: 1.0.0  
**Support**: Check USAGE.md for detailed troubleshooting
