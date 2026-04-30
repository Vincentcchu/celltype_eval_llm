# CASSIA Evaluation Pipeline

Batch evaluation script for CASSIA cell type predictions.

## Overview

CASSIA outputs cluster-level predictions in CSV format. This pipeline:
1. Matches ground truth h5ad files with CASSIA output
2. Converts cluster-level predictions to cell-level predictions
3. Evaluates using the standard evaluation pipeline (with LLM-based standardization)

## File Structure

CASSIA expects the following output structure:
```
cell_agents/agents/CASSIA/CASSIA_run/output/
├── <tissue>/
│   ├── Data_<identifier>/
│   │   ├── Data_<identifier>_FINAL_RESULTS.csv       # Cluster predictions
│   │   ├── Data_<identifier>_clustered.h5ad          # Clustered data
│   │   └── intermediate_files/
│   │       └── Data_<identifier>_summary.csv         # Alternative CSV
│   └── ...
└── ...
```

Ground truth files:
```
data/
├── <tissue>/
│   └── h5ad/
│       └── Data_<identifier>.h5ad                     # Ground truth
```

## Usage

### Basic Usage

```bash
# Evaluate all tissues
python run_evaluation_cassia.py

# Evaluate specific tissue(s)
python run_evaluation_cassia.py --tissue hematologic
python run_evaluation_cassia.py --tissue brain breast colorectal
```

### Options

```bash
# Skip LLM calls (use existing mappings only)
python run_evaluation_cassia.py --skip-llm

# Disable plot generation
python run_evaluation_cassia.py --no-plots

# Keep temporary cell-level prediction files
python run_evaluation_cassia.py --keep-temp-files

# Custom output directory
python run_evaluation_cassia.py --output-dir my_results

# Custom data and output roots
python run_evaluation_cassia.py \
    --data-root /path/to/data \
    --output-root /path/to/cassia/output

# Verbose logging
python run_evaluation_cassia.py --verbose
```

### Full Options

```
--tissue TISSUE [TISSUE ...]
    Tissue name(s) to evaluate (e.g., brain, breast).
    If not provided, evaluates all tissues.

--data-root PATH
    Root directory containing tissue data (default: auto-detect)

--output-root PATH
    Root directory containing CASSIA outputs (default: auto-detect)

--output-dir PATH
    Directory to save evaluation results (default: evaluation_results_cassia)

--pred-column STR
    Column name for predictions after conversion (default: predicted_cell_type)

--gt-column STR
    Column name for ground truth in GT files (default: cell_type)

--cluster-column STR
    Column name for cluster assignments in clustered h5ad (default: cluster)

--skip-llm
    Skip LLM calls and only use existing mappings

--no-plots
    Disable plot generation

--keep-temp-files
    Keep temporary h5ad files with cell-level predictions

--verbose
    Enable verbose logging
```

## Output Structure

Results are saved in the output directory (default: `evaluation_results_cassia/`):

```
evaluation_results_cassia/
├── <tissue>_<dataset>/
│   ├── evaluation_metrics.json          # Main metrics
│   ├── overall_metrics.csv              # Summary metrics
│   ├── classification_report.csv        # Per-class metrics
│   ├── confusion_matrix.csv             # Confusion matrix
│   ├── confusion_matrix.png             # Visualization
│   └── umap_comparison.png              # UMAP visualization
├── temp_predictions/                    # Temporary files (deleted unless --keep-temp-files)
│   └── <tissue>/
│       └── <dataset>_cell_predictions.h5ad
└── evaluation_summary.csv               # Aggregate summary (if multiple datasets)
```

## How It Works

### Step 1: File Matching

The script scans for:
- Ground truth files in `data/<tissue>/h5ad/*.h5ad`
- CASSIA output directories in `output/<tissue>/Data_*/`
- Validates presence of both CSV and clustered h5ad files

### Step 2: Cluster-to-Cell Conversion

For each matched dataset:
1. Load cluster predictions from CSV (`Predicted Main Cell Type` column)
2. Load cluster assignments from h5ad (`obs['cluster']`)
3. Map each cell to its cluster's prediction
4. Create temporary h5ad file with cell-level predictions

### Step 3: Evaluation

Uses the standard evaluation pipeline:
1. Load ground truth and predictions
2. Standardize cell type labels using LLM-based mapping (optional)
3. Calculate metrics (accuracy, F1, etc.)
4. Generate visualizations (confusion matrix, UMAP)
5. Save results

## Examples

### Evaluate all hematologic datasets

```bash
python run_evaluation_cassia.py --tissue hematologic
```

Output:
```
============================================================
CASSIA EVALUATION PIPELINE
============================================================
Data root: /path/to/data
CASSIA output root: /path/to/CASSIA_run/output
Results will be saved to: evaluation_results_cassia

============================================================
STEP 1: MATCHING FILES
============================================================
Available tissues: brain, breast, colorectal, head_neck, hematologic, ...
Processing tissues: hematologic
Found 10 matched dataset pairs

============================================================
STEP 2: CONVERTING CLUSTER PREDICTIONS TO CELL LEVEL
============================================================
Processing 1/10: hematologic/Caron2020_Hematologic
✓ Successfully converted 10/10 datasets

============================================================
STEP 3: EVALUATING DATASETS
============================================================
Evaluating: hematologic/Caron2020_Hematologic
  Accuracy:    0.8234
  Macro F1:    0.7891
  Weighted F1: 0.8156
...
```

### Run without LLM calls (faster, uses cached mappings)

```bash
python run_evaluation_cassia.py --tissue brain --skip-llm
```

### Keep temporary files for inspection

```bash
python run_evaluation_cassia.py --tissue breast --keep-temp-files
```

Temporary files will be saved in `evaluation_results_cassia/temp_predictions/`

## Troubleshooting

### "No matched dataset pairs found"

- Check that CASSIA has been run on the tissue
- Verify the output directory structure matches expected format
- Use `--verbose` to see detailed matching logs

### "H5AD missing 'cluster' column"

- Check the cluster column name in the h5ad file
- Use `--cluster-column` to specify a different column name
- Example: `--cluster-column leiden` if using Leiden clustering

### "CSV missing 'True Cell Type' column"

- Verify the CSV file is a CASSIA output file
- Check if using `_FINAL_RESULTS.csv` or `_summary.csv`

### Memory issues with large datasets

- Process datasets one at a time using `--tissue`
- Use `--no-plots` to reduce memory usage
- Consider running on a machine with more RAM

## Related Files

- `batch_evaluation/file_matcher_cassia.py` - CASSIA-specific file matching
- `batch_evaluation/cassia_adapter.py` - Cluster-to-cell conversion logic
- `batch_evaluation/evaluator.py` - Batch evaluation logic
- `celltype_standardizer/` - LLM-based cell type standardization

## Notes

- CASSIA predictions are cluster-level, so all cells in a cluster receive the same prediction
- The evaluation is performed at the cell level by mapping cluster predictions to cells
- LLM-based standardization helps match different cell type naming conventions
- Large datasets (1GB+ h5ad files) may take several minutes to process
