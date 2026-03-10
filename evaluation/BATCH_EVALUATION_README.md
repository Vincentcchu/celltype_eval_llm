# Batch Evaluation Pipeline

Automated pipeline for evaluating cell type predictions across multiple datasets.

## Overview

This pipeline automatically:
1. Matches ground truth files with prediction files
2. Standardizes cell type labels to L3 vocabulary
3. Computes evaluation metrics (accuracy, F1, confusion matrix)
4. Saves results in a structured format with visualizations

## Installation

The pipeline uses the existing `celltype_standardizer` package. Ensure it's installed:

```bash
cd /cs/student/projects2/aisd/2024/shekchu/projects/celltype_eval_llm
pip install -e .
```

## Usage

### Basic Usage

```bash
# Run evaluation on all tissues
python run_evaluation.py

# Run evaluation on specific tissue(s)
python run_evaluation.py --tissue brain
python run_evaluation.py --tissue brain breast colorectal
```

### Advanced Options

```bash
# Skip LLM calls (use existing mappings only)
python run_evaluation.py --tissue brain --skip-llm

# Disable plot generation (faster)
python run_evaluation.py --no-plots

# Custom output directory
python run_evaluation.py --output-dir custom_results

# Verbose logging
python run_evaluation.py --tissue brain --verbose
```

### Full Options

```
usage: run_evaluation.py [-h] [--tissue TISSUE [TISSUE ...]]
                          [--data-root DATA_ROOT] [--output-root OUTPUT_ROOT]
                          [--output-dir OUTPUT_DIR] [--pred-column PRED_COLUMN]
                          [--gt-column GT_COLUMN] [--skip-llm] [--no-plots]
                          [--verbose]

Options:
  --tissue TISSUE          Tissue name(s) to evaluate (e.g., brain, breast)
  --data-root DATA_ROOT    Root directory containing tissue data
  --output-root OUTPUT_ROOT Root directory containing agent outputs
  --output-dir OUTPUT_DIR  Directory to save results (default: evaluation_results)
  --pred-column PRED_COLUMN Column name for predictions (default: cell_type)
  --gt-column GT_COLUMN    Column name for ground truth (default: cell_type)
  --skip-llm              Skip LLM calls, use existing mappings only
  --no-plots              Disable plot generation
  --verbose               Enable verbose logging
```

## File Structure

### Input Files

The pipeline expects the following structure:

```
data/
├── brain/
│   └── h5ad/
│       ├── Data_Choudhury2022_Brain.h5ad
│       ├── Data_Filbin2018_Brain.h5ad
│       └── ...
├── breast/
│   └── h5ad/
│       └── ...
└── other_tissues/

cell_agents/agents/biomaster/output/
├── brain_choudhury2022_cell_type_annotation/
│   ├── annotated_*.h5ad
│   └── ...
├── brain_filbin2018_cell_type_annotation/
└── ...
```

### Output Files

Results are saved in the following structure:

```
evaluation_results/
├── batch_summary.json          # Overall batch summary
├── batch_summary.csv           # Summary table
├── brain_Choudhury2022/
│   ├── overall_metrics.csv
│   ├── per_class_metrics.csv
│   ├── confusion_matrix.csv
│   ├── evaluation_metrics.json
│   ├── per_class_metrics.png
│   └── confusion_matrix.png
├── brain_Filbin2018/
│   └── ...
└── ...
```

## File Matching Logic

The pipeline uses the following logic to match ground truth and prediction files:

1. **Ground Truth Files**: Scans `data/<tissue>/h5ad/*.h5ad`
2. **Dataset Identifier**: Extracts from filename (e.g., `Data_Choudhury2022_Brain` → `Choudhury2022`)
3. **Prediction Files**: Finds matching subdirectory in `output/` by:
   - Looking for `<tissue>_<dataset_id>` pattern (case-insensitive)
   - Selecting the most recently modified `.h5ad` file
   - Preferring files with "annotated" in the name

### Edge Cases

- **No match found**: Logs warning and skips the dataset
- **Multiple matches**: Selects best match by string similarity and logs warning
- **Invalid filename pattern**: Skips with warning

## Output Metrics

For each dataset, the pipeline computes:

- **Overall Metrics**:
  - Accuracy
  - Macro F1 Score
  - Weighted F1 Score

- **Per-Class Metrics**:
  - Precision
  - Recall
  - F1 Score
  - Support

- **Confusion Matrix**: Full confusion matrix for all L3 labels

## Example Output

```
==============================================================
BATCH EVALUATION SUMMARY
==============================================================
Timestamp: 2024-02-27T10:30:45.123456
Total datasets: 15
Successful: 14
Failed: 1

--------------------------------------------------------------
SUCCESSFUL EVALUATIONS:
--------------------------------------------------------------

brain/Choudhury2022:
  Accuracy:    0.8745
  Macro F1:    0.8231
  Weighted F1: 0.8698

brain/Filbin2018:
  Accuracy:    0.9123
  Macro F1:    0.8876
  Weighted F1: 0.9087

...

--------------------------------------------------------------
FAILED EVALUATIONS:
--------------------------------------------------------------
  - brain/Yuan2018

==============================================================
Results saved to: evaluation_results/
==============================================================
```

## Architecture

The pipeline consists of three main modules:

### 1. `file_matcher.py`
- Scans data directories for ground truth files
- Matches prediction files using fuzzy matching
- Handles edge cases (no match, multiple matches)

### 2. `evaluator.py`
- Runs evaluations using the `celltype_standardizer` package
- Saves results in structured format
- Generates visualization plots

### 3. `run_evaluation.py`
- CLI interface
- Argument parsing and validation
- Orchestrates the evaluation workflow

## Troubleshooting

### Issue: "No matched dataset pairs found"
- Check that ground truth files exist in `data/<tissue>/h5ad/`
- Verify prediction files exist in `cell_agents/agents/biomaster/output/`
- Run with `--verbose` to see detailed matching logs

### Issue: "Invalid tissue names"
- Run without `--tissue` to see available tissues
- Check directory structure under `data/`

### Issue: Evaluation fails for specific dataset
- Check the logs for specific error messages
- Verify the .h5ad file is not corrupted
- Check that required columns exist in the file

## Notes

- The pipeline reuses existing L3 mappings when available
- LLM calls are only made for new, unmapped labels
- Use `--skip-llm` for faster runs when all labels are already mapped
- Pre-standardized ground truth files (in `h5ad_l3/`) are used automatically if available
