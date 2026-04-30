import json
import csv
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

# Define paths
project_root = Path('/cs/student/projects2/aisd/2024/shekchu/projects')
eval_results_dir = project_root / 'celltype_eval_llm' / 'evaluation' / 'evaluation_results_celltypeagent'
celltypeagent_output_dir = project_root / 'agent_outputs' / 'non_clustered' / 'celltypeagent'
output_csv = project_root / 'celltype_eval_llm' / 'evaluation' / 'summary' / 'results_summary_celltypeagent.csv'


def parse_folder_name(folder_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract tissue and dataset from folder names like:
    brain_Choudhury2022_Brain -> tissue: brain, dataset: Choudhury2022_Brain
    head_neck_Chen2020_Head-and-Neck -> tissue: head_neck, dataset: Chen2020_Head-and-Neck
    """
    multi_word_tissues = ['head_neck']

    for tissue_prefix in multi_word_tissues:
        if folder_name.startswith(tissue_prefix + '_'):
            tissue = tissue_prefix
            dataset = folder_name[len(tissue_prefix) + 1:]
            return tissue, dataset

    parts = folder_name.split('_', 1)
    if len(parts) == 2:
        tissue = parts[0]
        dataset = parts[1]
        return tissue, dataset

    return None, None


def read_metrics(metrics_file: Path):
    """Read accuracy, macro f1, and weighted f1 from overall_metrics.csv."""
    try:
        df = pd.read_csv(metrics_file)
        metrics = {}
        for _, row in df.iterrows():
            metrics[row['Metric']] = row['Value']

        return {
            'accuracy': metrics.get('Accuracy', None),
            'macro_f1': metrics.get('Macro F1', None),
            'weighted_f1': metrics.get('Weighted F1', None),
        }
    except Exception as e:
        print(f"Error reading {metrics_file}: {e}")
        return None


def format_duration(seconds):
    """Convert seconds to HH:MM:SS format."""
    if seconds is None:
        return None

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def find_prediction_csv_run_dir(tissue: str, dataset: str, output_root: Path) -> Optional[Path]:
    """
    Find the latest run directory that contains the dataset prediction CSV.

    Expected structure:
      output_root/<tissue>/Data_<dataset>_formatted/prediction/<model>/<timestamp>/Data_<dataset>_formatted.csv

    Returns:
      Path to the run directory (<timestamp>) containing the CSV, or None.
    """
    dataset_dir = output_root / tissue / f"Data_{dataset}_formatted" / 'prediction'
    if not dataset_dir.exists():
        return None

    csv_name = f"Data_{dataset}_formatted.csv"
    matches = [p for p in dataset_dir.glob(f'**/{csv_name}') if p.is_file()]

    if not matches:
        return None

    latest_csv = max(matches, key=lambda path: path.stat().st_mtime)
    return latest_csv.parent


def read_run_metrics(run_metrics_file: Path):
    """Read token usage, cost, and timing from CellTypeAgent run_metrics.json."""
    try:
        with open(run_metrics_file, 'r') as f:
            data = json.load(f)

        duration_seconds = data.get('total_time_seconds', None)

        return {
            'total_tokens': data.get('total_tokens', None),
            'cost': data.get('total_cost_usd', None),
            'duration_seconds': duration_seconds,
            'duration_formatted': format_duration(duration_seconds) if duration_seconds is not None else None,
        }
    except Exception as e:
        print(f"Error reading {run_metrics_file}: {e}")
        return None


# Main processing
results = []

if not eval_results_dir.exists():
    raise FileNotFoundError(f"Evaluation results directory not found: {eval_results_dir}")
if not celltypeagent_output_dir.exists():
    raise FileNotFoundError(f"CellTypeAgent output directory not found: {celltypeagent_output_dir}")

# Get all subdirectories in evaluation_results_celltypeagent
eval_folders = [f for f in eval_results_dir.iterdir() if f.is_dir()]

for folder in sorted(eval_folders):
    folder_name = folder.name
    tissue, dataset = parse_folder_name(folder_name)

    if tissue is None:
        continue

    metrics_file = folder / 'overall_metrics.csv'

    result = {
        'Tissue Type': tissue,
        'Dataset': dataset,
        'Success': None,
        'Accuracy': None,
        'Macro F1': None,
        'Weighted F1': None,
        'Cost': None,
        'Tokens': None,
        'Time (seconds)': None,
        'Time (formatted)': None,
    }

    if metrics_file.exists():
        result['Success'] = 'Y'

        metrics = read_metrics(metrics_file)
        if metrics:
            result['Accuracy'] = metrics['accuracy']
            result['Macro F1'] = metrics['macro_f1']
            result['Weighted F1'] = metrics['weighted_f1']

        run_dir = find_prediction_csv_run_dir(tissue, dataset, celltypeagent_output_dir)
        if run_dir:
            run_metrics_file = run_dir / 'run_metrics.json'
            if run_metrics_file.exists():
                run_data = read_run_metrics(run_metrics_file)
                if run_data:
                    result['Cost'] = run_data['cost']
                    result['Tokens'] = run_data['total_tokens']
                    result['Time (seconds)'] = run_data['duration_seconds']
                    result['Time (formatted)'] = run_data['duration_formatted']
                else:
                    print(f"Warning: Could not parse run metrics for {folder_name}")
            else:
                print(f"Warning: run_metrics.json not found for {folder_name} in {run_dir}")
        else:
            print(f"Warning: prediction CSV run folder not found for {folder_name}")
    else:
        result['Success'] = 'N'

    results.append(result)


# Write to CSV
output_csv.parent.mkdir(parents=True, exist_ok=True)
with open(output_csv, 'w', newline='') as f:
    fieldnames = [
        'Tissue Type',
        'Dataset',
        'Success',
        'Accuracy',
        'Macro F1',
        'Weighted F1',
        'Cost',
        'Tokens',
        'Time (seconds)',
        'Time (formatted)',
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults written to: {output_csv}")
print(f"Total datasets processed: {len(results)}")
print(f"Successful: {sum(1 for r in results if r['Success'] == 'Y')}")
print(f"Failed: {sum(1 for r in results if r['Success'] == 'N')}")
