import json
import csv
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple


# Define paths
project_root = Path('/cs/student/projects2/aisd/2024/shekchu/projects')
eval_results_dir = project_root / 'celltype_eval_llm' / 'evaluation' / 'evaluation_results_biomaster'
biomaster_output_dir = project_root / 'agent_outputs' / 'clustered' / 'biomaster'
output_csv = project_root / 'celltype_eval_llm' / 'evaluation' / 'summary' / 'results_summary_biomaster.csv'


def parse_folder_name(folder_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract tissue and dataset from folder names like:
    brain_Choudhury2022 -> tissue: brain, dataset: Choudhury2022
    head_neck_Chen2020 -> tissue: head_neck, dataset: Chen2020
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


def find_run_summary(tissue: str, dataset: str, output_dir: Path) -> Optional[Path]:
    """
    Find the corresponding biomaster RUN_SUMMARY.json file.

    Handles common patterns and tissue-specific suffix conventions.
    """
    tissue_lower = tissue.lower()
    dataset_lower = dataset.lower()

    patterns = [
        f"{tissue_lower}_{dataset_lower}_cell_type_annotation_RUN_SUMMARY.json",
    ]

    # Tissue-specific dataset suffix conventions seen in run IDs
    if tissue_lower == 'head_neck':
        patterns.append(
            f"{tissue_lower}_{dataset_lower}head-and-neck_cell_type_annotation_RUN_SUMMARY.json"
        )
    if tissue_lower == 'liver':
        patterns.append(
            f"{tissue_lower}_{dataset_lower}liver-biliary_cell_type_annotation_RUN_SUMMARY.json"
        )
    if tissue_lower == 'scarcoma':
        patterns.append(
            f"{tissue_lower}_{dataset_lower}sarcoma_cell_type_annotation_RUN_SUMMARY.json"
        )

    for pattern in patterns:
        candidate = output_dir / pattern
        if candidate.exists():
            return candidate

    # Fallback: any file that starts with tissue + dataset and ends with RUN_SUMMARY
    wildcard_matches = sorted(output_dir.glob(f"{tissue_lower}_{dataset_lower}*_RUN_SUMMARY.json"))
    if wildcard_matches:
        return wildcard_matches[0]

    return None


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


def read_run_summary(summary_file: Path):
    """Read token usage, cost, and timing from BioMaster RUN_SUMMARY.json."""
    try:
        with open(summary_file, 'r') as f:
            data = json.load(f)

        token_usage = data.get('token_usage', {})
        duration_seconds = data.get('duration_seconds', None)

        # Prefer calculated cost if present, otherwise callback-reported cost.
        cost = token_usage.get('calculated_cost', None)
        if cost is None:
            cost = token_usage.get('callback_reported_cost', None)

        return {
            'total_tokens': token_usage.get('total_tokens', None),
            'cost': cost,
            'duration_seconds': duration_seconds,
            'duration_formatted': format_duration(duration_seconds) if duration_seconds is not None else None,
        }
    except Exception as e:
        print(f"Error reading {summary_file}: {e}")
        return None


# Main processing
results = []

if not eval_results_dir.exists():
    raise FileNotFoundError(f"Evaluation results directory not found: {eval_results_dir}")
if not biomaster_output_dir.exists():
    raise FileNotFoundError(f"BioMaster output directory not found: {biomaster_output_dir}")

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

        summary_file = find_run_summary(tissue, dataset, biomaster_output_dir)
        if summary_file:
            run_data = read_run_summary(summary_file)
            if run_data:
                result['Cost'] = run_data['cost']
                result['Tokens'] = run_data['total_tokens']
                result['Time (seconds)'] = run_data['duration_seconds']
                result['Time (formatted)'] = run_data['duration_formatted']
            else:
                print(f"Warning: Could not parse RUN_SUMMARY for {folder_name}")
        else:
            print(f"Warning: RUN_SUMMARY not found for {folder_name}")
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