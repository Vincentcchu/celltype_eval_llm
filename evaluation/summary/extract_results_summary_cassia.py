import json
import csv
import pandas as pd
from pathlib import Path

# Define paths
eval_results_dir = Path('/cs/student/projects2/aisd/2024/shekchu/projects/celltype_eval_llm/evaluation/evaluation_results_cassia')
cassia_output_dir = Path('/cs/student/projects2/aisd/2024/shekchu/projects/agent_outputs/clustered/cassia')
output_csv = Path('/cs/student/projects2/aisd/2024/shekchu/projects/results_summary_cassia.csv')


def parse_folder_name(folder_name):
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


def find_token_tracking(tissue, dataset, cassia_dir):
    """Find dataset-level token tracking JSON for a given tissue and dataset."""
    dataset_dir = cassia_dir / tissue / f"Data_{dataset}"
    token_file = dataset_dir / f"Data_{dataset}_token_tracking.json"

    if token_file.exists():
        return token_file

    # Fallback: look for any token tracking JSON under the expected dataset folder
    if dataset_dir.exists():
        matches = sorted(dataset_dir.glob('*_token_tracking.json'))
        if matches:
            return matches[0]

    return None


def read_metrics(metrics_file):
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


def read_token_tracking(token_file):
    """Read token usage, cost, and timing from CASSIA token tracking JSON."""
    try:
        with open(token_file, 'r') as f:
            data = json.load(f)

        timing = data.get('timing', {})
        tokens = data.get('tokens', {})
        cost = data.get('cost', {})

        duration_seconds = timing.get('total_seconds', None)

        return {
            'total_tokens': tokens.get('total_tokens', None),
            'cost': cost.get('total_cost_usd', None),
            'duration_seconds': duration_seconds,
            'duration_formatted': format_duration(duration_seconds) if duration_seconds is not None else None,
        }
    except Exception as e:
        print(f"Error reading {token_file}: {e}")
        return None


# Main processing
results = []

# Get all subdirectories in evaluation_results_cassia
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

        token_file = find_token_tracking(tissue, dataset, cassia_output_dir)
        if token_file:
            run_data = read_token_tracking(token_file)
            if run_data:
                result['Cost'] = run_data['cost']
                result['Tokens'] = run_data['total_tokens']
                result['Time (seconds)'] = run_data['duration_seconds']
                result['Time (formatted)'] = run_data['duration_formatted']
            else:
                print(f"Warning: Could not read token tracking for {folder_name}")
        else:
            print(f"Warning: token tracking not found for {folder_name}")
    else:
        result['Success'] = 'N'

    results.append(result)


# Write to CSV
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