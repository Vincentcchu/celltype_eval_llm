import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# Define paths
project_root = Path('/cs/student/projects2/aisd/2024/shekchu/projects')
eval_results_dir = project_root / 'celltype_eval_llm' / 'evaluation' / 'evaluation_results_biomni'
biomni_output_dir = project_root / 'agent_outputs' / 'clustered' / 'biomni'
output_csv = project_root / 'celltype_eval_llm' / 'evaluation' / 'summary' / 'results_summary_biomni.csv'


def parse_folder_name(folder_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract tissue and dataset from folder names like:
    biomni_brain_Choudhury2022 -> tissue: brain, dataset: Choudhury2022
    biomni_head_neck_Chen2020 -> tissue: head_neck, dataset: Chen2020
    """
    if folder_name.startswith('biomni_'):
        folder_name = folder_name[len('biomni_'):]

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


def get_tissue_aliases(tissue: str) -> List[str]:
    """Return possible tissue aliases used in directory names."""
    aliases = [tissue]
    lower = tissue.lower()

    if lower == 'sarcoma':
        aliases.append('scarcoma')
    if lower == 'scarcoma':
        aliases.append('sarcoma')

    return aliases


def find_biomni_output(tissue: str, dataset: str, output_dir: Path) -> Optional[Path]:
    """
    Find the corresponding Biomni biomni_output.json file.

    Expected run dir format:
    {tissue}_Data_{dataset}_*/biomni_output.json
    """
    for tissue_alias in get_tissue_aliases(tissue):
        pattern = f"{tissue_alias}_Data_{dataset}_*/biomni_output.json"
        matches = sorted(output_dir.glob(pattern))
        if matches:
            return matches[0]

    return None


def read_metrics(metrics_file: Path) -> Optional[Dict[str, float]]:
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
    except Exception as exc:
        print(f"Error reading {metrics_file}: {exc}")
        return None


def format_duration(seconds: Optional[float]) -> Optional[str]:
    """Convert seconds to HH:MM:SS format."""
    if seconds is None:
        return None

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def read_biomni_output(summary_file: Path) -> Optional[Dict[str, Optional[float]]]:
    """Read token usage, cost, and timing from Biomni biomni_output.json."""
    try:
        with open(summary_file, 'r') as handle:
            data = json.load(handle)

        metrics = data.get('metrics', {}) if isinstance(data, dict) else {}

        # biomni_output.json nests run accounting under metrics.
        token_usage = metrics.get('token_usage', {})
        total_usage = token_usage.get('total', {}) if isinstance(token_usage, dict) else {}
        cost_usd = metrics.get('cost_usd', {}) if isinstance(metrics, dict) else {}

        timing_seconds = metrics.get('timing_seconds', {}) if isinstance(metrics, dict) else {}
        duration_seconds = timing_seconds.get('total', None) if isinstance(timing_seconds, dict) else None

        return {
            'total_tokens': total_usage.get('total_tokens', None),
            'cost': cost_usd.get('total', None),
            'duration_seconds': duration_seconds,
            'duration_formatted': format_duration(duration_seconds),
        }
    except Exception as exc:
        print(f"Error reading {summary_file}: {exc}")
        return None


def main() -> None:
    """Build the Biomni summary CSV from evaluation and run artifacts."""
    results = []

    if not eval_results_dir.exists():
        raise FileNotFoundError(f"Evaluation results directory not found: {eval_results_dir}")
    if not biomni_output_dir.exists():
        raise FileNotFoundError(f"Biomni output directory not found: {biomni_output_dir}")

    eval_folders = [path for path in eval_results_dir.iterdir() if path.is_dir()]

    for folder in sorted(eval_folders):
        tissue, dataset = parse_folder_name(folder.name)
        if tissue is None or dataset is None:
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

            summary_file = find_biomni_output(tissue, dataset, biomni_output_dir)
            if summary_file:
                run_data = read_biomni_output(summary_file)
                if run_data:
                    result['Cost'] = run_data['cost']
                    result['Tokens'] = run_data['total_tokens']
                    result['Time (seconds)'] = run_data['duration_seconds']
                    result['Time (formatted)'] = run_data['duration_formatted']
                else:
                    print(f"Warning: Could not parse biomni_output.json for {folder.name}")
            else:
                print(f"Warning: biomni_output.json not found for {folder.name}")
        else:
            result['Success'] = 'N'

        results.append(result)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='') as handle:
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to: {output_csv}")
    print(f"Total datasets processed: {len(results)}")
    print(f"Successful: {sum(1 for row in results if row['Success'] == 'Y')}")
    print(f"Failed: {sum(1 for row in results if row['Success'] == 'N')}")


if __name__ == '__main__':
    main()
