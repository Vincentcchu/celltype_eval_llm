import os
import json
import csv
import pandas as pd
from pathlib import Path

# Define paths
eval_results_dir = Path('/cs/student/projects2/aisd/2024/shekchu/projects/celltype_eval_llm/evaluation/evaluation_results_mllmcelltype')
mllm_output_dir = Path('/cs/student/projects2/aisd/2024/shekchu/projects/agent_outputs/clustered/mllmcelltype')
output_csv = Path('/cs/student/projects2/aisd/2024/shekchu/projects/results_summary.csv')

# Function to extract dataset name and tissue from folder name
def parse_folder_name(folder_name):
    """
    Extract tissue and dataset from folder names like:
    brain_Choudhury2022 -> tissue: brain, dataset: Choudhury2022
    head_neck_Chen2020 -> tissue: head_neck, dataset: Chen2020
    """
    # Handle special multi-word tissue types
    multi_word_tissues = ['head_neck']
    
    for tissue_prefix in multi_word_tissues:
        if folder_name.startswith(tissue_prefix + '_'):
            tissue = tissue_prefix
            dataset = folder_name[len(tissue_prefix) + 1:]
            return tissue, dataset
    
    # Default: split on first underscore
    parts = folder_name.split('_', 1)
    if len(parts) == 2:
        tissue = parts[0]
        dataset = parts[1]
        return tissue, dataset
    return None, None

# Function to find matching RUN_SUMMARY.json file
def find_run_summary(tissue, dataset, mllm_dir):
    """
    Find the corresponding RUN_SUMMARY.json file for a given tissue and dataset.
    Try multiple patterns to match the naming conventions for mLLMCellType.
    """
    # Convert to lowercase for mLLMCellType naming
    dataset_lower = dataset.lower()
    tissue_lower = tissue.lower()
    
    # Pattern variations for mLLMCellType (uses celltype_annotation, not cell_type_annotation)
    patterns = [
        f"{tissue_lower}_{dataset_lower}_celltype_annotation_RUN_SUMMARY.json",
    ]
    
    # Special cases for tissues with different names in file paths
    if tissue == 'head_neck':
        patterns.extend([
            f"{tissue_lower}_{dataset_lower}head-and-neck_celltype_annotation_RUN_SUMMARY.json",
        ])
    if tissue == 'liver':
        patterns.extend([
            f"{tissue_lower}_{dataset_lower}liver-biliary_celltype_annotation_RUN_SUMMARY.json",
        ])
    if tissue == 'scarcoma':
        patterns.extend([
            f"{tissue_lower}_{dataset_lower}sarcoma_celltype_annotation_RUN_SUMMARY.json",
        ])
    
    for pattern in patterns:
        summary_file = mllm_dir / pattern
        if summary_file.exists():
            return summary_file
    
    return None

# Function to read metrics from overall_metrics.csv
def read_metrics(metrics_file):
    """Read accuracy, macro f1, and weighted f1 from overall_metrics.csv"""
    try:
        df = pd.read_csv(metrics_file)
        metrics = {}
        for _, row in df.iterrows():
            metrics[row['Metric']] = row['Value']
        
        return {
            'accuracy': metrics.get('Accuracy', None),
            'macro_f1': metrics.get('Macro F1', None),
            'weighted_f1': metrics.get('Weighted F1', None)
        }
    except Exception as e:
        print(f"Error reading {metrics_file}: {e}")
        return None

# Function to format duration in seconds to human-readable format
def format_duration(seconds):
    """Convert seconds to HH:MM:SS format"""
    if seconds is None:
        return None
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# Function to read cost and token info from RUN_SUMMARY.json
def read_run_summary(summary_file):
    """Read token usage, cost, and time from RUN_SUMMARY.json"""
    try:
        with open(summary_file, 'r') as f:
            data = json.load(f)
        
        # The JSON format has direct fields, not nested under token_usage
        duration_seconds = data.get('total_duration_seconds', None)
        return {
            'total_tokens': data.get('tokens_used', None),
            'cost': data.get('credits_from_balance_check', None),
            'duration_seconds': duration_seconds,
            'duration_formatted': format_duration(duration_seconds) if duration_seconds else None
        }
    except Exception as e:
        print(f"Error reading {summary_file}: {e}")
        return None

# Main processing
results = []

# Get all subdirectories in evaluation_results
eval_folders = [f for f in eval_results_dir.iterdir() if f.is_dir()]

for folder in sorted(eval_folders):
    folder_name = folder.name
    tissue, dataset = parse_folder_name(folder_name)
    
    if tissue is None:
        continue
    
    # Check if overall_metrics.csv exists
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
        'Time (formatted)': None
    }
    
    if metrics_file.exists():
        result['Success'] = 'Y'
        
        # Read performance metrics
        metrics = read_metrics(metrics_file)
        if metrics:
            result['Accuracy'] = metrics['accuracy']
            result['Macro F1'] = metrics['macro_f1']
            result['Weighted F1'] = metrics['weighted_f1']
        
        # Find and read RUN_SUMMARY
        summary_file = find_run_summary(tissue, dataset, mllm_output_dir)
        if summary_file:
            run_data = read_run_summary(summary_file)
            if run_data:
                result['Cost'] = run_data['cost']
                result['Tokens'] = run_data['total_tokens']
                result['Time (seconds)'] = run_data['duration_seconds']
                result['Time (formatted)'] = run_data['duration_formatted']
            else:
                print(f"Warning: Could not read RUN_SUMMARY for {folder_name}")
        else:
            print(f"Warning: RUN_SUMMARY not found for {folder_name}")
    else:
        # Folder exists but no metrics file
        result['Success'] = 'N'
    
    results.append(result)

# Write to CSV
with open(output_csv, 'w', newline='') as f:
    fieldnames = ['Tissue Type', 'Dataset', 'Success', 'Accuracy', 'Macro F1', 'Weighted F1', 
                  'Cost', 'Tokens', 'Time (seconds)', 'Time (formatted)']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults written to: {output_csv}")
print(f"Total datasets processed: {len(results)}")
print(f"Successful: {sum(1 for r in results if r['Success'] == 'Y')}")
print(f"Failed: {sum(1 for r in results if r['Success'] == 'N')}")
