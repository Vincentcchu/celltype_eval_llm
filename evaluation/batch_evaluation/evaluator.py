"""
Batch evaluation runner for cell type predictions.

This module handles:
1. Running evaluations on matched dataset pairs
2. Saving results in a structured format
3. Generating summary reports
"""

import logging
import re
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad

from .file_matcher import DatasetPair

logger = logging.getLogger(__name__)


class BatchEvaluator:
    """Handles batch evaluation of matched dataset pairs."""
    
    def __init__(
        self,
        output_dir: Path,
        pred_column: str = "cell_type",
        gt_column: str = "cell_type",
        skip_llm: bool = False,
        save_plots: bool = True
    ):
        """
        Initialize the batch evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
            pred_column: Column name for predictions in prediction files
            gt_column: Column name for ground truth in GT files
            skip_llm: Whether to skip LLM calls for standardization
            save_plots: Whether to save visualization plots
        """
        self.output_dir = Path(output_dir)
        self.pred_column = pred_column
        self.gt_column = gt_column
        self.skip_llm = skip_llm
        self.save_plots = save_plots
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import evaluation functions
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from celltype_standardizer import evaluate_h5ad
            self.evaluate_h5ad = evaluate_h5ad
        except ImportError as e:
            logger.error(f"Failed to import celltype_standardizer: {e}")
            raise
    
    def evaluate_single_pair(
        self,
        pair: DatasetPair,
        results_dir: Path
    ) -> Optional[Dict]:
        """
        Evaluate a single dataset pair.
        
        Args:
            pair: DatasetPair containing GT and prediction files
            results_dir: Directory to save results for this pair
        
        Returns:
            Dictionary containing evaluation metrics or None if evaluation fails
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {pair.tissue}/{pair.dataset_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Ground truth: {pair.gt_file.name}")
        logger.info(f"Prediction:   {pair.pred_file.name}")
        
        try:
            # Always use original ground truth file (standardize fresh each time)
            logger.info(f"Using ground truth: {pair.gt_file.name}")
            gt_file_for_eval = pair.gt_file
            gt_column = self.gt_column
            
            # Prepare metrics output path
            metrics_json_path = results_dir / 'evaluation_metrics.json'
            
            # Run evaluation
            logger.info("Running evaluation...")
            metrics = self.evaluate_h5ad(
                pred_h5ad=pair.pred_file,
                pred_column=self.pred_column,
                gt_h5ad=gt_file_for_eval,
                gt_column=gt_column,
                metrics_output_path=metrics_json_path,
                skip_llm=self.skip_llm,
            )
            
            logger.info("Evaluation complete!")
            logger.info(f"  Accuracy:    {metrics['accuracy']:.4f}")
            logger.info(f"  Macro F1:    {metrics['macro_f1']:.4f}")
            logger.info(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed for {pair.tissue}/{pair.dataset_name}: {e}")
            logger.exception(e)
            return None
    
    def save_results(
        self,
        metrics: Dict,
        pair: DatasetPair,
        results_dir: Path
    ) -> None:
        """
        Save evaluation results in a structured format.
        
        Args:
            metrics: Dictionary containing evaluation metrics
            pair: DatasetPair being evaluated
            results_dir: Directory to save results
        """
        logger.info("Saving results...")
        
        # 1. Save overall metrics to CSV
        overall_metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Macro F1', 'Weighted F1', 'Total Cells', 
                       'Unique Pred Labels (L3)', 'Unique GT Labels (L3)'],
            'Value': [
                metrics['accuracy'],
                metrics['macro_f1'],
                metrics['weighted_f1'],
                metrics['dataset_info']['total_cells'],
                metrics['dataset_info']['standardized_pred_labels'],
                metrics['dataset_info']['standardized_gt_labels']
            ]
        })
        overall_metrics_path = results_dir / 'overall_metrics.csv'
        overall_metrics_df.to_csv(overall_metrics_path, index=False)
        logger.info(f"✓ Saved overall metrics to: {overall_metrics_path.name}")
        
        # 2. Save per-class metrics to CSV
        per_class_df = pd.DataFrame(metrics['per_class_metrics']).T
        per_class_df.index.name = 'Cell_Type_L3'
        per_class_path = results_dir / 'per_class_metrics.csv'
        per_class_df.to_csv(per_class_path)
        logger.info(f"✓ Saved per-class metrics to: {per_class_path.name}")
        
        # 3. Save confusion matrix to CSV
        cm_data = metrics['confusion_matrix']
        cm_df = pd.DataFrame(
            cm_data['matrix'],
            index=cm_data['labels'],
            columns=cm_data['labels']
        )
        cm_df.index.name = 'True_Label'
        cm_df.columns.name = 'Predicted_Label'
        cm_path = results_dir / 'confusion_matrix.csv'
        cm_df.to_csv(cm_path)
        logger.info(f"✓ Saved confusion matrix to: {cm_path.name}")
        
        # 4. Save plots if enabled
        if self.save_plots:
            self._save_visualization_plots(metrics, results_dir)
    
    def _save_visualization_plots(
        self,
        metrics: Dict,
        results_dir: Path
    ) -> None:
        """
        Generate and save visualization plots.
        
        Args:
            metrics: Dictionary containing evaluation metrics
            results_dir: Directory to save plots
        """
        try:
            # Per-class metrics plot
            per_class = pd.DataFrame(metrics['per_class_metrics']).T
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            per_class['precision'].plot(kind='barh', ax=axes[0], color='skyblue')
            axes[0].set_title('Precision by L3 Cell Type')
            axes[0].set_xlabel('Precision')
            axes[0].set_xlim([0, 1])
            
            per_class['recall'].plot(kind='barh', ax=axes[1], color='lightcoral')
            axes[1].set_title('Recall by L3 Cell Type')
            axes[1].set_xlabel('Recall')
            axes[1].set_xlim([0, 1])
            
            per_class['f1_score'].plot(kind='barh', ax=axes[2], color='lightgreen')
            axes[2].set_title('F1 Score by L3 Cell Type')
            axes[2].set_xlabel('F1 Score')
            axes[2].set_xlim([0, 1])
            
            plt.tight_layout()
            plot_path = results_dir / 'per_class_metrics.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✓ Saved per-class metrics plot to: {plot_path.name}")
            
            # Confusion matrix plot
            cm_data = metrics['confusion_matrix']
            cm_matrix = np.array(cm_data['matrix'])
            labels = cm_data['labels']
            
            plt.figure(figsize=(14, 12))
            sns.heatmap(
                cm_matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Count'}
            )
            plt.title('Confusion Matrix (L3 Labels)', fontsize=14, pad=20)
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            plot_path = results_dir / 'confusion_matrix.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✓ Saved confusion matrix plot to: {plot_path.name}")
            
        except Exception as e:
            logger.warning(f"Failed to save visualization plots: {e}")
    
    def evaluate_batch(
        self,
        pairs: List[DatasetPair]
    ) -> Dict:
        """
        Evaluate a batch of dataset pairs.
        
        Args:
            pairs: List of DatasetPair objects to evaluate
        
        Returns:
            Dictionary containing batch summary and results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH EVALUATION: {len(pairs)} datasets")
        logger.info(f"{'='*60}")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_datasets': len(pairs),
            'successful': 0,
            'failed': 0,
            'datasets': []
        }
        
        for i, pair in enumerate(pairs, 1):
            logger.info(f"\n[{i}/{len(pairs)}] Processing {pair.tissue}/{pair.dataset_name}")
            
            # Create results directory for this pair
            results_folder_name = f"{pair.tissue}_{pair.dataset_name}"
            results_dir = self.output_dir / results_folder_name
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Run evaluation
            metrics = self.evaluate_single_pair(pair, results_dir)
            
            if metrics is not None:
                # Save results
                self.save_results(metrics, pair, results_dir)
                
                results['successful'] += 1
                dataset_result = {
                    'tissue': pair.tissue,
                    'dataset': pair.dataset_name,
                    'status': 'success',
                    'accuracy': metrics['accuracy'],
                    'macro_f1': metrics['macro_f1'],
                    'weighted_f1': metrics['weighted_f1'],
                    'results_dir': str(results_dir)
                }
                
                # Add cell matching statistics
                dataset_info = metrics.get('dataset_info', {})
                if dataset_info.get('cells_filtered', False):
                    dataset_result['original_gt_cells'] = dataset_info.get('original_gt_cells', 0)
                    dataset_result['original_pred_cells'] = dataset_info.get('original_pred_cells', 0)
                    dataset_result['matched_cells'] = dataset_info.get('matched_cells', 0)
                    dataset_result['cells_filtered'] = True
                else:
                    dataset_result['cells_filtered'] = False
                
                results['datasets'].append(dataset_result)
            else:
                results['failed'] += 1
                results['datasets'].append({
                    'tissue': pair.tissue,
                    'dataset': pair.dataset_name,
                    'status': 'failed',
                    'results_dir': str(results_dir)
                })
        
        # Save batch summary
        summary_path = self.output_dir / 'batch_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n✓ Saved batch summary to: {summary_path}")
        
        # Generate summary CSV
        self._generate_summary_csv(results)
        
        return results
    
    def _generate_summary_csv(self, results: Dict) -> None:
        """
        Generate a summary CSV of all evaluations.
        
        Args:
            results: Dictionary containing batch results
        """
        # Create summary dataframe
        summary_data = []
        for dataset in results['datasets']:
            if dataset['status'] == 'success':
                row_data = {
                    'Tissue': dataset['tissue'],
                    'Dataset': dataset['dataset'],
                    'Status': dataset['status'],
                    'Accuracy': f"{dataset['accuracy']:.4f}",
                    'Macro F1': f"{dataset['macro_f1']:.4f}",
                    'Weighted F1': f"{dataset['weighted_f1']:.4f}",
                }
                
                # Add cell matching statistics if cells were filtered
                if dataset.get('cells_filtered', False):
                    row_data['Original GT Cells'] = dataset['original_gt_cells']
                    row_data['Original Pred Cells'] = dataset['original_pred_cells']
                    row_data['Matched Cells'] = dataset['matched_cells']
                    coverage_pct = dataset['matched_cells'] / dataset['original_pred_cells'] * 100
                    row_data['Coverage %'] = f"{coverage_pct:.1f}"
                else:
                    row_data['Original GT Cells'] = '-'
                    row_data['Original Pred Cells'] = '-'
                    row_data['Matched Cells'] = '-'
                    row_data['Coverage %'] = '100.0'
                
                row_data['Results Directory'] = Path(dataset['results_dir']).name
                summary_data.append(row_data)
            else:
                summary_data.append({
                    'Tissue': dataset['tissue'],
                    'Dataset': dataset['dataset'],
                    'Status': dataset['status'],
                    'Accuracy': 'N/A',
                    'Macro F1': 'N/A',
                    'Weighted F1': 'N/A',
                    'Original GT Cells': 'N/A',
                    'Original Pred Cells': 'N/A',
                    'Matched Cells': 'N/A',
                    'Coverage %': 'N/A',
                    'Results Directory': Path(dataset['results_dir']).name
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = self.output_dir / 'batch_summary.csv'
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"✓ Saved summary CSV to: {summary_csv_path}")
    
    def print_summary(self, results: Dict) -> None:
        """
        Print a human-readable summary of batch evaluation results.
        
        Args:
            results: Dictionary containing batch results
        """
        print("\n" + "="*60)
        print("BATCH EVALUATION SUMMARY")
        print("="*60)
        print(f"Timestamp: {results['timestamp']}")
        print(f"Total datasets: {results['total_datasets']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        
        if results['successful'] > 0:
            print("\n" + "-"*60)
            print("SUCCESSFUL EVALUATIONS:")
            print("-"*60)
            for dataset in results['datasets']:
                if dataset['status'] == 'success':
                    print(f"\n{dataset['tissue']}/{dataset['dataset']}:")
                    print(f"  Accuracy:    {dataset['accuracy']:.4f}")
                    print(f"  Macro F1:    {dataset['macro_f1']:.4f}")
                    print(f"  Weighted F1: {dataset['weighted_f1']:.4f}")
                    
                    # Display cell matching statistics if cells were filtered
                    if dataset.get('cells_filtered', False):
                        print(f"  Cell Matching:")
                        print(f"    Original GT cells:   {dataset['original_gt_cells']:,}")
                        print(f"    Original Pred cells: {dataset['original_pred_cells']:,}")
                        print(f"    Matched cells:       {dataset['matched_cells']:,}")
                        pct = dataset['matched_cells'] / dataset['original_pred_cells'] * 100
                        print(f"    Evaluation coverage: {pct:.1f}% of predictions")
        
        if results['failed'] > 0:
            print("\n" + "-"*60)
            print("FAILED EVALUATIONS:")
            print("-"*60)
            for dataset in results['datasets']:
                if dataset['status'] == 'failed':
                    print(f"  - {dataset['tissue']}/{dataset['dataset']}")
        
        print("\n" + "="*60)
        print(f"Results saved to: {self.output_dir}")
        print("="*60)
