"""
Workflow 2: Evaluation workflow (standardize → exact match/metrics).

Evaluates model predictions against ground truth after standardization to L3.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict, List
import anndata as ad
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import json

from .mapping_store import MappingStore
from .llm_judge import LLMSemanticJudge

logger = logging.getLogger(__name__)


def evaluate_h5ad(
    pred_h5ad: Union[str, Path, ad.AnnData],
    pred_column: str,
    gt_h5ad: Optional[Union[str, Path, ad.AnnData]] = None,
    gt_column: Optional[str] = None,
    metrics_output_path: Optional[Union[str, Path]] = None,
    mapping_store_path: Optional[str] = None,
    l3_vocab_path: Optional[str] = None,
    api_key: Optional[str] = None,
    llm_model: str = "gpt-4o-mini",
    skip_llm: bool = False,
    standardized_pred_column: str = "pred_l3",
    standardized_gt_column: str = "gt_l3",
) -> Dict:
    """
    Evaluate model predictions against ground truth after L3 standardization.
    
    This function:
    1. Loads prediction and ground truth data
    2. Standardizes both predicted and ground truth labels to L3
    3. Updates mapping store for any new labels encountered
    4. Computes evaluation metrics (accuracy, F1, confusion matrix, per-class metrics)
    5. Optionally saves detailed metrics report
    
    Args:
        pred_h5ad: Path to .h5ad with predictions or AnnData object
        pred_column: Column name in pred adata.obs containing predicted labels
        gt_h5ad: Path to ground truth .h5ad or AnnData object. 
                 If None, assumes GT is in pred_h5ad.
        gt_column: Column name containing ground truth labels.
                   If None and gt_h5ad is None, raises error.
        metrics_output_path: Optional path to save metrics JSON report
        mapping_store_path: Path to mapping store (uses default if None)
        l3_vocab_path: Path to L3 vocabulary (uses default if None)
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
        llm_model: OpenAI model to use (default: "gpt-4o-mini")
        skip_llm: If True, only use existing mappings without calling LLM
        standardized_pred_column: Column name for standardized predictions
        standardized_gt_column: Column name for standardized ground truth
        
    Returns:
        Dictionary containing evaluation metrics:
            - accuracy: Overall accuracy
            - macro_f1: Macro-averaged F1 score
            - weighted_f1: Weighted-averaged F1 score
            - confusion_matrix: Confusion matrix as nested list
            - per_class_metrics: Per-class precision, recall, F1
            - label_counts: Counts of each label in predictions and ground truth
            
    Raises:
        ValueError: If required columns don't exist or data is invalid
    """
    logger.info("=== Starting evaluation workflow ===")
    
    # Load prediction data
    if isinstance(pred_h5ad, (str, Path)):
        logger.info(f"Loading prediction data from {pred_h5ad}")
        pred_adata = ad.read_h5ad(pred_h5ad)
    else:
        pred_adata = pred_h5ad.copy()
    
    # Validate prediction column
    if pred_column not in pred_adata.obs.columns:
        raise ValueError(
            f"Prediction column '{pred_column}' not found in prediction data. "
            f"Available columns: {list(pred_adata.obs.columns)}"
        )
    
    # Handle ground truth data
    if gt_h5ad is None:
        # Ground truth is in the same AnnData
        if gt_column is None:
            raise ValueError(
                "When gt_h5ad is None, gt_column must be specified to locate "
                "ground truth labels in the prediction AnnData."
            )
        gt_adata = pred_adata
        logger.info("Using ground truth from prediction AnnData")
    else:
        # Load separate ground truth data
        if isinstance(gt_h5ad, (str, Path)):
            logger.info(f"Loading ground truth data from {gt_h5ad}")
            gt_adata = ad.read_h5ad(gt_h5ad)
        else:
            gt_adata = gt_h5ad.copy()
    
    # Validate ground truth column
    if gt_column not in gt_adata.obs.columns:
        raise ValueError(
            f"Ground truth column '{gt_column}' not found. "
            f"Available columns: {list(gt_adata.obs.columns)}"
        )
    
    # Validate matching cell counts
    if len(pred_adata) != len(gt_adata):
        raise ValueError(
            f"Mismatch in cell counts: predictions={len(pred_adata)}, "
            f"ground_truth={len(gt_adata)}. Cannot evaluate."
        )
    
    # Initialize components
    mapping_store = MappingStore(mapping_store_path)
    logger.info(f"Initial mapping store stats: {mapping_store.get_stats()}")
    
    # Collect all unique labels that need standardization
    pred_labels = set(pred_adata.obs[pred_column].astype(str).unique())
    gt_labels = set(gt_adata.obs[gt_column].astype(str).unique())
    all_labels = pred_labels | gt_labels
    
    logger.info(f"Found {len(pred_labels)} unique predicted labels")
    logger.info(f"Found {len(gt_labels)} unique ground truth labels")
    logger.info(f"Total {len(all_labels)} unique labels to standardize")
    
    # Check for unmapped labels
    unmapped_labels = mapping_store.get_unmapped_labels(all_labels)
    
    if unmapped_labels and not skip_llm:
        logger.info(f"Mapping {len(unmapped_labels)} new labels using LLM")
        judge = LLMSemanticJudge(
            api_key=api_key,
            model=llm_model,
            vocab_file=l3_vocab_path
        )
        
        new_mappings = {}
        for raw_label in sorted(unmapped_labels):
            result = judge.map_label(raw_label)
            new_mappings[raw_label] = result["selected_label"]
            logger.info(
                f"  '{raw_label}' -> '{result['selected_label']}' "
                f"(confidence: {result.get('confidence', 'N/A')})"
            )
        
        mapping_store.add_mappings_batch(new_mappings)
        logger.info(f"Added {len(new_mappings)} new mappings")
    elif unmapped_labels and skip_llm:
        logger.warning(
            f"Skipping LLM for {len(unmapped_labels)} unmapped labels. "
            f"These will be marked as 'UNMAPPED'."
        )
    
    # Get all mappings
    all_mappings = mapping_store.get_all_mappings()
    
    def map_label(raw_label: str) -> str:
        """Map a label with fallback."""
        return all_mappings.get(str(raw_label), "UNMAPPED")
    
    # Standardize predictions
    pred_adata.obs[standardized_pred_column] = (
        pred_adata.obs[pred_column].astype(str).apply(map_label)
    )
    
    # Standardize ground truth
    gt_adata.obs[standardized_gt_column] = (
        gt_adata.obs[gt_column].astype(str).apply(map_label)
    )
    
    # If using same AnnData, sync the columns
    if gt_adata is pred_adata:
        gt_standardized = pred_adata.obs[standardized_gt_column]
    else:
        gt_standardized = gt_adata.obs[standardized_gt_column]
    
    pred_standardized = pred_adata.obs[standardized_pred_column]
    
    # Check for UNMAPPED labels
    unmapped_pred_count = (pred_standardized == "UNMAPPED").sum()
    unmapped_gt_count = (gt_standardized == "UNMAPPED").sum()
    
    if unmapped_pred_count > 0:
        logger.warning(f"{unmapped_pred_count} predictions remain UNMAPPED")
    if unmapped_gt_count > 0:
        logger.warning(f"{unmapped_gt_count} ground truth labels remain UNMAPPED")
    
    # Compute metrics
    logger.info("Computing evaluation metrics...")
    
    # Convert to numpy arrays for sklearn
    y_true = gt_standardized.values
    y_pred = pred_standardized.values
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # F1 scores
    labels_present = sorted(set(y_true) | set(y_pred))
    macro_f1 = f1_score(y_true, y_pred, labels=labels_present, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels_present, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    
    # Per-class metrics
    report = classification_report(
        y_true, y_pred, 
        labels=labels_present,
        output_dict=True,
        zero_division=0
    )
    
    # Label counts
    pred_counts = pred_standardized.value_counts().to_dict()
    gt_counts = gt_standardized.value_counts().to_dict()
    
    # Prepare results
    results = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": {
            "labels": labels_present,
            "matrix": cm.tolist()
        },
        "per_class_metrics": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1_score": report[label]["f1-score"],
                "support": int(report[label]["support"])
            }
            for label in labels_present
            if label in report
        },
        "label_counts": {
            "predictions": pred_counts,
            "ground_truth": gt_counts
        },
        "dataset_info": {
            "total_cells": len(pred_adata),
            "unique_pred_labels": len(pred_labels),
            "unique_gt_labels": len(gt_labels),
            "standardized_pred_labels": len(set(pred_standardized)),
            "standardized_gt_labels": len(set(gt_standardized)),
            "unmapped_predictions": int(unmapped_pred_count),
            "unmapped_ground_truth": int(unmapped_gt_count)
        },
        "mapping_store_stats": mapping_store.get_stats()
    }
    
    # Log summary
    logger.info("=== Evaluation Results ===")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Macro F1: {macro_f1:.4f}")
    logger.info(f"Weighted F1: {weighted_f1:.4f}")
    logger.info(f"Total cells evaluated: {len(pred_adata)}")
    
    # Save metrics if requested
    if metrics_output_path:
        output_path = Path(metrics_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved metrics report to {output_path}")
    
    logger.info("=== Evaluation workflow complete ===")
    
    return results


def evaluate_with_separate_files(
    pred_h5ad_path: Union[str, Path],
    gt_h5ad_path: Union[str, Path],
    pred_column: str,
    gt_column: str,
    **kwargs
) -> Dict:
    """
    Convenience function for evaluating with separate prediction and GT files.
    
    Args:
        pred_h5ad_path: Path to predictions .h5ad
        gt_h5ad_path: Path to ground truth .h5ad
        pred_column: Column name for predictions
        gt_column: Column name for ground truth
        **kwargs: Additional arguments passed to evaluate_h5ad
        
    Returns:
        Dictionary with evaluation metrics
    """
    return evaluate_h5ad(
        pred_h5ad=pred_h5ad_path,
        pred_column=pred_column,
        gt_h5ad=gt_h5ad_path,
        gt_column=gt_column,
        **kwargs
    )
