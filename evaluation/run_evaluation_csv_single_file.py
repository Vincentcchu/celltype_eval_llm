#!/usr/bin/env python3
"""
Batch evaluation CLI for cell type predictions.

Usage:
    # Run evaluation on all tissues
    python run_evaluation.py

    # Run evaluation on one or more specific tissues
    python run_evaluation.py --tissue brain
    python run_evaluation.py --tissue brain breast

    # Skip LLM calls (use existing mappings only)
    python run_evaluation.py --tissue brain --skip-llm

    # Disable plot generation
    python run_evaluation.py --no-plots
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

# Load API key from config.json if not already set correctly
# This ensures we use the correct key from config, overriding any incorrect env vars
try:
    script_dir = Path(__file__).parent
    config_path = script_dir.parent / 'config' / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        api_key = config.get('openai', {}).get('api_key')
        if api_key and api_key != 'YOUR_OPENAI_API_KEY_HERE':
            os.environ['OPENAI_API_KEY'] = api_key
except Exception:
    pass  # Will fall back to existing env var or llm_judge's config loading

# Add parent directory to path to import celltype_standardizer
sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_evaluation import FileMatcher, BatchEvaluator, DatasetPair


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Reduce noise from matplotlib
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch evaluation pipeline for cell type predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                  # Evaluate all tissues
  %(prog)s --tissue brain                   # Evaluate brain only
  %(prog)s --tissue brain breast            # Evaluate multiple tissues
  %(prog)s --single-pred-csv pred.csv --single-gt-h5ad gt.h5ad
  %(prog)s --single-pred-csv pred.csv --single-gt-csv gt.csv
  %(prog)s --single-pred-csv pred.csv       # Use gt column from same CSV
  %(prog)s --skip-llm                       # Skip LLM calls
  %(prog)s --no-plots                       # Disable plot generation
  %(prog)s --output-dir custom_results      # Custom output directory
        """
    )

    parser.add_argument(
        '--single-pred-csv',
        type=Path,
        default=None,
        help='Evaluate one prediction CSV file instead of batch tissue matching.'
    )

    parser.add_argument(
        '--single-gt-h5ad',
        type=Path,
        default=None,
        help='Ground truth .h5ad for single-file mode (optional).'
    )

    parser.add_argument(
        '--single-gt-csv',
        type=Path,
        default=None,
        help='Ground truth CSV for single-file mode (optional).'
    )

    parser.add_argument(
        '--single-pred-id-column',
        type=str,
        default='cell_barcode',
        help='Cell ID column in prediction CSV for matching with GT (default: cell_barcode).'
    )

    parser.add_argument(
        '--single-gt-id-column',
        type=str,
        default='cell_barcode',
        help='Cell ID column in GT CSV for matching with predictions (default: cell_barcode).'
    )

    parser.add_argument(
        '--single-name',
        type=str,
        default=None,
        help='Optional name for single-file run output folder (default: prediction CSV stem).'
    )

    parser.add_argument(
        '--single-match-by-row-order',
        action='store_true',
        help='Match prediction and GT by observation row order instead of ID columns.'
    )
    
    parser.add_argument(
        '--tissue',
        nargs='+',
        metavar='TISSUE',
        help='Tissue name(s) to evaluate (e.g., brain, breast). If not provided, evaluates all tissues.'
    )
    
    parser.add_argument(
        '--data-root',
        type=Path,
        default=None,
        help='Root directory containing tissue data (default: auto-detect)'
    )
    
    parser.add_argument(
        '--output-root',
        type=Path,
        default=None,
        help='Root directory containing agent outputs (default: auto-detect)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('evaluation_results'),
        help='Directory to save evaluation results (default: evaluation_results)'
    )
    
    parser.add_argument(
        '--pred-column',
        type=str,
        default='openrouter_cell_type',
        help='Column name for predictions in prediction files (default: openrouter_cell_type)'
    )
    
    parser.add_argument(
        '--gt-column',
        type=str,
        default='cell_type',
        help='Column name for ground truth in GT files (default: cell_type)'
    )
    
    parser.add_argument(
        '--skip-llm',
        action='store_true',
        help='Skip LLM calls and only use existing mappings'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def auto_detect_paths() -> tuple[Path, Path]:
    """
    Auto-detect data root and output root paths.
    
    Returns:
        Tuple of (data_root, output_root)
    """
    # Assume script is in celltype_eval_llm/evaluation/
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    data_root = project_root / 'data'
    output_root = project_root / 'cell_agents' / 'agents' / 'mLLMCellType' / 'output'
    
    return data_root, output_root


def _sanitize_run_name(value: str) -> str:
    """Create a filesystem-safe run name."""
    safe = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in value.strip())
    return safe or 'single_run'


def _load_csv_as_adata(
    csv_path: Path,
    label_column: str,
    id_column: str,
    role: str,
    logger: logging.Logger,
    fallback_label_columns: list[str] | None = None,
    use_row_index: bool = False,
) -> ad.AnnData:
    """Load a CSV into AnnData.obs using one label column and one ID column."""
    if not csv_path.exists():
        raise ValueError(f"{role} CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)

    selected_label_column = label_column
    if selected_label_column not in df.columns and fallback_label_columns:
        for candidate in fallback_label_columns:
            if candidate in df.columns:
                logger.warning(
                    f"{role} label column '{label_column}' not found in {csv_path}. "
                    f"Using '{candidate}' instead."
                )
                selected_label_column = candidate
                break

    if selected_label_column not in df.columns:
        raise ValueError(
            f"{role} label column '{label_column}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    obs = df[[selected_label_column]].copy()
    if selected_label_column != label_column:
        obs = obs.rename(columns={selected_label_column: label_column})

    if use_row_index:
        obs.index = pd.Index([str(i) for i in range(len(obs))], name='obs_row')
    else:
        if id_column not in df.columns:
            raise ValueError(
                f"{role} ID column '{id_column}' not found in {csv_path}. "
                f"Available columns: {list(df.columns)}"
            )
        obs.index = df[id_column].astype(str).values
        obs.index.name = id_column

    if obs.index.has_duplicates:
        duplicate_count = int(obs.index.duplicated().sum())
        logger.warning(
            f"{role} CSV has {duplicate_count} duplicate IDs in '{id_column}'. "
            "Keeping first occurrence for each ID."
        )
        obs = obs[~obs.index.duplicated(keep='first')].copy()

    # Metrics workflow only needs obs columns and matching obs_names.
    return ad.AnnData(X=np.empty((len(obs), 0), dtype=np.float32), obs=obs)


def run_single_csv_evaluation(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Run evaluation for one prediction CSV against GT from H5AD, CSV, or same CSV."""
    if args.single_gt_h5ad is not None and args.single_gt_csv is not None:
        logger.error("Use only one of --single-gt-h5ad or --single-gt-csv")
        return 1

    if not args.single_pred_csv.exists():
        logger.error(f"Prediction CSV does not exist: {args.single_pred_csv}")
        return 1

    evaluator = BatchEvaluator(
        output_dir=args.output_dir,
        pred_column=args.pred_column,
        gt_column=args.gt_column,
        skip_llm=args.skip_llm,
        save_plots=not args.no_plots
    )

    run_name_raw = args.single_name or args.single_pred_csv.stem
    run_name = _sanitize_run_name(run_name_raw)
    results_dir = args.output_dir / f"single_{run_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 60)
    logger.info("SINGLE FILE EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Prediction CSV: {args.single_pred_csv}")

    pred_adata = _load_csv_as_adata(
        csv_path=args.single_pred_csv,
        label_column=args.pred_column,
        id_column=args.single_pred_id_column,
        role='Prediction',
        logger=logger,
        fallback_label_columns=['cell_type_annotation', 'predicted_cell_type', 'cell_type'],
        use_row_index=args.single_match_by_row_order,
    )

    if args.single_gt_h5ad is not None:
        if not args.single_gt_h5ad.exists():
            logger.error(f"GT H5AD does not exist: {args.single_gt_h5ad}")
            return 1
        if args.single_match_by_row_order:
            gt_source = ad.read_h5ad(args.single_gt_h5ad)
            gt_source.obs_names = pd.Index([str(i) for i in range(len(gt_source))], name='obs_row')
        else:
            gt_source = args.single_gt_h5ad
        gt_desc = str(args.single_gt_h5ad)
    elif args.single_gt_csv is not None:
        gt_source = _load_csv_as_adata(
            csv_path=args.single_gt_csv,
            label_column=args.gt_column,
            id_column=args.single_gt_id_column,
            role='Ground truth',
            logger=logger,
            use_row_index=args.single_match_by_row_order,
        )
        gt_desc = str(args.single_gt_csv)
    else:
        # Evaluate using ground-truth column present in the prediction CSV itself.
        gt_source = None
        gt_desc = f"same prediction CSV column '{args.gt_column}'"

    logger.info(f"Ground truth source: {gt_desc}")
    logger.info(f"Prediction label column: {args.pred_column}")
    logger.info(f"Ground-truth label column: {args.gt_column}")
    logger.info(
        f"Matching mode: {'row-order' if args.single_match_by_row_order else 'ID-column'}"
    )

    try:
        metrics_json_path = results_dir / 'evaluation_metrics.json'
        metrics = evaluator.evaluate_h5ad(
            pred_h5ad=pred_adata,
            pred_column=args.pred_column,
            gt_h5ad=gt_source,
            gt_column=args.gt_column,
            metrics_output_path=metrics_json_path,
            skip_llm=args.skip_llm,
        )

        pair = DatasetPair(
            tissue='single_file',
            dataset_name=run_name,
            gt_file=(args.single_gt_h5ad or args.single_gt_csv or args.single_pred_csv),
            pred_file=args.single_pred_csv,
            match_score=1.0,
        )
        evaluator.save_results(metrics, pair, results_dir)

        logger.info("\n" + "=" * 60)
        logger.info("SINGLE FILE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
        logger.info(f"Macro F1:    {metrics['macro_f1']:.4f}")
        logger.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        logger.info(f"Results saved to: {results_dir}")

        return 0
    except Exception as e:
        logger.error(f"Single file evaluation failed: {e}")
        logger.exception(e)
        return 1


def main() -> int:
    """Main entry point for the CLI."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("CELL TYPE EVALUATION PIPELINE")
    logger.info("="*60)
    
    try:
        if args.single_pred_csv is not None:
            return run_single_csv_evaluation(args, logger)

        # Auto-detect paths if not provided
        if args.data_root is None or args.output_root is None:
            logger.info("Auto-detecting paths...")
            data_root, output_root = auto_detect_paths()
            if args.data_root is None:
                args.data_root = data_root
            if args.output_root is None:
                args.output_root = output_root
        
        logger.info(f"Data root: {args.data_root}")
        logger.info(f"Output root: {args.output_root}")
        logger.info(f"Results will be saved to: {args.output_dir}")
        
        # Validate paths
        if not args.data_root.exists():
            logger.error(f"Data root does not exist: {args.data_root}")
            return 1
        
        if not args.output_root.exists():
            logger.error(f"Output root does not exist: {args.output_root}")
            return 1
        
        # Initialize file matcher
        logger.info("\n" + "="*60)
        logger.info("STEP 1: MATCHING FILES")
        logger.info("="*60)
        
        matcher = FileMatcher(
            data_root=args.data_root,
            output_root=args.output_root
        )
        
        # Get available tissues for validation
        available_tissues = matcher.get_available_tissues()
        logger.info(f"Available tissues: {', '.join(available_tissues)}")
        
        # Validate requested tissues
        if args.tissue:
            invalid_tissues = [t for t in args.tissue if t not in available_tissues]
            if invalid_tissues:
                logger.error(f"Invalid tissue names: {', '.join(invalid_tissues)}")
                logger.error(f"Available tissues: {', '.join(available_tissues)}")
                return 1
            
            logger.info(f"Processing tissues: {', '.join(args.tissue)}")
            tissues_to_process = args.tissue
        else:
            logger.info(f"Processing all tissues: {', '.join(available_tissues)}")
            tissues_to_process = None
        
        # Match datasets
        matched_pairs = matcher.match_datasets(tissues=tissues_to_process)
        
        if not matched_pairs:
            logger.error("No matched dataset pairs found. Exiting.")
            return 1
        
        logger.info(f"\nFound {len(matched_pairs)} matched dataset pairs")
        
        # Initialize batch evaluator
        logger.info("\n" + "="*60)
        logger.info("STEP 2: EVALUATING DATASETS")
        logger.info("="*60)
        
        evaluator = BatchEvaluator(
            output_dir=args.output_dir,
            pred_column=args.pred_column,
            gt_column=args.gt_column,
            skip_llm=args.skip_llm,
            save_plots=not args.no_plots
        )
        
        # Run batch evaluation
        results = evaluator.evaluate_batch(matched_pairs)
        
        # Print summary
        evaluator.print_summary(results)
        
        # Return exit code based on results
        if results['failed'] > 0:
            logger.warning(f"{results['failed']} evaluations failed")
            return 2  # Partial success
        
        return 0  # Success
        
    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.exception(e)
        return 1


if __name__ == '__main__':
    sys.exit(main())


# # Match and evaluate all CSV pairs in directories
# python run_evaluation_csv_batch.py --pred-dir predictions/ --gt-dir ground_truth/

# # Explicit pairs
# python run_evaluation_csv_batch.py --pairs pred1.csv gt1.csv pred2.csv gt2.csv

# # Use glob pattern
# python run_evaluation_csv_batch.py --pred-pattern "results/*.csv" --gt-dir ground_truth/

# # With options
# python run_evaluation_csv_batch.py --pred-dir preds/ --gt-dir truth/ --skip-llm --no-plots