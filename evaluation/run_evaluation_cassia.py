#!/usr/bin/env python3
"""
Batch evaluation CLI for CASSIA predictions.

CASSIA outputs cluster-level predictions in CSV format. This script:
1. Matches ground truth h5ad files with CASSIA output
2. Converts cluster-level predictions to cell-level predictions (in-memory)
3. Evaluates using the standard evaluation pipeline

Usage:
    # Run evaluation on all tissues
    python run_evaluation_cassia.py

    # Run evaluation on one or more specific tissues
    python run_evaluation_cassia.py --tissue brain
    python run_evaluation_cassia.py --tissue brain breast

    # Skip LLM calls (use existing mappings only)
    python run_evaluation_cassia.py --tissue brain --skip-llm

    # Disable plot generation
    python run_evaluation_cassia.py --no-plots
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import anndata as ad

# Load API key from config.json if not already set correctly
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
    pass  # Will fall back to existing env var

# Add parent directory to path to import celltype_standardizer
sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_evaluation.file_matcher_cassia import CASSIAFileMatcher, CASSIADatasetPair
from batch_evaluation.cassia_adapter import CASSIAAdapter
from batch_evaluation import BatchEvaluator
from batch_evaluation.file_matcher import DatasetPair


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
        description='Batch evaluation pipeline for CASSIA predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                  # Evaluate all tissues
  %(prog)s --tissue brain                   # Evaluate brain only
  %(prog)s --tissue brain breast            # Evaluate multiple tissues
  %(prog)s --skip-llm                       # Skip LLM calls
  %(prog)s --no-plots                       # Disable plot generation
  %(prog)s --output-dir custom_results      # Custom output directory
        """
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
        help='Root directory containing CASSIA outputs (default: auto-detect)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('evaluation_results_cassia'),
        help='Directory to save evaluation results (default: evaluation_results_cassia)'
    )
    
    parser.add_argument(
        '--pred-column',
        type=str,
        default='predicted_cell_type',
        help='Column name for predictions after conversion (default: predicted_cell_type)'
    )
    
    parser.add_argument(
        '--gt-column',
        type=str,
        default='cell_type',
        help='Column name for ground truth in GT files (default: cell_type)'
    )
    
    parser.add_argument(
        '--cluster-column',
        type=str,
        default='cluster',
        help='Column name for cluster assignments in clustered h5ad (default: cluster)'
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
    Auto-detect data root and CASSIA output root paths.
    
    Returns:
        Tuple of (data_root, output_root)
    """
    # Assume script is in celltype_eval_llm/evaluation/
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    data_root = project_root / 'data'
    # output_root = project_root / 'cell_agents' / 'agents' / 'CASSIA' / 'CASSIA_run' / 'output'
    output_root = project_root / 'agent_outputs' / 'clustered' / 'cassia'
    
    return data_root, output_root


def convert_cassia_to_cell_level(
    pairs: List[CASSIADatasetPair],
    adapter: CASSIAAdapter
) -> List[tuple[CASSIADatasetPair, ad.AnnData]]:
    """
    Convert CASSIA cluster-level predictions to cell-level for all pairs (in-memory).
    
    Args:
        pairs: List of CASSIA dataset pairs
        adapter: CASSIAAdapter instance
    
    Returns:
        List of (pair, adata_object) tuples
    """
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("STEP 2: CONVERTING CLUSTER PREDICTIONS TO CELL LEVEL")
    logger.info("="*60)
    
    converted_pairs = []
    
    for i, pair in enumerate(pairs, 1):
        logger.info(f"\nProcessing {i}/{len(pairs)}: {pair.tissue}/{pair.dataset_name}")
        
        try:
            # Convert cluster predictions to cell level (in-memory to avoid disk quota issues)
            _, adata_obj = adapter.create_cell_level_predictions(
                csv_path=pair.pred_csv,
                h5ad_path=pair.clustered_h5ad,
                return_adata=True  # Always use in-memory mode
            )
            
            converted_pairs.append((pair, adata_obj))
            
        except Exception as e:
            logger.error(f"Failed to convert {pair.tissue}/{pair.dataset_name}: {e}")
            logger.exception(e)
            continue
    
    logger.info(f"\n✓ Successfully converted {len(converted_pairs)}/{len(pairs)} datasets")
    
    return converted_pairs


def main() -> int:
    """Main entry point for the CLI."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("CASSIA EVALUATION PIPELINE")
    logger.info("="*60)
    
    try:
        # Auto-detect paths if not provided
        if args.data_root is None or args.output_root is None:
            logger.info("Auto-detecting paths for CASSIA...")
            data_root, output_root = auto_detect_paths()
            if args.data_root is None:
                args.data_root = data_root
            if args.output_root is None:
                args.output_root = output_root
        
        logger.info(f"Data root: {args.data_root}")
        logger.info(f"CASSIA output root: {args.output_root}")
        logger.info(f"Results will be saved to: {args.output_dir}")
        
        # Validate paths
        if not args.data_root.exists():
            logger.error(f"Data root does not exist: {args.data_root}")
            return 1
        
        if not args.output_root.exists():
            logger.error(f"CASSIA output root does not exist: {args.output_root}")
            return 1
        
        # Initialize file matcher
        logger.info("\n" + "="*60)
        logger.info("STEP 1: MATCHING FILES")
        logger.info("="*60)
        
        matcher = CASSIAFileMatcher(
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
        
        # Initialize adapter
        adapter = CASSIAAdapter(
            cluster_column=args.cluster_column,
            prediction_column="Predicted Main Cell Type"
        )
        
        # Convert cluster predictions to cell level (in-memory)
        converted_pairs = convert_cassia_to_cell_level(
            pairs=matched_pairs,
            adapter=adapter
        )
        
        if not converted_pairs:
            logger.error("No datasets successfully converted. Exiting.")
            return 1
        
        # Initialize batch evaluator
        logger.info("\n" + "="*60)
        logger.info("STEP 3: EVALUATING DATASETS")
        logger.info("="*60)
        
        evaluator = BatchEvaluator(
            output_dir=args.output_dir,
            pred_column=args.pred_column,
            gt_column=args.gt_column,
            skip_llm=args.skip_llm,
            save_plots=not args.no_plots
        )
        
        # Convert to format expected by BatchEvaluator
        # Pass AnnData objects directly (evaluate_h5ad supports this)
        eval_pairs = []
        for cassia_pair, adata_obj in converted_pairs:
            eval_pair = DatasetPair(
                tissue=cassia_pair.tissue,
                dataset_name=cassia_pair.dataset_name,
                gt_file=cassia_pair.gt_file,
                pred_file=adata_obj,  # Pass AnnData object directly
                match_score=cassia_pair.match_score
            )
            eval_pairs.append(eval_pair)
        
        # Run batch evaluation
        results = evaluator.evaluate_batch(eval_pairs)
        
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
