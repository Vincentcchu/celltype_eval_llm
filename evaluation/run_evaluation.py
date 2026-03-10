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
import logging
import sys
from pathlib import Path

# Add parent directory to path to import celltype_standardizer
sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_evaluation import FileMatcher, BatchEvaluator


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
