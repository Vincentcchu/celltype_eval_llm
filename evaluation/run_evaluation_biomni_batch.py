#!/usr/bin/env python3
"""
Batch evaluation pipeline for Biomni cell type predictions.

This script automatically discovers CSV files in the Biomni outputs directory
and matches them against ground truth h5ad files, then evaluates all pairs.

Usage:
    # Evaluate all Biomni outputs with auto-detected paths
    python run_evaluation_biomni_batch.py

    # Specify custom Biomni output directory
    python run_evaluation_biomni_batch.py --biomni-dir /path/to/biomni/outputs

    # Specify custom GT and results directories
    python run_evaluation_biomni_batch.py --gt-root /path/to/gt/data --output-dir results

    # Skip LLM standardization
    python run_evaluation_biomni_batch.py --skip-llm

    # Disable plot generation
    python run_evaluation_biomni_batch.py --no-plots

    # Enable verbose logging
    python run_evaluation_biomni_batch.py --verbose
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd

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
    pass

# Add parent directory to path to import celltype_standardizer
sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_evaluation import BatchEvaluator, DatasetPair


# Default paths for this repository layout.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_BIOMNI_DIR = PROJECT_ROOT / 'agent_outputs' / 'clustered' / 'biomni'
DEFAULT_GT_ROOT = PROJECT_ROOT / 'data'
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / 'evaluation_results_biomni'


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch evaluation pipeline for Biomni cell type predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                           # Auto-detect all paths
  %(prog)s --biomni-dir /custom/path/biomni          # Custom Biomni dir
  %(prog)s --gt-root /path/to/data --biomni-dir /path/to/biomni
  %(prog)s --skip-llm                                # Skip LLM standardization
  %(prog)s --no-plots                                # Disable visualizations
  %(prog)s --output-dir custom_results               # Custom output directory
  %(prog)s --pred-column cell_type             # Custom prediction column
        """
    )

    parser.add_argument(
        '--biomni-dir',
        type=Path,
        default=DEFAULT_BIOMNI_DIR,
        help=f'Root directory containing Biomni CSV outputs (default: {DEFAULT_BIOMNI_DIR})'
    )

    parser.add_argument(
        '--gt-root',
        type=Path,
        default=DEFAULT_GT_ROOT,
        help=f'Root directory containing ground truth data with tissue subdirectories (default: {DEFAULT_GT_ROOT})'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Directory to save evaluation results (default: {DEFAULT_OUTPUT_DIR})'
    )

    parser.add_argument(
        '--pred-column',
        type=str,
        default='cell_type',
        help='Column name for predictions in Biomni CSV (default: cell_type)'
    )

    parser.add_argument(
        '--gt-column',
        type=str,
        default='cell_type',
        help='Column name for ground truth in GT files (default: cell_type)'
    )

    parser.add_argument(
        '--pred-id-column',
        type=str,
        default='cell_id',
        help='Cell ID column in Biomni CSV (default: cell_id)'
    )

    parser.add_argument(
        '--gt-id-column',
        type=str,
        default='cell_barcode',
        help='Cell ID column in GT h5ad (default: cell_barcode)'
    )

    parser.add_argument(
        '--match-by-row-order',
        action='store_true',
        help='Match cells by row order instead of ID columns'
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


def auto_detect_paths() -> Tuple[Path, Path]:
    """
    Auto-detect Biomni and GT paths.
    
    Returns:
        Tuple of (biomni_dir, gt_root)
    """
    # Assume script is in celltype_eval_llm/evaluation/
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    biomni_dir = project_root / 'agent_outputs' / 'clustered' / 'biomni'
    gt_root = project_root / 'data'

    return biomni_dir, gt_root


def normalize_tissue_name(tissue: str) -> str:
    """Normalize tissue name for path matching (e.g., 'sarcoma' -> 'scarcoma')."""
    # Special case mappings if needed
    mapping = {
        'sarcoma': 'scarcoma',
        'head_and_neck': 'head_neck',
        'head-neck': 'head_neck',
    }
    normalized = mapping.get(tissue.lower(), tissue.lower())
    return normalized


def find_prediction_files(biomni_dir: Path) -> List[Tuple[str, Path, Optional[str], str]]:
    """
    Discover prediction files in Biomni directory.

    Preferred source is biomni_output.json, with CSV fallback.

    Supports nested Biomni output folders, e.g.:
    - {biomni_dir}/{tissue}_Data_*/biomni_output.json
    - {biomni_dir}/{tissue}_Data_*/cell_type_annotations.csv
    - {biomni_dir}/{tissue}_cell_type_annotations.csv (legacy flat format)

    Args:
        biomni_dir: Path to Biomni outputs directory

    Returns:
        List of tuples: (tissue, prediction_path, gt_stem, source_type)
        - gt_stem is expected GT h5ad stem (e.g. Data_Choudhury2022_Brain)
          for nested Biomni output folders.
        - gt_stem is None for legacy flat-format CSV filenames.
        - source_type is one of: 'biomni_json', 'csv'.
    """
    prediction_files: List[Tuple[str, Path, Optional[str], str]] = []
    seen_json_parent_dirs: set[Path] = set()
    
    if not biomni_dir.exists():
        return prediction_files

    # Preferred Biomni format metadata file.
    for json_file in sorted(biomni_dir.rglob('biomni_output.json')):
        parent_name = json_file.parent.name
        if '_Data_' in parent_name:
            tissue, dataset_suffix = parent_name.split('_Data_', 1)
            gt_stem = f"Data_{dataset_suffix}"
        else:
            tissue = parent_name
            gt_stem = None

        prediction_files.append((tissue, json_file, gt_stem, 'biomni_json'))
        seen_json_parent_dirs.add(json_file.parent.resolve())
    
    # CSV fallback in nested folders.
    for csv_file in sorted(biomni_dir.rglob('cell_type_annotations.csv')):
        if csv_file.parent.resolve() in seen_json_parent_dirs:
            # If biomni_output.json exists in the same dataset folder, prefer JSON source.
            continue

        # Parent directory often encodes tissue, e.g. scarcoma_Data_...
        parent_name = csv_file.parent.name
        if '_Data_' in parent_name:
            tissue, dataset_suffix = parent_name.split('_Data_', 1)
            gt_stem = f"Data_{dataset_suffix}"
        else:
            tissue = parent_name
            gt_stem = None

        prediction_files.append((tissue, csv_file, gt_stem, 'csv'))

    # Backward-compatible flat filename pattern.
    for csv_file in sorted(biomni_dir.glob('*_cell_type_annotations.csv')):
        tissue = csv_file.stem.replace('_cell_type_annotations', '')
        prediction_files.append((tissue, csv_file, None, 'csv'))
    
    return prediction_files


def resolve_prediction_csv_from_biomni_output(
    biomni_output_path: Path,
    logger: logging.Logger,
) -> Path:
    """Resolve the prediction CSV path from a biomni_output.json file."""
    if not biomni_output_path.exists():
        raise ValueError(f"Biomni output JSON does not exist: {biomni_output_path}")

    with open(biomni_output_path, 'r') as handle:
        payload = json.load(handle)

    csv_pattern = re.compile(r'([^\s`"\']*cell_type_annotations\.csv)')
    candidate_paths: List[Path] = []

    # Check explicit metadata fields first if present.
    for key in ('output_csv', 'csv_path', 'prediction_csv', 'predictions_csv'):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            candidate_paths.append(Path(value.strip()))

    # Parse free-text sections for emitted CSV paths.
    for key in ('answer',):
        value = payload.get(key)
        if isinstance(value, str):
            for match in csv_pattern.findall(value):
                candidate_paths.append(Path(match))

    log_value = payload.get('log')
    if isinstance(log_value, list):
        for line in log_value:
            if isinstance(line, str):
                for match in csv_pattern.findall(line):
                    candidate_paths.append(Path(match))

    # Always try local sibling fallback.
    candidate_paths.append(biomni_output_path.parent / 'cell_type_annotations.csv')

    checked: set[Path] = set()
    for candidate in candidate_paths:
        resolved_candidate = candidate if candidate.is_absolute() else (biomni_output_path.parent / candidate)
        resolved_candidate = resolved_candidate.resolve()
        if resolved_candidate in checked:
            continue
        checked.add(resolved_candidate)
        if resolved_candidate.exists():
            if resolved_candidate.parent == biomni_output_path.parent:
                logger.debug(f"Resolved prediction CSV from Biomni JSON sibling: {resolved_candidate}")
            else:
                logger.debug(f"Resolved prediction CSV from Biomni JSON content: {resolved_candidate}")
            return resolved_candidate

    raise ValueError(
        f"Could not resolve a prediction CSV from {biomni_output_path}. "
        "Expected to find cell_type_annotations.csv in JSON metadata/text or as a sibling file."
    )


def find_h5ad_by_stem(gt_root: Path, tissue: str, gt_stem: str) -> Optional[Path]:
    """Find a GT h5ad by exact stem within a tissue directory."""
    normalized_tissue = normalize_tissue_name(tissue)
    candidate = gt_root / normalized_tissue / 'h5ad' / f'{gt_stem}.h5ad'
    if candidate.exists():
        return candidate
    return None


def find_h5ad_files(gt_root: Path, tissue: str) -> List[Path]:
    """
    Find h5ad files for a given tissue.
    
    Expected structure: {gt_root}/{tissue}/h5ad/*.h5ad
    
    Args:
        gt_root: Root directory of ground truth data
        tissue: Tissue name
    
    Returns:
        List of paths to h5ad files
    """
    h5ad_files = []
    
    # Normalize tissue name
    normalized_tissue = normalize_tissue_name(tissue)
    tissue_dir = gt_root / normalized_tissue
    
    h5ad_dir = tissue_dir / 'h5ad'
    if h5ad_dir.exists():
        h5ad_files = sorted(h5ad_dir.glob('*.h5ad'))
    
    return h5ad_files


def load_csv_as_adata(
    csv_path: Path,
    label_column: str,
    id_column: str,
    role: str,
    logger: logging.Logger,
    use_row_index: bool = False,
) -> Tuple[ad.AnnData, bool]:
    """Load a CSV into AnnData.obs format."""
    if not csv_path.exists():
        raise ValueError(f"{role} CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)

    requested_label_column = label_column
    source_label_column = label_column

    if source_label_column not in df.columns:
        # Biomni exports use heterogeneous label column names across datasets.
        # Prefer human/consensus/final-style annotations first, then automated labels.
        if role == 'Prediction':
            prediction_label_candidates = [
                'cell_type',
                'predicted_cell_type',
                'predicted_celltype',
                'final_cell_type',
                'final_celltype',
                'consensus_annotation',
                'consensus_cell_type',
                'combined_annotation',
                'refined_cell_type',
                'manual_annotation',
                'cell_type_annotation',
                'cluster_celltype',
                'celltype',
                'cell_type_major',
                'cell_type_detailed',
                'broad_cell_type',
                'marker_based_annotation',
                'marker_based_cell_type',
                'marker_based_celltype',
                'celltypist_majority_voting',
                'celltypist_majority_vote',
                'celltypist_prediction',
                'celltypist_predicted',
                'celltypist_annotation',
            ]

            for candidate in prediction_label_candidates:
                if candidate in df.columns:
                    source_label_column = candidate
                    logger.warning(
                        f"{role} label column '{requested_label_column}' not found in {csv_path}. "
                        f"Falling back to '{candidate}'."
                    )
                    break
            else:
                raise ValueError(
                    f"{role} label column '{source_label_column}' not found in {csv_path}. "
                    f"Available columns: {list(df.columns)}"
                )
        else:
            raise ValueError(
                f"{role} label column '{source_label_column}' not found in {csv_path}. "
                f"Available columns: {list(df.columns)}"
            )

    obs = df[[source_label_column]].copy()
    if source_label_column != requested_label_column:
        obs = obs.rename(columns={source_label_column: requested_label_column})

    effective_use_row_index = use_row_index

    if not effective_use_row_index and id_column not in df.columns:
        if role == 'Prediction':
            logger.warning(
                f"{role} ID column '{id_column}' not found in {csv_path}. "
                "Falling back to row-order matching for this dataset pair."
            )
            effective_use_row_index = True
        else:
            raise ValueError(
                f"{role} ID column '{id_column}' not found in {csv_path}. "
                f"Available columns: {list(df.columns)}"
            )

    if effective_use_row_index:
        obs.index = pd.Index([str(i) for i in range(len(obs))], name='obs_row')
    else:
        obs.index = df[id_column].astype(str).values
        obs.index.name = id_column

    if obs.index.has_duplicates:
        duplicate_count = int(obs.index.duplicated().sum())
        logger.warning(
            f"{role} CSV has {duplicate_count} duplicate IDs. "
            "Keeping first occurrence for each ID."
        )
        obs = obs[~obs.index.duplicated(keep='first')].copy()

    return ad.AnnData(X=np.empty((len(obs), 0), dtype=np.float32), obs=obs), effective_use_row_index


def match_csv_to_h5ad_files(
    csv_path: Path,
    h5ad_files: List[Path],
    logger: logging.Logger
) -> List[Tuple[str, Path]]:
    """
    Match a Biomni CSV to one or more h5ad files.
    
    If multiple h5ad files exist for a tissue, create separate evaluation pairs.
    
    Args:
        csv_path: Path to Biomni CSV
        h5ad_files: List of h5ad files for the tissue
        logger: Logger instance
    
    Returns:
        List of (dataset_name, h5ad_path) tuples
    """
    pairs = []
    
    if not h5ad_files:
        logger.warning(f"No h5ad files found for {csv_path.stem}")
        return pairs
    
    for h5ad_file in h5ad_files:
        # Extract dataset name from h5ad filename
        # Format: Data_<Author><Year>_<Tissue>.h5ad
        stem = h5ad_file.stem
        # Remove 'Data_' prefix and tissue suffix
        dataset_name = stem.replace('Data_', '')
        # Remove trailing _<Tissue>
        parts = dataset_name.rsplit('_', 1)
        if len(parts) == 2:
            dataset_name = parts[0]
        
        pairs.append((dataset_name, h5ad_file))
    
    return pairs


def run_batch_evaluation(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Run batch evaluation for all Biomni outputs."""
    # Setup paths
    if args.biomni_dir is None or args.gt_root is None:
        logger.info("Auto-detecting paths...")
        biomni_dir, gt_root = auto_detect_paths()
        if args.biomni_dir is None:
            args.biomni_dir = biomni_dir
        if args.gt_root is None:
            args.gt_root = gt_root
    
    logger.info("=" * 70)
    logger.info("BIOMNI BATCH EVALUATION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Biomni output directory: {args.biomni_dir}")
    logger.info(f"Ground truth root: {args.gt_root}")
    logger.info(f"Results will be saved to: {args.output_dir}")
    
    # Validate paths
    if not args.biomni_dir.exists():
        logger.error(f"Biomni directory does not exist: {args.biomni_dir}")
        return 1
    
    if not args.gt_root.exists():
        logger.error(f"GT root does not exist: {args.gt_root}")
        return 1
    
    # Discover prediction files
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: DISCOVERING PREDICTION FILES")
    logger.info("=" * 70)
    
    prediction_files = find_prediction_files(args.biomni_dir)
    
    if not prediction_files:
        logger.error(f"No Biomni prediction files found in {args.biomni_dir}")
        return 1
    
    logger.info(f"Found {len(prediction_files)} prediction file(s):")
    for tissue, prediction_path, gt_stem, source_type in prediction_files:
        if gt_stem:
            logger.info(
                f"  - {tissue}: {prediction_path.parent.name}/{prediction_path.name} "
                f"[{source_type}] -> {gt_stem}.h5ad"
            )
        else:
            logger.info(
                f"  - {tissue}: {prediction_path.name} [{source_type}] "
                "(legacy tissue-level matching)"
            )
    
    # Create evaluation pairs
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: MATCHING FILES WITH GROUND TRUTH")
    logger.info("=" * 70)
    
    dataset_pairs: List[Tuple[str, Path, str, Path, str]] = []
    # (tissue, prediction_input_path, source_type, h5ad, dataset_name)
    discovery_failures: List[Dict[str, str]] = []
    
    for tissue, prediction_input_path, gt_stem, source_type in prediction_files:
        matches: List[Tuple[str, Path]] = []

        if gt_stem:
            exact_h5ad = find_h5ad_by_stem(args.gt_root, tissue, gt_stem)
            if exact_h5ad is not None:
                matches = match_csv_to_h5ad_files(prediction_input_path, [exact_h5ad], logger)
            else:
                error_message = (
                    f"Exact GT match not found for {prediction_input_path.parent.name} "
                    f"(expected {gt_stem}.h5ad)"
                )
                logger.warning(f"\n{tissue.upper()}: {error_message}")
                discovery_failures.append({
                    'tissue': tissue,
                    'dataset_name': gt_stem.replace('Data_', '', 1),
                    'error': error_message,
                })
        else:
            # Backward-compatible path for older flat exports that do not
            # encode dataset identity in folder name.
            h5ad_files = find_h5ad_files(args.gt_root, tissue)
            matches = match_csv_to_h5ad_files(prediction_input_path, h5ad_files, logger)
        
        if matches:
            logger.info(f"\n{tissue.upper()}:")
            logger.info(
                f"  Prediction source ({source_type}): {prediction_input_path}"
            )
            for dataset_name, h5ad_file in matches:
                logger.info(f"    -> {h5ad_file.name}")
                dataset_pairs.append((tissue, prediction_input_path, source_type, h5ad_file, dataset_name))
        else:
            logger.warning(f"\n{tissue.upper()}: No ground truth files found")
    
    if not dataset_pairs:
        logger.error("No matching dataset pairs found. Exiting.")
        if discovery_failures:
            logger.info("\nDiscovery failures:")
            for result in discovery_failures:
                logger.info(f"  ✗ {result['tissue']}/{result['dataset_name']}: {result['error']}")
        return 2 if discovery_failures else 1
    
    logger.info(f"\nTotal matched pairs: {len(dataset_pairs)}")
    
    # Initialize evaluator
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: EVALUATING DATASETS")
    logger.info("=" * 70)
    
    evaluator = BatchEvaluator(
        output_dir=args.output_dir,
        pred_column=args.pred_column,
        gt_column=args.gt_column,
        skip_llm=args.skip_llm,
        save_plots=not args.no_plots
    )
    
    # Run evaluation on each pair
    results_summary = {
        'successful': [],
        'failed': [],
        'metrics': {}
    }
    results_summary['failed'].extend(discovery_failures)
    
    for idx, (tissue, prediction_input_path, source_type, h5ad_file, dataset_name) in enumerate(dataset_pairs, 1):
        logger.info(f"\n[{idx}/{len(dataset_pairs)}] Evaluating: {tissue} / {dataset_name}")
        logger.info("-" * 70)
        
        try:
            # Resolve prediction CSV from selected source.
            if source_type == 'biomni_json':
                csv_path = resolve_prediction_csv_from_biomni_output(prediction_input_path, logger)
                logger.info(f"  Resolved prediction CSV from biomni_output.json: {csv_path}")
            else:
                csv_path = prediction_input_path

            # Load prediction CSV.
            pred_adata, pred_used_row_index = load_csv_as_adata(
                csv_path=csv_path,
                label_column=args.pred_column,
                id_column=args.pred_id_column,
                role='Prediction',
                logger=logger,
                use_row_index=args.match_by_row_order,
            )
            logger.info(f"  Loaded {len(pred_adata)} predictions")
            
            # Load GT h5ad
            if pred_used_row_index:
                gt_adata = ad.read_h5ad(h5ad_file)
                gt_adata.obs.index = pd.Index(
                    [str(i) for i in range(len(gt_adata))],
                    name='obs_row'
                )
            else:
                gt_adata = ad.read_h5ad(h5ad_file)
            logger.info(f"  Loaded {len(gt_adata)} ground truth samples")
            
            # Create results directory
            safe_name = ''.join(
                c if c.isalnum() or c in ('-', '_') else '_'
                for c in f"{tissue}_{dataset_name}"
            )
            results_dir = args.output_dir / f"biomni_{safe_name}"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Run evaluation
            metrics_json_path = results_dir / 'evaluation_metrics.json'
            metrics = evaluator.evaluate_h5ad(
                pred_h5ad=pred_adata,
                pred_column=args.pred_column,
                gt_h5ad=gt_adata,
                gt_column=args.gt_column,
                metrics_output_path=metrics_json_path,
                skip_llm=args.skip_llm,
            )
            
            # Save results
            pair = DatasetPair(
                tissue=tissue,
                dataset_name=dataset_name,
                gt_file=h5ad_file,
                pred_file=prediction_input_path,
                match_score=1.0,
            )
            evaluator.save_results(metrics, pair, results_dir)
            
            # Log metrics
            logger.info(f"  ✓ Evaluation complete")
            logger.info(f"    Accuracy:    {metrics['accuracy']:.4f}")
            logger.info(f"    Macro F1:    {metrics['macro_f1']:.4f}")
            logger.info(f"    Weighted F1: {metrics['weighted_f1']:.4f}")
            
            results_summary['successful'].append({
                'tissue': tissue,
                'dataset_name': dataset_name,
                'results_dir': str(results_dir)
            })
            results_summary['metrics'][f"{tissue}_{dataset_name}"] = metrics
            
        except Exception as e:
            logger.error(f"  ✗ Evaluation failed: {e}")
            logger.exception(e)
            results_summary['failed'].append({
                'tissue': tissue,
                'dataset_name': dataset_name,
                'error': str(e)
            })
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 70)
    
    successful = len(results_summary['successful'])
    failed = len(results_summary['failed'])
    total = successful + failed
    
    logger.info(f"Total evaluations: {total}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    if results_summary['successful']:
        logger.info("\nSuccessful evaluations:")
        for result in results_summary['successful']:
            logger.info(f"  ✓ {result['tissue']}/{result['dataset_name']}")
    
    if results_summary['failed']:
        logger.info("\nFailed evaluations:")
        for result in results_summary['failed']:
            logger.info(f"  ✗ {result['tissue']}/{result['dataset_name']}: {result['error']}")
    
    logger.info(f"\nResults saved to: {args.output_dir}")
    
    if failed > 0:
        return 2  # Partial success
    
    return 0


def main() -> int:
    """Main entry point."""
    args = parse_arguments()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        return run_batch_evaluation(args, logger)
    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.exception(e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
