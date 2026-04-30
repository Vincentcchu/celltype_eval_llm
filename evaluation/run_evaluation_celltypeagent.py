#!/usr/bin/env python3
"""
Batch evaluation CLI for CellTypeAgent predictions.

CellTypeAgent outputs cluster-level predictions in CSV format. This script:
1. Matches ground truth h5ad files with CellTypeAgent outputs
2. Converts cluster-level predictions to cell-level predictions (in-memory)
3. Evaluates using the standard evaluation pipeline

Usage:
    # Run evaluation on all tissues
    python run_evaluation_celltypeagent.py

    # Run evaluation on one or more specific tissues
    python run_evaluation_celltypeagent.py --tissue brain
    python run_evaluation_celltypeagent.py --tissue brain breast

    # Skip LLM calls (use existing mappings only)
    python run_evaluation_celltypeagent.py --tissue brain --skip-llm

    # Disable plot generation
    python run_evaluation_celltypeagent.py --no-plots
"""

import argparse
import ast
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, cast

import anndata as ad
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
    pass  # Will fall back to existing env var

# Add parent directory to path to import celltype_standardizer
sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_evaluation import BatchEvaluator
from batch_evaluation.file_matcher import DatasetPair

logger = logging.getLogger(__name__)


@dataclass
class CellTypeAgentDatasetPair:
    """Represents a matched GT file, CellTypeAgent CSV, and clustered h5ad."""
    tissue: str
    dataset_name: str
    gt_file: Path
    pred_csv: Path
    clustered_h5ad: Path
    match_score: float = 1.0


class CellTypeAgentFileMatcher:
    """Matches GT files with CellTypeAgent output CSVs and clustered h5ad files."""

    def __init__(self, data_root: Path, output_root: Path):
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)

        if not self.data_root.exists():
            raise ValueError(f"Data root does not exist: {self.data_root}")
        if not self.output_root.exists():
            raise ValueError(f"Output root does not exist: {self.output_root}")

    def get_available_tissues(self) -> List[str]:
        tissues = []
        for item in self.data_root.iterdir():
            if not item.is_dir() or item.name.startswith('.'):
                continue
            if (item / 'h5ad').exists() and (self.output_root / item.name).exists():
                tissues.append(item.name)
        return sorted(tissues)

    @staticmethod
    def _extract_full_dataset_name(gt_filename: str) -> Optional[str]:
        match = re.search(r'^Data_(.+)\.h5ad$', gt_filename)
        return match.group(1) if match else None

    def find_ground_truth_files(self, tissue: str) -> List[Path]:
        h5ad_dir = self.data_root / tissue / 'h5ad'
        if not h5ad_dir.exists():
            logger.warning(f"h5ad directory not found for tissue '{tissue}': {h5ad_dir}")
            return []
        return sorted(h5ad_dir.glob('*.h5ad'))

    def _find_prediction_csv(self, tissue: str, dataset_full_name: str) -> Optional[Path]:
        tissue_output_dir = self.output_root / tissue
        if not tissue_output_dir.exists():
            return None

        expected_dir = tissue_output_dir / f"Data_{dataset_full_name}_formatted"
        dataset_dirs: List[Path]
        if expected_dir.exists() and expected_dir.is_dir():
            dataset_dirs = [expected_dir]
        else:
            prefix = f"Data_{dataset_full_name}"
            dataset_dirs = [
                item for item in tissue_output_dir.iterdir()
                if item.is_dir() and item.name.startswith(prefix)
            ]

        if not dataset_dirs:
            return None

        pred_candidates: List[Path] = []
        for dataset_dir in dataset_dirs:
            prediction_root = dataset_dir / 'prediction'
            if not prediction_root.exists():
                continue

            exact_name = f"{dataset_dir.name}.csv"
            pred_candidates.extend(prediction_root.glob(f"**/{exact_name}"))

            if not pred_candidates:
                pred_candidates.extend(
                    p for p in prediction_root.glob('**/*.csv')
                    if p.name not in {'run_metrics_summary.csv'}
                )

        if not pred_candidates:
            return None

        return max(pred_candidates, key=lambda path: path.stat().st_mtime)

    def _find_clustered_h5ad(self, tissue: str, dataset_full_name: str) -> Optional[Path]:
        # h5ad_path = (
        #     self.data_root
        #     / tissue
        #     / 'celltypeagent_format'
        #     / f"Data_{dataset_full_name}_clustered.h5ad"
        # )
        h5ad_path = (
            self.data_root
            / tissue
            / 'h5ad_unlabelled_clustered'
            / f"Data_{dataset_full_name}.h5ad"
        )
        return h5ad_path if h5ad_path.exists() else None

    def match_datasets(self, tissues: Optional[List[str]] = None) -> List[CellTypeAgentDatasetPair]:
        if tissues is None:
            tissues = self.get_available_tissues()
        else:
            available = self.get_available_tissues()
            invalid = [t for t in tissues if t not in available]
            if invalid:
                raise ValueError(f"Invalid tissue names: {invalid}. Available tissues: {available}")

        matched_pairs: List[CellTypeAgentDatasetPair] = []
        skipped = []

        for tissue in tissues:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing tissue: {tissue}")
            logger.info(f"{'=' * 60}")

            for gt_file in self.find_ground_truth_files(tissue):
                dataset_full_name = self._extract_full_dataset_name(gt_file.name)
                if dataset_full_name is None:
                    skipped.append((tissue, gt_file.name, 'Invalid GT filename pattern'))
                    continue

                pred_csv = self._find_prediction_csv(tissue, dataset_full_name)
                if pred_csv is None:
                    skipped.append((tissue, gt_file.name, 'No matching CellTypeAgent prediction CSV'))
                    continue

                clustered_h5ad = self._find_clustered_h5ad(tissue, dataset_full_name)
                if clustered_h5ad is None:
                    skipped.append((tissue, gt_file.name, 'Missing clustered h5ad in celltypeagent_format'))
                    continue

                matched_pairs.append(
                    CellTypeAgentDatasetPair(
                        tissue=tissue,
                        dataset_name=dataset_full_name,
                        gt_file=gt_file,
                        pred_csv=pred_csv,
                        clustered_h5ad=clustered_h5ad,
                    )
                )
                logger.info(f"✓ Matched: {tissue}/{dataset_full_name}")

        logger.info(f"\n{'=' * 60}")
        logger.info('MATCHING SUMMARY')
        logger.info(f"{'=' * 60}")
        logger.info(f"Successfully matched: {len(matched_pairs)} dataset pairs")

        if skipped:
            logger.info(f"Skipped: {len(skipped)} datasets")
            for tissue, filename, reason in skipped:
                logger.info(f"  - {tissue}/{filename}: {reason}")

        return matched_pairs


class CellTypeAgentAdapter:
    """Converts CellTypeAgent cluster-level predictions to cell-level predictions."""

    def __init__(
        self,
        pred_column: str = 'cell_type_pred',
        output_pred_column: str = 'predicted_cell_type',
        cluster_column: str = 'cluster',
    ):
        self.pred_column = pred_column
        self.output_pred_column = output_pred_column
        self.cluster_column = cluster_column

    @staticmethod
    def _normalize_cluster_id(value: object) -> str:
        value_str = str(value).strip()
        if re.fullmatch(r'-?\d+\.0+', value_str):
            return str(int(float(value_str)))
        return value_str

    @staticmethod
    def _extract_cluster_id(raw_cluster_value: object) -> str:
        text = str(raw_cluster_value).strip()
        match = re.search(r'cluster\s*(-?\d+)', text, re.IGNORECASE)
        if match:
            return match.group(1)
        return CellTypeAgentAdapter._normalize_cluster_id(text)

    @staticmethod
    def _first_scalar(value: object) -> Optional[str]:
        if value is None:
            return None

        if isinstance(value, float) and pd.isna(value):
            return None

        if isinstance(value, str):
            raw = value.strip()
            if raw == '' or raw.lower() == 'nan':
                return None

            try:
                parsed = ast.literal_eval(raw)
            except Exception:
                return raw

            return CellTypeAgentAdapter._first_scalar(parsed)

        if isinstance(value, (list, tuple)):
            if not value:
                return None
            return CellTypeAgentAdapter._first_scalar(value[0])

        return str(value)

    def _build_cluster_prediction_map(self, pred_df: pd.DataFrame) -> Dict[str, str]:
        if self.pred_column not in pred_df.columns:
            raise ValueError(
                f"Prediction CSV missing '{self.pred_column}' column. "
                f"Available columns: {list(pred_df.columns)}"
            )

        cluster_col = None
        for candidate in ('manual_annotation', 'cluster', 'Cluster'):
            if candidate in pred_df.columns:
                cluster_col = candidate
                break

        cluster_to_pred: Dict[str, str] = {}

        for row_idx, row in pred_df.iterrows():
            cluster_id = (
                self._extract_cluster_id(row[cluster_col])
                if cluster_col is not None
                else str(row_idx)
            )
            first_prediction = self._first_scalar(row[self.pred_column])
            cluster_to_pred[self._normalize_cluster_id(cluster_id)] = (
                first_prediction if first_prediction else 'Unknown'
            )

        return cluster_to_pred

    def create_cell_level_predictions(
        self,
        csv_path: Path,
        h5ad_path: Path,
        return_adata: bool = True,
    ) -> tuple[Optional[Path], Optional[ad.AnnData]]:
        logger.info(f"Loading CellTypeAgent CSV: {csv_path}")
        logger.info(f"Loading clustered h5ad: {h5ad_path}")

        pred_df = pd.read_csv(csv_path)
        cluster_to_pred = self._build_cluster_prediction_map(pred_df)
        logger.info(f"Loaded predictions for {len(cluster_to_pred)} clusters")

        adata = ad.read_h5ad(h5ad_path)

        cluster_column = self.cluster_column
        if cluster_column not in adata.obs.columns:
            if 'leiden' in adata.obs.columns:
                cluster_column = 'leiden'
                logger.warning("'cluster' column not found; falling back to 'leiden'")
            else:
                raise ValueError(
                    f"Cluster column '{self.cluster_column}' not found in h5ad. "
                    f"Available columns: {list(adata.obs.columns)}"
                )

        cluster_ids = adata.obs[cluster_column].map(self._normalize_cluster_id)
        cell_predictions = cluster_ids.map(cluster_to_pred)

        unmapped = int(cell_predictions.isna().sum())
        if unmapped > 0:
            logger.warning(
                f"{unmapped} cells have no cluster prediction "
                f"({unmapped / len(cell_predictions) * 100:.1f}%). Filling with 'Unknown'."
            )
            cell_predictions = cell_predictions.fillna('Unknown')

        adata.obs[self.output_pred_column] = cell_predictions.values
        logger.info(f"✓ Added '{self.output_pred_column}' for {adata.n_obs} cells")

        if return_adata:
            return None, adata

        return None, adata


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
        description='Batch evaluation pipeline for CellTypeAgent predictions',
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
        help='Root directory containing CellTypeAgent outputs (default: auto-detect)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('evaluation_results_celltypeagent'),
        help='Directory to save evaluation results (default: evaluation_results_celltypeagent)'
    )
    parser.add_argument(
        '--pred-column',
        type=str,
        default='predicted_cell_type',
        help='Column name for converted cell-level predictions (default: predicted_cell_type)'
    )
    parser.add_argument(
        '--csv-pred-column',
        type=str,
        default='cell_type_pred',
        help='Column name in CellTypeAgent CSV (default: cell_type_pred)'
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
        help="Cluster column name in clustered h5ad (default: cluster; falls back to 'leiden')"
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
    """Auto-detect data root and CellTypeAgent output root paths."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    data_root = project_root / 'data'
    output_root = project_root / 'agent_outputs' / 'clustered' / 'celltypeagent2'

    return data_root, output_root


def convert_celltypeagent_to_cell_level(
    pairs: List[CellTypeAgentDatasetPair],
    adapter: CellTypeAgentAdapter,
) -> List[tuple[CellTypeAgentDatasetPair, ad.AnnData]]:
    """Convert CellTypeAgent cluster-level predictions to cell-level predictions in memory."""
    logger.info("\n" + "=" * 60)
    logger.info('STEP 2: CONVERTING CLUSTER PREDICTIONS TO CELL LEVEL')
    logger.info("=" * 60)

    converted_pairs: List[tuple[CellTypeAgentDatasetPair, ad.AnnData]] = []

    for i, pair in enumerate(pairs, 1):
        logger.info(f"\nProcessing {i}/{len(pairs)}: {pair.tissue}/{pair.dataset_name}")

        try:
            _, adata_obj = adapter.create_cell_level_predictions(
                csv_path=pair.pred_csv,
                h5ad_path=pair.clustered_h5ad,
                return_adata=True,
            )
            if adata_obj is None:
                raise RuntimeError('Adapter returned no AnnData object')

            converted_pairs.append((pair, adata_obj))

        except Exception as e:
            logger.error(f"Failed to convert {pair.tissue}/{pair.dataset_name}: {e}")
            logger.exception(e)

    logger.info(f"\n✓ Successfully converted {len(converted_pairs)}/{len(pairs)} datasets")
    return converted_pairs


def main() -> int:
    """Main entry point for the CLI."""
    args = parse_arguments()

    setup_logging(args.verbose)
    logger.info('=' * 60)
    logger.info('CELLTYPEAGENT EVALUATION PIPELINE')
    logger.info('=' * 60)

    try:
        if args.data_root is None or args.output_root is None:
            logger.info('Auto-detecting paths for CellTypeAgent...')
            data_root, output_root = auto_detect_paths()
            if args.data_root is None:
                args.data_root = data_root
            if args.output_root is None:
                args.output_root = output_root

        logger.info(f"Data root: {args.data_root}")
        logger.info(f"CellTypeAgent output root: {args.output_root}")
        logger.info(f"Results will be saved to: {args.output_dir}")

        if not args.data_root.exists():
            logger.error(f"Data root does not exist: {args.data_root}")
            return 1
        if not args.output_root.exists():
            logger.error(f"CellTypeAgent output root does not exist: {args.output_root}")
            return 1

        logger.info("\n" + '=' * 60)
        logger.info('STEP 1: MATCHING FILES')
        logger.info('=' * 60)

        matcher = CellTypeAgentFileMatcher(
            data_root=args.data_root,
            output_root=args.output_root,
        )

        available_tissues = matcher.get_available_tissues()
        logger.info(f"Available tissues: {', '.join(available_tissues)}")

        if args.tissue:
            invalid_tissues = [t for t in args.tissue if t not in available_tissues]
            if invalid_tissues:
                logger.error(f"Invalid tissue names: {', '.join(invalid_tissues)}")
                logger.error(f"Available tissues: {', '.join(available_tissues)}")
                return 1
            tissues_to_process = args.tissue
            logger.info(f"Processing tissues: {', '.join(tissues_to_process)}")
        else:
            tissues_to_process = None
            logger.info(f"Processing all tissues: {', '.join(available_tissues)}")

        matched_pairs = matcher.match_datasets(tissues=tissues_to_process)
        if not matched_pairs:
            logger.error('No matched dataset pairs found. Exiting.')
            return 1

        logger.info(f"\nFound {len(matched_pairs)} matched dataset pairs")

        adapter = CellTypeAgentAdapter(
            pred_column=args.csv_pred_column,
            output_pred_column=args.pred_column,
            cluster_column=args.cluster_column,
        )

        converted_pairs = convert_celltypeagent_to_cell_level(matched_pairs, adapter)
        if not converted_pairs:
            logger.error('No datasets successfully converted. Exiting.')
            return 1

        logger.info("\n" + '=' * 60)
        logger.info('STEP 3: EVALUATING DATASETS')
        logger.info('=' * 60)

        evaluator = BatchEvaluator(
            output_dir=args.output_dir,
            pred_column=args.pred_column,
            gt_column=args.gt_column,
            skip_llm=args.skip_llm,
            save_plots=not args.no_plots,
        )

        eval_pairs: List[DatasetPair] = []
        for pair, adata_obj in converted_pairs:
            eval_pairs.append(
                DatasetPair(
                    tissue=pair.tissue,
                    dataset_name=pair.dataset_name,
                    gt_file=pair.gt_file,
                    pred_file=cast(Path, adata_obj),
                    match_score=pair.match_score,
                )
            )

        results = evaluator.evaluate_batch(eval_pairs)
        evaluator.print_summary(results)

        if results['failed'] > 0:
            logger.warning(f"{results['failed']} evaluations failed")
            return 2

        return 0

    except KeyboardInterrupt:
        logger.info('\nEvaluation interrupted by user')
        return 130

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.exception(e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
