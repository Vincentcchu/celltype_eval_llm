"""
File matching logic for CASSIA ground truth and prediction files.

This module handles:
1. Scanning ground truth files in data/<tissue>/h5ad/
2. Finding matching CASSIA prediction files in cell_agents/agents/CASSIA/CASSIA_run/output/
3. Validating presence of both CSV and clustered h5ad files
"""

import re
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CASSIADatasetPair:
    """Represents a matched ground truth and CASSIA prediction file pair."""
    tissue: str
    dataset_name: str
    gt_file: Path
    pred_csv: Path
    clustered_h5ad: Path
    match_score: float = 1.0  # Confidence score for the match


class CASSIAFileMatcher:
    """Handles matching of ground truth and CASSIA prediction files."""
    
    def __init__(
        self,
        data_root: Path,
        output_root: Path
    ):
        """
        Initialize the CASSIA file matcher.
        
        Args:
            data_root: Root directory containing tissue data (e.g., .../data/)
            output_root: Root directory containing CASSIA outputs (e.g., .../CASSIA_run/output/)
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        
        if not self.data_root.exists():
            raise ValueError(f"Data root does not exist: {self.data_root}")
        if not self.output_root.exists():
            raise ValueError(f"Output root does not exist: {self.output_root}")
    
    def get_available_tissues(self) -> List[str]:
        """Get list of available tissues from data directory."""
        tissues = []
        for item in self.data_root.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if it has h5ad subdirectory
                h5ad_dir = item / 'h5ad'
                if h5ad_dir.exists() and h5ad_dir.is_dir():
                    tissues.append(item.name)
        return sorted(tissues)
    
    def extract_dataset_identifier(self, filename: str) -> Optional[str]:
        """
        Extract dataset identifier from filename.
        
        Examples:
            Data_Caron2020_Hematologic.h5ad -> Caron2020_Hematologic
            Data_Choudhury2022_Brain.h5ad -> Choudhury2022_Brain
        
        Args:
            filename: Name of the h5ad file
        
        Returns:
            Dataset identifier or None if pattern doesn't match
        """
        # Pattern: Data_<Identifier>.h5ad
        match = re.search(r'Data_(.+)\.h5ad$', filename)
        if match:
            return match.group(1)
        return None
    
    def find_ground_truth_files(self, tissue: str) -> List[Path]:
        """
        Find all ground truth .h5ad files for a given tissue.
        
        Args:
            tissue: Tissue name (e.g., 'brain', 'breast')
        
        Returns:
            List of paths to ground truth .h5ad files
        """
        h5ad_dir = self.data_root / tissue / 'h5ad'
        if not h5ad_dir.exists():
            logger.warning(f"h5ad directory not found for tissue '{tissue}': {h5ad_dir}")
            return []
        
        gt_files = list(h5ad_dir.glob('*.h5ad'))
        logger.info(f"Found {len(gt_files)} ground truth files for tissue '{tissue}'")
        return sorted(gt_files)
    
    def find_cassia_output(
        self,
        tissue: str,
        dataset_identifier: str
    ) -> Optional[Tuple[Path, Path, float]]:
        """
        Find the CASSIA output directory and validate required files.
        
        CASSIA structure:
            output/<tissue>/Data_<identifier>/
                Data_<identifier>_FINAL_RESULTS.csv
                Data_<identifier>_clustered.h5ad
        
        Args:
            tissue: Tissue name
            dataset_identifier: Dataset identifier (e.g., 'Caron2020_Hematologic')
        
        Returns:
            Tuple of (csv_path, h5ad_path, match_score) or None if not found
        """
        # Expected directory name: Data_<identifier>
        expected_dir_name = f"Data_{dataset_identifier}"
        
        # Look in output/<tissue>/
        tissue_output_dir = self.output_root / tissue
        if not tissue_output_dir.exists():
            logger.warning(f"Tissue output directory not found: {tissue_output_dir}")
            return None
        
        # Find matching directory
        dataset_dir = tissue_output_dir / expected_dir_name
        if not dataset_dir.exists():
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            return None
        
        # Check for required files
        csv_file = dataset_dir / f"{expected_dir_name}_FINAL_RESULTS.csv"
        h5ad_file = dataset_dir / f"{expected_dir_name}_clustered.h5ad"
        
        # Fallback to summary.csv in intermediate_files if FINAL_RESULTS doesn't exist
        if not csv_file.exists():
            csv_file = dataset_dir / "intermediate_files" / f"{expected_dir_name}_summary.csv"
        
        if not csv_file.exists():
            logger.warning(f"CSV file not found in {dataset_dir}")
            return None
        
        if not h5ad_file.exists():
            logger.warning(f"Clustered h5ad file not found: {h5ad_file}")
            return None
        
        logger.info(f"Found CASSIA output:")
        logger.info(f"  CSV: {csv_file.name}")
        logger.info(f"  H5AD: {h5ad_file.name}")
        
        return csv_file, h5ad_file, 1.0
    
    def match_datasets(self, tissues: Optional[List[str]] = None) -> List[CASSIADatasetPair]:
        """
        Match ground truth and CASSIA prediction files for specified tissues.
        
        Args:
            tissues: List of tissue names to process. If None, process all tissues.
        
        Returns:
            List of matched CASSIADatasetPair objects
        """
        if tissues is None:
            tissues = self.get_available_tissues()
        else:
            # Validate tissue names
            available = self.get_available_tissues()
            invalid = [t for t in tissues if t not in available]
            if invalid:
                logger.error(
                    f"Invalid tissue names: {invalid}. "
                    f"Available tissues: {available}"
                )
                raise ValueError(f"Invalid tissue names: {invalid}")
        
        matched_pairs = []
        skipped = []
        
        for tissue in tissues:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing tissue: {tissue}")
            logger.info(f"{'='*60}")
            
            gt_files = self.find_ground_truth_files(tissue)
            
            for gt_file in gt_files:
                dataset_id = self.extract_dataset_identifier(gt_file.name)
                
                if dataset_id is None:
                    logger.warning(
                        f"Could not extract dataset identifier from {gt_file.name}. Skipping."
                    )
                    skipped.append((tissue, gt_file.name, "Invalid filename pattern"))
                    continue
                
                # Find matching CASSIA output
                result = self.find_cassia_output(tissue, dataset_id)
                
                if result is None:
                    logger.warning(
                        f"No CASSIA output found for {tissue}/{dataset_id}. Skipping."
                    )
                    skipped.append((tissue, gt_file.name, "No matching CASSIA output"))
                    continue
                
                csv_file, h5ad_file, match_score = result
                
                pair = CASSIADatasetPair(
                    tissue=tissue,
                    dataset_name=dataset_id,
                    gt_file=gt_file,
                    pred_csv=csv_file,
                    clustered_h5ad=h5ad_file,
                    match_score=match_score
                )
                
                matched_pairs.append(pair)
                logger.info(f"✓ Matched: {tissue}/{dataset_id}")
        
        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info("MATCHING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Successfully matched: {len(matched_pairs)} dataset pairs")
        
        if skipped:
            logger.info(f"Skipped: {len(skipped)} datasets")
            for tissue, filename, reason in skipped:
                logger.info(f"  - {tissue}/{filename}: {reason}")
        
        return matched_pairs
