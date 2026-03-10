"""
File matching logic for ground truth and prediction files.

This module handles:
1. Scanning ground truth files in data/<tissue>/h5ad/
2. Finding matching prediction files in cell_agents/agents/biomaster/output/
3. Handling edge cases (no match, multiple matches)
"""

import re
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DatasetPair:
    """Represents a matched ground truth and prediction file pair."""
    tissue: str
    dataset_name: str
    gt_file: Path
    pred_file: Path
    match_score: float = 1.0  # Confidence score for the match


class FileMatcher:
    """Handles matching of ground truth and prediction files."""
    
    def __init__(
        self,
        data_root: Path,
        output_root: Path,
        agent_name: str = "biomaster"
    ):
        """
        Initialize the file matcher.
        
        Args:
            data_root: Root directory containing tissue data (e.g., .../data/)
            output_root: Root directory containing agent outputs (e.g., .../cell_agents/agents/biomaster/output/)
            agent_name: Name of the agent (default: "biomaster")
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.agent_name = agent_name
        
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
            Data_Choudhury2022_Brain.h5ad -> Choudhury2022
            Data_Filbin2018_Brain.h5ad -> Filbin2018
        
        Args:
            filename: Name of the h5ad file
        
        Returns:
            Dataset identifier or None if pattern doesn't match
        """
        # Pattern: Data_<Identifier>_<Tissue>.h5ad
        match = re.search(r'Data_([^_]+)', filename)
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
    
    def find_prediction_file(
        self,
        tissue: str,
        dataset_identifier: str
    ) -> Optional[Tuple[Path, float]]:
        """
        Find the prediction file matching the dataset identifier.
        
        Supports two structures:
        1. Flat files: output_root/brain_choudhury2022_*.h5ad (mLLMCellType)
        2. Subdirectories: output_root/brain_choudhury2022_*/annotated.h5ad (biomaster)
        
        Args:
            tissue: Tissue name
            dataset_identifier: Dataset identifier (e.g., 'Choudhury2022')
        
        Returns:
            Tuple of (prediction file path, match score) or None if not found
        """
        # Build regex pattern: e.g., brain_choudhury2022.*\.h5ad$
        # Case-insensitive pattern matching tissue_identifier_*.h5ad
        pattern = re.compile(
            rf"^{re.escape(tissue)}_{re.escape(dataset_identifier.lower())}.*\.h5ad$",
            re.IGNORECASE
        )
        
        # First, try to find matching .h5ad files directly in output_root (flat structure)
        matching_files = []
        for item in self.output_root.iterdir():
            if item.is_file() and pattern.match(item.name):
                matching_files.append(item)
        
        if matching_files:
            if len(matching_files) > 1:
                logger.warning(
                    f"Multiple matching h5ad files found for {tissue}_{dataset_identifier}: "
                    f"{[f.name for f in matching_files]}. "
                    f"Using first match: {matching_files[0].name}"
                )
            
            pred_file = matching_files[0]
            logger.info(f"Found prediction file: {pred_file.name}")
            return pred_file, 1.0
        
        # If no flat files found, try subdirectory structure (biomaster style)
        # Pattern for directory names: tissue_identifier*
        dir_pattern = re.compile(
            rf"^{re.escape(tissue)}_{re.escape(dataset_identifier.lower())}",
            re.IGNORECASE
        )
        
        matching_dirs = []
        for item in self.output_root.iterdir():
            if item.is_dir() and dir_pattern.match(item.name):
                matching_dirs.append(item)
        
        if not matching_dirs:
            logger.warning(
                f"No matching output file or directory found for {tissue}_{dataset_identifier}. "
                f"Searched in: {self.output_root}"
            )
            return None
        
        if len(matching_dirs) > 1:
            logger.warning(
                f"Multiple matching directories found for {tissue}_{dataset_identifier}: "
                f"{[d.name for d in matching_dirs]}. "
                f"Using first match: {matching_dirs[0].name}"
            )
        
        best_dir = matching_dirs[0]
        
        # Find the most recent .h5ad file in the directory
        h5ad_files = list(best_dir.glob('*.h5ad'))
        if not h5ad_files:
            logger.warning(f"No .h5ad files found in {best_dir}")
            return None
        
        # Prefer files with 'annotated' in the name, otherwise use most recent
        annotated_files = [f for f in h5ad_files if 'annotated' in f.name.lower()]
        if annotated_files:
            # Use most recent annotated file
            pred_file = max(annotated_files, key=lambda f: f.stat().st_mtime)
        else:
            # Use most recent file
            pred_file = max(h5ad_files, key=lambda f: f.stat().st_mtime)
        
        logger.info(f"Found prediction file: {pred_file.name}")
        return pred_file, 1.0
    
    def match_datasets(self, tissues: Optional[List[str]] = None) -> List[DatasetPair]:
        """
        Match ground truth and prediction files for specified tissues.
        
        Args:
            tissues: List of tissue names to process. If None, process all tissues.
        
        Returns:
            List of matched DatasetPair objects
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
                
                # Find matching prediction file
                result = self.find_prediction_file(tissue, dataset_id)
                
                if result is None:
                    logger.warning(
                        f"No prediction file found for {tissue}/{dataset_id}. Skipping."
                    )
                    skipped.append((tissue, gt_file.name, "No matching prediction file"))
                    continue
                
                pred_file, match_score = result
                
                pair = DatasetPair(
                    tissue=tissue,
                    dataset_name=dataset_id,
                    gt_file=gt_file,
                    pred_file=pred_file,
                    match_score=match_score
                )
                matched_pairs.append(pair)
                logger.info(f"✓ Matched: {tissue}/{dataset_id}")
        
        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info(f"MATCHING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Successfully matched: {len(matched_pairs)} datasets")
        logger.info(f"Skipped: {len(skipped)} datasets")
        
        if skipped:
            logger.info("\nSkipped datasets:")
            for tissue, filename, reason in skipped:
                logger.info(f"  - {tissue}/{filename}: {reason}")
        
        return matched_pairs
