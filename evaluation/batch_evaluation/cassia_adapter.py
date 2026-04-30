"""
Adapter to convert CASSIA cluster-level predictions to cell-level predictions.

CASSIA outputs cluster-level predictions in CSV format. This module:
1. Loads cluster predictions from CSV (_FINAL_RESULTS.csv)
2. Loads cluster assignments from h5ad file (_clustered.h5ad)
3. Maps predictions to individual cells
4. Creates a temporary h5ad file compatible with the evaluation pipeline
"""

import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import anndata as ad
import numpy as np

logger = logging.getLogger(__name__)


class CASSIAAdapter:
    """Converts CASSIA cluster-level predictions to cell-level predictions."""
    
    def __init__(
        self,
        cluster_column: str = "cluster",
        prediction_column: str = "Predicted Main Cell Type"
    ):
        """
        Initialize the CASSIA adapter.
        
        Args:
            cluster_column: Column name in h5ad.obs containing cluster assignments
            prediction_column: Column name in CSV containing predictions
        """
        self.cluster_column = cluster_column
        self.prediction_column = prediction_column
    
    def create_cell_level_predictions(
        self,
        csv_path: Path,
        h5ad_path: Path,
        output_path: Optional[Path] = None,
        return_adata: bool = False
    ) -> tuple[Optional[Path], Optional[ad.AnnData]]:
        """
        Create an h5ad file with cell-level predictions from cluster-level predictions.
        
        Args:
            csv_path: Path to CASSIA CSV file with cluster predictions
            h5ad_path: Path to clustered h5ad file
            output_path: Path to save output h5ad. If None, uses temp file.
            return_adata: If True, return AnnData object instead of saving
        
        Returns:
            Tuple of (path, adata). If return_adata=True, returns (None, adata).
            Otherwise returns (path, None).
        """
        logger.info("Loading CASSIA cluster predictions...")
        logger.info(f"  CSV: {csv_path}")
        logger.info(f"  H5AD: {h5ad_path}")
        
        # Load cluster predictions from CSV
        pred_df = pd.read_csv(csv_path)
        
        # Validate required columns
        if 'True Cell Type' not in pred_df.columns:
            raise ValueError(f"CSV missing 'True Cell Type' column: {csv_path}")
        if self.prediction_column not in pred_df.columns:
            raise ValueError(f"CSV missing '{self.prediction_column}' column: {csv_path}")
        
        # Create cluster -> prediction mapping
        # The 'True Cell Type' column contains cluster IDs (0, 1, 2, ...)
        cluster_to_pred = dict(zip(
            pred_df['True Cell Type'].astype(str),
            pred_df[self.prediction_column]
        ))
        
        logger.info(f"Loaded predictions for {len(cluster_to_pred)} clusters")
        
        # Load clustered h5ad file (just obs for efficiency)
        logger.info("Loading cluster assignments from h5ad...")
        
        try:
            # Try to load just obs first for efficiency
            adata = ad.read_h5ad(h5ad_path, backed='r')
            obs_df = adata.obs.copy()
            
            # Get the full adata for creating output
            adata = ad.read_h5ad(h5ad_path)
        except Exception as e:
            logger.warning(f"Could not use backed mode, loading full file: {e}")
            adata = ad.read_h5ad(h5ad_path)
            obs_df = adata.obs.copy()
        
        logger.info(f"Loaded h5ad with {adata.n_obs} cells")
        
        # Validate cluster column exists
        if self.cluster_column not in obs_df.columns:
            raise ValueError(
                f"H5AD missing '{self.cluster_column}' column. "
                f"Available columns: {list(obs_df.columns)}"
            )
        
        # Map cluster assignments to predictions
        logger.info("Mapping cluster predictions to individual cells...")
        
        # Convert cluster IDs to strings for matching
        cluster_ids = obs_df[self.cluster_column].astype(str)
        
        # Map to predictions
        cell_predictions = cluster_ids.map(cluster_to_pred)
        
        # Check for unmapped cells
        unmapped = cell_predictions.isna().sum()
        if unmapped > 0:
            logger.warning(
                f"{unmapped} cells have no cluster prediction "
                f"({unmapped/len(cell_predictions)*100:.1f}%)"
            )
            # Fill with a default value
            cell_predictions = cell_predictions.fillna("Unknown")
        
        logger.info(f"Successfully mapped predictions to {len(cell_predictions)} cells")
        logger.info(f"Unique predictions: {cell_predictions.nunique()}")
        
        # Add predictions to adata
        adata.obs['predicted_cell_type'] = cell_predictions.values
        
        # If return_adata is True, return the AnnData object directly
        if return_adata:
            logger.info("✓ Cell-level predictions created in memory")
            return None, adata
        
        # Determine output path
        if output_path is None:
            # Create temp file in same directory as input
            output_path = h5ad_path.parent / f"{h5ad_path.stem}_with_predictions.h5ad"
        else:
            output_path = Path(output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save output
        logger.info(f"Saving cell-level predictions to: {output_path}")
        try:
            adata.write_h5ad(output_path)
            logger.info("✓ Cell-level prediction file created successfully")
            return output_path, None
        except OSError as e:
            if "Disk quota exceeded" in str(e) or "quota" in str(e).lower():
                logger.warning(f"Disk quota exceeded, returning in-memory object instead")
                return None, adata
            else:
                raise
    
    def get_cluster_statistics(self, csv_path: Path, h5ad_path: Path) -> dict:
        """
        Get statistics about cluster-level predictions.
        
        Args:
            csv_path: Path to CASSIA CSV file
            h5ad_path: Path to clustered h5ad file
        
        Returns:
            Dictionary with statistics
        """
        pred_df = pd.read_csv(csv_path)
        adata = ad.read_h5ad(h5ad_path, backed='r')
        
        cluster_counts = adata.obs[self.cluster_column].value_counts()
        
        stats = {
            'num_clusters': len(pred_df),
            'num_cells': adata.n_obs,
            'avg_cells_per_cluster': adata.n_obs / len(pred_df),
            'min_cluster_size': cluster_counts.min(),
            'max_cluster_size': cluster_counts.max(),
            'unique_predictions': pred_df[self.prediction_column].nunique()
        }
        
        return stats
