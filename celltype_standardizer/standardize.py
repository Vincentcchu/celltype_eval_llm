"""
Workflow 1: Standalone standardization + mapping update (no evaluation).

Standardizes cell-type labels in AnnData to L3 taxonomy and updates 
the persistent mapping store.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict
import anndata as ad
import pandas as pd

from .mapping_store import MappingStore
from .llm_judge import LLMSemanticJudge

logger = logging.getLogger(__name__)


def standardize_h5ad_and_update_mapping(
    input_h5ad: Union[str, Path, ad.AnnData],
    obs_column: str,
    output_h5ad: Optional[Union[str, Path]] = None,
    output_obs_column: str = "cell_type_level3",
    mapping_store_path: Optional[str] = None,
    l3_vocab_path: Optional[str] = None,
    api_key: Optional[str] = None,
    llm_model: str = "gpt-4o-mini",
    skip_llm: bool = False,
) -> ad.AnnData:
    """
    Standardize cell-type labels to L3 taxonomy and update mapping store.
    
    This function:
    1. Extracts unique raw labels from specified obs column
    2. Checks existing mappings in the persistent store
    3. Uses LLM semantic judge for unmapped labels (unless skip_llm=True)
    4. Updates the mapping store with new mappings
    5. Writes standardized L3 labels to output column
    6. Optionally saves a new .h5ad file
    
    Args:
        input_h5ad: Path to input .h5ad file or AnnData object
        obs_column: Column name in adata.obs containing raw cell-type labels
        output_h5ad: Optional path to save standardized .h5ad. If None, doesn't save.
        output_obs_column: Column name for standardized L3 labels (default: "cell_type_level3")
        mapping_store_path: Path to mapping store JSON file (uses default if None)
        l3_vocab_path: Path to L3 vocabulary JSON file (uses default if None)
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
        llm_model: OpenAI model to use (default: "gpt-4o-mini")
        skip_llm: If True, only apply existing mappings without calling LLM for new labels
        
    Returns:
        AnnData object with standardized labels in output_obs_column
        
    Raises:
        ValueError: If obs_column doesn't exist in the AnnData object
    """
    logger.info("=== Starting standardization workflow ===")
    
    # Load AnnData if path provided
    if isinstance(input_h5ad, (str, Path)):
        logger.info(f"Loading AnnData from {input_h5ad}")
        adata = ad.read_h5ad(input_h5ad)
    else:
        adata = input_h5ad.copy()
    
    # Validate obs_column exists
    if obs_column not in adata.obs.columns:
        raise ValueError(
            f"Column '{obs_column}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    
    # Extract unique raw labels
    raw_labels = set(adata.obs[obs_column].astype(str).unique())
    logger.info(f"Found {len(raw_labels)} unique raw labels in column '{obs_column}'")
    
    # Initialize mapping store
    mapping_store = MappingStore(mapping_store_path)
    logger.info(f"Mapping store stats: {mapping_store.get_stats()}")
    
    # Check which labels need mapping
    unmapped_labels = mapping_store.get_unmapped_labels(raw_labels)
    logger.info(f"Found {len(unmapped_labels)} labels without existing mappings")
    
    # Get new mappings using LLM if needed
    new_mappings = {}
    if unmapped_labels and not skip_llm:
        logger.info(f"Calling LLM to map {len(unmapped_labels)} new labels")
        judge = LLMSemanticJudge(
            api_key=api_key,
            model=llm_model,
            vocab_file=l3_vocab_path
        )
        
        # Map each unmapped label
        for raw_label in sorted(unmapped_labels):
            result = judge.map_label(raw_label)
            new_mappings[raw_label] = result["selected_label"]
            logger.info(
                f"  '{raw_label}' -> '{result['selected_label']}' "
                f"(confidence: {result.get('confidence', 'N/A')})"
            )
        
        # Add new mappings to store
        mapping_store.add_mappings_batch(new_mappings)
        logger.info(f"Added {len(new_mappings)} new mappings to persistent store")
    elif unmapped_labels and skip_llm:
        logger.warning(
            f"Skipping LLM mapping for {len(unmapped_labels)} unmapped labels. "
            f"These will be marked as 'UNMAPPED'."
        )
    
    # Apply standardization: get all mappings and create standardized column
    all_mappings = mapping_store.get_all_mappings()
    
    def map_label(raw_label: str) -> str:
        """Map a single label with fallback."""
        return all_mappings.get(str(raw_label), "UNMAPPED")
    
    adata.obs[output_obs_column] = adata.obs[obs_column].astype(str).apply(map_label)
    
    # Report statistics
    standardized_labels = set(adata.obs[output_obs_column].unique())
    unmapped_count = (adata.obs[output_obs_column] == "UNMAPPED").sum()
    
    logger.info(f"Standardization complete:")
    logger.info(f"  - Standardized to {len(standardized_labels)} unique L3 labels")
    logger.info(f"  - {unmapped_count} cells remain unmapped")
    
    if unmapped_count > 0:
        unmapped_labels_list = adata.obs[adata.obs[output_obs_column] == "UNMAPPED"][obs_column].unique()
        logger.warning(f"  - Unmapped raw labels: {list(unmapped_labels_list)}")
    
    # Save output file if requested
    if output_h5ad:
        output_path = Path(output_h5ad)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(output_path)
        logger.info(f"Saved standardized AnnData to {output_path}")
    
    logger.info("=== Standardization workflow complete ===")
    
    return adata


def get_label_coverage_report(
    input_h5ad: Union[str, Path, ad.AnnData],
    obs_column: str,
    mapping_store_path: Optional[str] = None,
) -> Dict:
    """
    Generate a coverage report showing which labels are already mapped.
    
    Args:
        input_h5ad: Path to .h5ad file or AnnData object
        obs_column: Column name containing cell-type labels
        mapping_store_path: Path to mapping store (uses default if None)
        
    Returns:
        Dictionary with coverage statistics and lists of mapped/unmapped labels
    """
    # Load AnnData if path provided
    if isinstance(input_h5ad, (str, Path)):
        adata = ad.read_h5ad(input_h5ad)
    else:
        adata = input_h5ad
    
    # Extract unique labels
    raw_labels = set(adata.obs[obs_column].astype(str).unique())
    
    # Check mapping store
    mapping_store = MappingStore(mapping_store_path)
    unmapped = mapping_store.get_unmapped_labels(raw_labels)
    mapped = raw_labels - unmapped
    
    return {
        "total_unique_labels": len(raw_labels),
        "mapped_count": len(mapped),
        "unmapped_count": len(unmapped),
        "coverage_percent": (len(mapped) / len(raw_labels) * 100) if raw_labels else 0.0,
        "mapped_labels": sorted(mapped),
        "unmapped_labels": sorted(unmapped),
        "store_stats": mapping_store.get_stats()
    }
