"""
Persistent mapping store for raw cell-type labels to L3 standardized labels.
Thread-safe operations with file locking.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set
import fcntl
import logging

logger = logging.getLogger(__name__)


class MappingStore:
    """Manages persistent storage of raw label -> L3 label mappings."""
    
    def __init__(self, mapping_file: Optional[str] = None):
        """
        Initialize the mapping store.
        
        Args:
            mapping_file: Path to the mapping JSON file. If None, uses default location.
        """
        if mapping_file is None:
            # Default to mappings/label_mappings.json relative to project root
            project_root = Path(__file__).parent.parent
            mapping_file = project_root / "mappings" / "label_mappings.json"
        
        self.mapping_file = Path(mapping_file)
        self._ensure_mapping_file_exists()
    
    def _ensure_mapping_file_exists(self):
        """Create the mapping file if it doesn't exist."""
        if not self.mapping_file.exists():
            self.mapping_file.parent.mkdir(parents=True, exist_ok=True)
            initial_data = {
                "version": "1.0",
                "description": "Persistent mapping store: raw labels -> L3 standardized labels",
                "mappings": {},
                "metadata": {
                    "last_updated": None,
                    "total_mappings": 0
                }
            }
            with open(self.mapping_file, 'w') as f:
                json.dump(initial_data, f, indent=2)
    
    def _read_with_lock(self) -> Dict:
        """Read mapping file with file lock."""
        with open(self.mapping_file, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return data
    
    def _write_with_lock(self, data: Dict):
        """Write mapping file with file lock."""
        with open(self.mapping_file, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    def get_mapping(self, raw_label: str) -> Optional[str]:
        """
        Get the L3 mapping for a raw label.
        
        Args:
            raw_label: The raw cell-type label to look up.
            
        Returns:
            The L3 standardized label if found, None otherwise.
        """
        data = self._read_with_lock()
        return data["mappings"].get(raw_label)
    
    def get_all_mappings(self) -> Dict[str, str]:
        """
        Get all mappings.
        
        Returns:
            Dictionary of raw_label -> L3_label mappings.
        """
        data = self._read_with_lock()
        return data["mappings"].copy()
    
    def add_mapping(self, raw_label: str, l3_label: str):
        """
        Add a new mapping to the store.
        
        Args:
            raw_label: The raw cell-type label.
            l3_label: The L3 standardized label.
        """
        data = self._read_with_lock()
        
        if raw_label not in data["mappings"]:
            data["mappings"][raw_label] = l3_label
            data["metadata"]["total_mappings"] = len(data["mappings"])
            data["metadata"]["last_updated"] = datetime.now().isoformat()
            
            self._write_with_lock(data)
            logger.info(f"Added new mapping: '{raw_label}' -> '{l3_label}'")
        else:
            existing = data["mappings"][raw_label]
            if existing != l3_label:
                logger.warning(
                    f"Mapping conflict for '{raw_label}': "
                    f"existing='{existing}', new='{l3_label}'. Keeping existing."
                )
    
    def add_mappings_batch(self, mappings: Dict[str, str]):
        """
        Add multiple mappings at once (more efficient than repeated single adds).
        
        Args:
            mappings: Dictionary of raw_label -> L3_label mappings to add.
        """
        data = self._read_with_lock()
        
        new_count = 0
        for raw_label, l3_label in mappings.items():
            if raw_label not in data["mappings"]:
                data["mappings"][raw_label] = l3_label
                new_count += 1
                logger.info(f"Added new mapping: '{raw_label}' -> '{l3_label}'")
            else:
                existing = data["mappings"][raw_label]
                if existing != l3_label:
                    logger.warning(
                        f"Mapping conflict for '{raw_label}': "
                        f"existing='{existing}', new='{l3_label}'. Keeping existing."
                    )
        
        if new_count > 0:
            data["metadata"]["total_mappings"] = len(data["mappings"])
            data["metadata"]["last_updated"] = datetime.now().isoformat()
            self._write_with_lock(data)
            logger.info(f"Added {new_count} new mappings to store")
    
    def get_unmapped_labels(self, labels: Set[str]) -> Set[str]:
        """
        Find which labels from a set are not yet mapped.
        
        Args:
            labels: Set of raw labels to check.
            
        Returns:
            Set of labels that don't have mappings yet.
        """
        data = self._read_with_lock()
        existing_mappings = set(data["mappings"].keys())
        return labels - existing_mappings
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the mapping store.
        
        Returns:
            Dictionary with store statistics.
        """
        data = self._read_with_lock()
        return {
            "total_mappings": data["metadata"]["total_mappings"],
            "last_updated": data["metadata"]["last_updated"],
            "file_path": str(self.mapping_file)
        }
