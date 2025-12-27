"""
LLM-based semantic judge for mapping raw cell-type labels to L3 vocabulary.
Supports OpenAI API with structured output and fallback parsing.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import logging
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)


class L3Vocabulary:
    """Manages the L3 vocabulary list."""
    
    def __init__(self, vocab_file: Optional[str] = None):
        """
        Initialize L3 vocabulary.
        
        Args:
            vocab_file: Path to the L3 vocabulary JSON file. If None, uses default location.
        """
        if vocab_file is None:
            project_root = Path(__file__).parent.parent
            vocab_file = project_root / "mappings" / "l3_vocabulary.json"
        
        self.vocab_file = Path(vocab_file)
        self.labels = self._load_vocabulary()
    
    def _load_vocabulary(self) -> List[str]:
        """Load L3 vocabulary from JSON file."""
        with open(self.vocab_file, 'r') as f:
            data = json.load(f)
        return data["labels"]
    
    def get_labels(self) -> List[str]:
        """Get the list of L3 labels."""
        return self.labels.copy()


class LLMSemanticJudge:
    """LLM-based semantic judge for mapping raw labels to L3 vocabulary."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        vocab_file: Optional[str] = None
    ):
        """
        Initialize the semantic judge.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: OpenAI model to use for semantic matching.
            vocab_file: Path to L3 vocabulary file. If None, uses default.
        """
        self.model = model
        self.vocabulary = L3Vocabulary(vocab_file)
        self.l3_labels = self.vocabulary.get_labels()
        
        # Initialize OpenAI client
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                    "or pass api_key parameter."
                )
        
        self.client = OpenAI(api_key=api_key)
    
    def _create_prompt(self, raw_label: str) -> str:
        """Create the prompt for the LLM."""
        prompt = f"""You are a cell-type label standardization expert. Your task is to map a raw cell-type label to the most appropriate label from the Level 3 (L3) canonical taxonomy.

Raw cell-type label: "{raw_label}"

Allowed L3 labels (choose exactly one):
{chr(10).join(f"- {label}" for label in self.l3_labels)}

Instructions:
1. Select the single best matching L3 label based on semantic meaning
2. Consider biological relationships and cell-type hierarchy
3. If the raw label is already an L3 label (exact match), return it
4. For ambiguous cases, choose the most specific applicable L3 label
5. You MUST choose one label from the list above

Respond with a JSON object containing:
- "selected_label": the chosen L3 label (must be from the list above)
- "confidence": a value from 0.0 to 1.0 indicating match confidence
- "rationale": brief explanation (1-2 sentences) of why this mapping is appropriate

Example response format:
{{
  "selected_label": "Endothelial",
  "confidence": 0.95,
  "rationale": "The raw label refers to endothelial cells, which directly matches the L3 category."
}}
"""
        return prompt
    
    def map_label(self, raw_label: str) -> Dict[str, any]:
        """
        Map a raw label to an L3 label using LLM semantic matching.
        
        Args:
            raw_label: The raw cell-type label to map.
            
        Returns:
            Dictionary with:
                - selected_label: The chosen L3 label
                - confidence: Confidence score (0.0-1.0)
                - rationale: Explanation of the mapping
        """
        # First, check for exact match (case-insensitive)
        for l3_label in self.l3_labels:
            if raw_label.lower() == l3_label.lower():
                logger.info(f"Exact match found: '{raw_label}' -> '{l3_label}'")
                return {
                    "selected_label": l3_label,
                    "confidence": 1.0,
                    "rationale": "Exact match with L3 vocabulary"
                }
        
        # Use LLM for semantic matching
        prompt = self._create_prompt(raw_label)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cell-type taxonomy expert. Always respond with valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,  # Deterministic output
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            # Validate that selected label is in vocabulary
            selected = result.get("selected_label")
            if selected not in self.l3_labels:
                logger.error(
                    f"LLM returned invalid label '{selected}' for '{raw_label}'. "
                    f"Using fallback."
                )
                # Fallback to closest match
                selected = self._fuzzy_match_fallback(raw_label)
                result["selected_label"] = selected
                result["confidence"] = 0.5
                result["rationale"] = f"Fallback mapping after LLM error"
            
            logger.info(
                f"LLM mapping: '{raw_label}' -> '{result['selected_label']}' "
                f"(confidence: {result.get('confidence', 'N/A')})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error calling LLM for label '{raw_label}': {e}")
            # Fallback to simple fuzzy matching
            selected = self._fuzzy_match_fallback(raw_label)
            return {
                "selected_label": selected,
                "confidence": 0.3,
                "rationale": f"Fallback mapping due to LLM error: {str(e)}"
            }
    
    def _fuzzy_match_fallback(self, raw_label: str) -> str:
        """
        Simple fallback fuzzy matching if LLM fails.
        Checks for substring matches (case-insensitive).
        """
        raw_lower = raw_label.lower()
        
        # Try to find L3 label as substring in raw label
        for l3_label in self.l3_labels:
            if l3_label.lower() in raw_lower:
                return l3_label
        
        # Try to find raw label as substring in L3 labels
        for l3_label in self.l3_labels:
            if raw_lower in l3_label.lower():
                return l3_label
        
        # Default to most generic label
        logger.warning(f"No good match found for '{raw_label}', defaulting to 'Non neoplastic'")
        return "Non neoplastic"
    
    def map_labels_batch(self, raw_labels: List[str]) -> Dict[str, Dict]:
        """
        Map multiple labels in batch.
        
        Args:
            raw_labels: List of raw labels to map.
            
        Returns:
            Dictionary mapping raw_label -> result dict
        """
        results = {}
        for raw_label in raw_labels:
            results[raw_label] = self.map_label(raw_label)
        return results
