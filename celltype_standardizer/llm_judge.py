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
    
    def _create_retry_prompt(self, raw_label: str, invalid_label: str) -> str:
        """Create a stronger prompt for retry after invalid response."""
        prompt = f"""CRITICAL: You MUST choose a label from the EXACT list provided below. Do not make up new labels.

Raw cell-type label: "{raw_label}"

You previously returned "{invalid_label}" which is NOT in the allowed list.

ALLOWED L3 LABELS (you MUST choose EXACTLY one of these, copy it character-by-character):
{chr(10).join(f"- {label}" for label in self.l3_labels)}

Instructions:
1. Look at the list above carefully
2. Select the single BEST MATCH from that exact list
3. Copy the label EXACTLY as written above (case-sensitive)
4. Return valid JSON with "selected_label", "confidence", and "rationale"

Do not create variations or new labels - ONLY use labels from the list above.
"""
        return prompt
    
    def map_label(self, raw_label: str, max_retries: int = 2) -> Dict[str, any]:
        """
        Map a raw label to an L3 label using LLM semantic matching.
        
        Args:
            raw_label: The raw cell-type label to map.
            max_retries: Maximum number of attempts (default: 2)
            
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
        
        # Try LLM with retries
        last_invalid_label = None
        for attempt in range(1, max_retries + 1):
            try:
                # Use regular prompt on first attempt, retry prompt on subsequent attempts
                if attempt == 1:
                    prompt = self._create_prompt(raw_label)
                else:
                    prompt = self._create_retry_prompt(raw_label, last_invalid_label)
                    logger.warning(f"Retry attempt {attempt}/{max_retries} for '{raw_label}'")
                
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
                if selected in self.l3_labels:
                    # Success!
                    logger.info(
                        f"LLM mapping (attempt {attempt}): '{raw_label}' -> '{selected}' "
                        f"(confidence: {result.get('confidence', 'N/A')})"
                    )
                    return result
                else:
                    # Invalid label - prepare for retry
                    last_invalid_label = selected
                    logger.error(
                        f"Attempt {attempt}/{max_retries}: LLM returned invalid label '{selected}' "
                        f"for '{raw_label}'. Valid labels: {self.l3_labels}"
                    )
                    
                    # If this was the last attempt, fall through to final fallback
                    if attempt == max_retries:
                        break
                    
            except json.JSONDecodeError as e:
                logger.error(f"Attempt {attempt}/{max_retries}: JSON parse error for '{raw_label}': {e}")
                if attempt == max_retries:
                    break
                    
            except Exception as e:
                logger.error(f"Attempt {attempt}/{max_retries}: Error calling LLM for '{raw_label}': {e}")
                if attempt == max_retries:
                    break
        
        # If we get here, all retries failed - use simple fallback
        logger.error(
            f"All {max_retries} attempts failed for '{raw_label}'. "
            f"Using 'Other/unknown' as last resort."
        )
        return {
            "selected_label": "Other/unknown",
            "confidence": 0.0,
            "rationale": f"Fallback after {max_retries} failed LLM attempts"
        }
    
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
