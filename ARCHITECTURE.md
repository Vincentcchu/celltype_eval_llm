# System Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                      │
├─────────────────────────────────┬───────────────────────────┤
│       Python API                │         CLI               │
│  - standardize_h5ad_...()       │  - standardize           │
│  - evaluate_h5ad()              │  - evaluate              │
│  - get_label_coverage_report()  │  - coverage              │
└─────────────────────────────────┴───────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│                    Workflow Layer                            │
├──────────────────────────────┬──────────────────────────────┤
│   Workflow 1: Standardize    │   Workflow 2: Evaluate       │
│   - Load AnnData             │   - Load pred & GT           │
│   - Extract labels           │   - Standardize both         │
│   - Map to L3                │   - Compute metrics          │
│   - Save results             │   - Generate report          │
└──────────────────────────────┴──────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│                    Core Services Layer                        │
├────────────────┬────────────────┬──────────────┬────────────┤
│ MappingStore   │ LLMSemanticJudge│ L3Vocabulary │  Metrics   │
│ - get_mapping()│ - map_label()   │ - get_labels()│ - accuracy │
│ - add_mapping()│ - exact_match() │              │ - F1       │
│ - get_stats()  │ - llm_call()    │              │ - confusion│
└────────────────┴────────────────┴──────────────┴────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│                    Data & External Services                   │
├────────────────┬────────────────┬──────────────┬────────────┤
│ Mapping Store  │ L3 Vocabulary  │  OpenAI API  │  AnnData   │
│ (JSON file)    │ (JSON file)    │  (GPT-4o)    │  (.h5ad)   │
└────────────────┴────────────────┴──────────────┴────────────┘
```

## Data Flow

### Workflow 1: Standardization

```
Input .h5ad
    │
    ↓
Extract unique labels from obs[column]
    │
    ↓
┌───Check MappingStore for existing mappings───┐
│                                               │
│  Found ────→ Use existing mapping            │
│                                               │
│  Not Found ───→ LLMSemanticJudge             │
│                      │                        │
│                      ↓                        │
│                 Exact match?                  │
│                 │         │                   │
│               Yes        No                   │
│                 │         │                   │
│                 │    Call OpenAI API          │
│                 │    (select from L3)         │
│                 │         │                   │
│                 ↓         ↓                   │
│            Return L3 label                    │
│                 │                             │
│                 ↓                             │
│            Add to MappingStore                │
└───────────────────────────────────────────────┘
    │
    ↓
Apply mappings: obs[column] → obs[output_column]
    │
    ↓
Optionally save standardized .h5ad
```

### Workflow 2: Evaluation

```
Load pred.h5ad + gt.h5ad
    │
    ↓
Standardize predictions to L3 (using Workflow 1 logic)
    │
    ↓
Standardize ground truth to L3 (using Workflow 1 logic)
    │
    ↓
Compute metrics on L3 labels
    │
    ├─→ Accuracy (sklearn.accuracy_score)
    ├─→ Macro F1 (sklearn.f1_score)
    ├─→ Weighted F1 (sklearn.f1_score)
    ├─→ Confusion Matrix (sklearn.confusion_matrix)
    └─→ Per-class metrics (sklearn.classification_report)
    │
    ↓
Generate JSON report
    │
    ↓
Return metrics dict + optionally save to file
```

## Component Interactions

### MappingStore (Thread-Safe Persistent Storage)

```
┌──────────────────────────────────────┐
│         MappingStore                 │
├──────────────────────────────────────┤
│  File: label_mappings.json           │
│  Format: {raw_label → L3_label}      │
│                                      │
│  Operations:                         │
│  - get_mapping(label)                │
│    • Read with shared lock           │
│    • Return L3 label or None         │
│                                      │
│  - add_mapping(raw, l3)              │
│    • Read with shared lock           │
│    • Write with exclusive lock       │
│    • Update metadata                 │
│                                      │
│  - get_unmapped_labels(labels)       │
│    • Set difference operation        │
│    • Returns labels needing mapping  │
└──────────────────────────────────────┘
```

### LLMSemanticJudge (Intelligent Mapping)

```
┌──────────────────────────────────────┐
│       LLMSemanticJudge               │
├──────────────────────────────────────┤
│  1. Exact Match (case-insensitive)   │
│     raw_label.lower() == l3.lower()  │
│     ↓                                │
│  2. LLM Semantic Matching            │
│     • Prompt with L3 vocabulary      │
│     • Structured JSON output         │
│     • Confidence scoring             │
│     ↓                                │
│  3. Validation                       │
│     • Verify output in L3 vocab      │
│     ↓                                │
│  4. Fallback (if LLM fails)          │
│     • Fuzzy substring matching       │
│     • Default to "Non neoplastic"    │
│     ↓                                │
│  Return: {                           │
│    "selected_label": L3_label,       │
│    "confidence": 0.0-1.0,            │
│    "rationale": "explanation"        │
│  }                                   │
└──────────────────────────────────────┘
```

## File Formats

### L3 Vocabulary (mappings/l3_vocabulary.json)

```json
{
  "version": "1.0",
  "description": "Level 3 cell-type taxonomy",
  "labels": [
    "Malignant",
    "Benign",
    "Non neoplastic",
    "Fibroblasts",
    "..."
  ],
  "parent_child_structure": {
    "note": "Empty by design"
  }
}
```

### Mapping Store (mappings/label_mappings.json)

```json
{
  "version": "1.0",
  "description": "Persistent raw→L3 mappings",
  "mappings": {
    "T cell": "Lymphoid cells",
    "endothelial cell": "Endothelial",
    "CAF": "Fibroblasts"
  },
  "metadata": {
    "last_updated": "2025-12-27T10:30:00.123Z",
    "total_mappings": 3
  }
}
```

### Evaluation Metrics (output)

```json
{
  "accuracy": 0.8765,
  "macro_f1": 0.8234,
  "weighted_f1": 0.8543,
  "confusion_matrix": {
    "labels": ["Fibroblasts", "Endothelial", "..."],
    "matrix": [
      [1000, 50, 10],
      [30, 950, 20],
      [...]
    ]
  },
  "per_class_metrics": {
    "Fibroblasts": {
      "precision": 0.92,
      "recall": 0.89,
      "f1_score": 0.90,
      "support": 1234
    }
  },
  "label_counts": {
    "predictions": {"Fibroblasts": 1100, "...": "..."},
    "ground_truth": {"Fibroblasts": 1234, "...": "..."}
  },
  "dataset_info": {
    "total_cells": 10000,
    "unique_pred_labels": 15,
    "standardized_pred_labels": 12,
    "unmapped_predictions": 0
  }
}
```

## Error Handling & Guardrails

```
┌─────────────────────────────────────────────────┐
│             Guardrails & Error Handling          │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. Input Validation                            │
│     ✓ Column exists in AnnData                  │
│     ✓ Cell counts match (pred vs GT)            │
│     ✓ File paths valid                          │
│                                                 │
│  2. Mapping Validation                          │
│     ✓ Exact match before LLM                    │
│     ✓ LLM output in L3 vocabulary               │
│     ✓ Fallback to fuzzy matching                │
│                                                 │
│  3. Concurrency Safety                          │
│     ✓ File locking (shared/exclusive)           │
│     ✓ Atomic read-modify-write                  │
│                                                 │
│  4. API Failures                                │
│     ✓ Catch OpenAI exceptions                   │
│     ✓ Fallback mapping strategies               │
│     ✓ Log all failures                          │
│                                                 │
│  5. Data Quality                                │
│     ✓ Report UNMAPPED labels                    │
│     ✓ Log confidence scores                     │
│     ✓ Track coverage statistics                 │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Scalability Considerations

### Current Design

- **Single machine**: File-based locking
- **Sequential LLM calls**: One label at a time
- **In-memory processing**: Full AnnData loaded

### Potential Optimizations

```
For large-scale deployment:

1. Distributed Mapping Store
   - Replace JSON with database (Redis, PostgreSQL)
   - Distributed locking (Redis locks, DB transactions)

2. Parallel LLM Calls
   - Batch API requests
   - Async processing with asyncio
   - Rate limiting / queuing

3. Chunked Processing
   - Process large .h5ad in chunks
   - Streaming evaluation
   - Lazy loading

4. Caching Layer
   - In-memory LRU cache for frequent mappings
   - Reduce file I/O
```

## Extension Points

### Adding New LLM Providers

```python
# llm_judge.py
class BaseLLMJudge(ABC):
    @abstractmethod
    def map_label(self, raw_label: str) -> Dict:
        pass

class OpenAIJudge(BaseLLMJudge):
    # Current implementation
    pass

class AnthropicJudge(BaseLLMJudge):
    # New provider
    pass

class LocalModelJudge(BaseLLMJudge):
    # Local LLM
    pass
```

### Adding New Storage Backends

```python
# mapping_store.py
class BaseMappingStore(ABC):
    @abstractmethod
    def get_mapping(self, raw_label: str) -> Optional[str]:
        pass
    
    @abstractmethod
    def add_mapping(self, raw_label: str, l3_label: str):
        pass

class JSONMappingStore(BaseMappingStore):
    # Current implementation
    pass

class DatabaseMappingStore(BaseMappingStore):
    # SQL backend
    pass
```

### Adding New Metrics

```python
# evaluate.py
def compute_custom_metrics(y_true, y_pred):
    """Add domain-specific metrics"""
    return {
        "matthews_corrcoef": ...,
        "balanced_accuracy": ...,
        "per_class_sensitivity": ...,
    }
```

---

This architecture provides:
- ✅ Modularity (easy to extend)
- ✅ Reusability (shared mapping system)
- ✅ Reliability (error handling, validation)
- ✅ Persistence (growing knowledge base)
- ✅ Scalability (clear optimization paths)
