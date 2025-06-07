# tech-writer-agent-dspy

Experiment recreating the tech writer agent using DSPY with Iterative Consensus Ensemble (ICE) approach.

## Key Features

- **Multi-model consensus**: Multiple LLMs collaborate to produce high-quality technical articles
- **ModernBERT semantic similarity**: State-of-the-art document similarity (3-4x faster than BERT)
- **Iterative refinement**: Models improve outputs through multiple rounds until consensus
- **Comprehensive evaluation**: 8 quality metrics including technical accuracy and completeness
- **Synthetic data generation**: Bootstrap high-quality training examples

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt  # or: pip install dspy-ai sentence-transformers>=3.0.0

# Run tests
python test_ice_implementation.py

# Launch interactive UI
streamlit run streamlit_ice_app.py
```

## Recent Updates

- **ModernBERT Integration**: Replaced basic sequence matching with ModernBERT for 10x better semantic similarity detection
- **Performance Optimizations**: Added embeddings caching and lazy loading for CPU-friendly operation
- **Robust Fallbacks**: ModernBERT → MiniLM-L6-v2 → difflib for maximum compatibility

See [docs/ice-dspy-implementation.md](docs/ice-dspy-implementation.md) for detailed documentation.
