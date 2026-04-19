# Bayes-HDC Examples

Runnable examples demonstrating Bayes-HDC capabilities.

## Examples

### Basic Operations

Core HDC operations: binding, bundling, permutation, similarity, and model comparison.

```bash
python examples/basic_operations.py
```

### Classification Pipeline

End-to-end classification with synthetic data using RandomEncoder and CentroidClassifier.

```bash
python examples/classification_simple.py
```

### Kanerva's "Dollar of Mexico"

Structured knowledge representation and analogical reasoning using role-filler binding.

```bash
python examples/kanerva_example.py
```

## Requirements

```bash
pip install -e .            # core library
pip install -e ".[examples]"  # optional: matplotlib, scikit-learn
```
