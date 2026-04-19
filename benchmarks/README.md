# Bayes-HDC Performance Benchmarks

Reproducible performance benchmarks for core HDC operations and comparison with [TorchHD](https://github.com/hyperdimensional-computing/torchhd).

## Methodology

- **Dimensions**: 10,000 (standard HDC hypervector size)
- **Warmup**: 20 iterations (covers JIT compilation)
- **Trials**: 200 per operation
- **Device**: CPU (both libraries) for fair comparison
- **Report**: Mean ± Std (milliseconds)

## Running Benchmarks

### Bayes-HDC only

```bash
python benchmarks/benchmark_operations.py
```

### Bayes-HDC vs TorchHD comparison

```bash
pip install -e ".[benchmark]"
python benchmarks/benchmark_compare.py
```

Results are printed to stdout and saved to `benchmark_results.json`.

## Benchmarked Operations

| Operation | Description |
|-----------|-------------|
| MAP bind (2 HVs) | Element-wise multiply of two MAP hypervectors |
| MAP bundle (10 HVs) | Normalized sum of 10 MAP hypervectors |
| Cosine similarity | Cosine similarity between two vectors |
| RandomEncoder (100x20) | Encode 100 samples with 20 discrete features |

## Reproducibility

For comparable results:
1. Use the same hardware (CPU/GPU)
2. Close other heavy processes
3. Run multiple times; results vary by machine
