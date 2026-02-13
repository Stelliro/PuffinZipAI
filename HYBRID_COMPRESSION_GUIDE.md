# HYBRID COMPRESSION ENGINE ARCHITECTURE

## Overview

PuffinZipAI now features a revolutionary **Hybrid Compression Engine** that:

1. **Supports Multiple Languages**: Python, Rust, C/C++, and CUDA implementations
2. **Discovers Novel Methods**: AI-generated compression algorithms evolved through patterns
3. **Dynamic Method Registry**: Add new methods at runtime without code changes
4. **Multi-Strategy Approach**: Combine different compression techniques
5. **Performance Tracking**: Automatic metrics collection and optimization

## Architecture Components

### 1. compression_method_registry.py
**Central registry for all compression methods**

- `CompressionMethod`: Descriptor class for any compression method
- `CompressionLanguage`: Enum supporting PYTHON, RUST, CPP, CUDA, HYBRID
- `CompressionRegistry`: Global registry with metrics tracking
- `CompressionMetric`: Automatic performance metric collection

```python
from puffinzip_ai.compression_method_registry import CompressionMethod, CompressionLanguage

method = CompressionMethod(
    name="my_method",
    language=CompressionLanguage.PYTHON,
    compress_fn=my_compress,
    decompress_fn=my_decompress,
    is_novelty=False
)
```

### 2. novel_compression_generator.py
**AI-Powered Compression Method Discovery**

Uses pattern recognition to generate novel compression methods:

#### Base Techniques (10 pre-defined patterns):
- RLE (Run-Length Encoding) - adaptive thresholds
- Dictionary-based substitution
- Frequency sorting
- Delta encoding
- Entropy coding
- Bidirectional RLE
- Context modeling
- Burrows-Wheeler transforms
- LZ77 variants
- Arithmetic coding

#### Method Generation:
```python
from puffinzip_ai.novel_compression_generator import get_generator

generator = get_generator()

# Generate random novel method
method = generator.generate_novelty_method("my_novel_method")

# Evolve multiple new methods
methods = generator.evolve_methods(num_mutations=5)
```

Generated methods combine multiple patterns:
- *novelty_v1*: RLE + Dictionary + Delta Encoding
- *novelty_v2*: Frequency Sort + Entropy Coding
- *evolved_gen0*: Dynamic pattern combinations

### 3. rust_compression_interface.py
**High-Performance Rust Integration**

Provides Python interface to Rust implementations:

#### Built-in Rust Method: BURST
**Bidirectional RLE with Entropy Suppression and Repeated Tuple Recognition**

Novel multi-pass approach:
1. **Forward RLE**: Compress with forward run-length encoding
2. **Entropy Analysis**: Skip incompressible sections
3. **Backward RLE**: Apply RLE to reversed data
4. **Tuple Recognition**: Identify and encode repeated patterns

```python
from puffinzip_ai.rust_compression_interface import get_rust_interface

rust = get_rust_interface()
compressed = rust.compress_burst(data, method="burst_rle")
decompressed = rust.decompress_burst(compressed)
```

### 4. hybrid_compression_engine.py
**Unified Compression Interface**

Orchestrates all compression methods and provides simple API:

```python
from puffinzip_ai.hybrid_compression_engine import get_hybrid_engine

engine = get_hybrid_engine()

# Use any method
compressed = engine.compress(text, method="burst")
decompressed = engine.decompress(compressed, method="burst")

# Discover new methods
novel_method = engine.discover_novelty_method()

# Evolve compression methods
methods = engine.evolve_methods(num_mutations=10)

# Get best method for a metric
best = engine.get_best_method(metric="compression_ratio")
```

## Built-in Methods

### 1. BURST (Rust/Python)
- **Type**: Bidirectional RLE + Entropy aware
- **Best For**: Text with patterns and repetition
- **Language**: Hybrid (CUDA-accelerated in Rust)

### 2. Delta + RLE (Python)
- **Type**: Delta encoding + Run-length compression
- **Best For**: Sequential or numeric-like text
- **Language**: Python

### 3. Frequency Codec (Python)
- **Type**: Character remapping by frequency
- **Best For**: Non-uniform character distributions
- **Language**: Python

## Method Discovery Process

The AI can generate novel compression methods through:

### 1. Pattern Combination
```
Combines N random patterns from the 10 base techniques
Example: [RLE, Dictionary, Delta] → novelty_v3
```

### 2. Evolutionary Mutation
```
Modify parameter values and pattern selections
Automatic fitness estimation based on test data
```

### 3. Hybrid Generation
```
Python + Rust fusion for optimal performance
GPU acceleration where beneficial
```

## Usage Examples

### Basic Compression
```python
from puffinzip_ai.hybrid_compression_engine import get_hybrid_engine

engine = get_hybrid_engine()

text = "AAABBBCCCDDD" * 1000
compressed = engine.compress(text, method="burst")
original = engine.decompress(compressed, method="burst")
```

### Discover Novel Methods
```python
# Generate a completely new compression method
novel = engine.discover_novelty_method("my_invention")

# Test it
metrics = engine.test_method("my_invention", test_data)
print(f"Compression ratio: {metrics['compression_ratio']:.2%}")
```

### Evolve Methods for Specific Data
```python
# Generate 5 novel methods optimized for your data
methods = engine.evolve_methods(num_mutations=5)

# Test each one
for method in methods:
    metrics = engine.test_method(method.name, your_data)
    print(f"{method.name}: {metrics['compression_ratio']:.2%}")
```

### Benchmark All Methods
```python
test_data = "Your benchmark data here"
results = {}

for method_name in engine.registry.list_methods():
    results[method_name] = engine.test_method(method_name, test_data)

# Find best for your data
best = max(results.items(), key=lambda x: 1/x[1].get('compression_ratio', 1))
print(f"Best for your data: {best[0]}")
```

## Integration with AI Evolution

The evolutionary optimizer can now:

1. **Discover Compression Methods**
```python
# During evolution, generate novel methods
method = generator.generate_novelty_method()

# Test fitness on benchmark
fitness = evaluate_method(method, benchmark_data)

# Keep top performers
population.append(method)
```

2. **Optimize for Specific Workloads**
```python
# Evolve methods specifically tuned to your data
for generation in range(generations):
    methods = generator.evolve_methods()
    
    for method in methods:
        fitness = engine.test_method(method.name, data)["compression_ratio"]
        if fitness > best_fitness:
            best_method = method
```

## Future Extensions

### Planned Language Support
- *Rust*: FFI bindings for high-performance core algorithms
- *C/C++*: Hand-optimized implementations for specific methods
- *CUDA*: GPU-accelerated compression on NVIDIA GPUs
- *OpenCL*: GPU acceleration on AMD/Intel

### Planned Methods
- Streaming compression for real-time data
- Adaptive methods that change based on data characteristics
- Context-aware compression for specific file types
- Machine learning-based pattern prediction

## Performance Tips

1. **For Text Data**: Use `burst` or `delta_rle`
2. **For Random Data**: Use `frequency_codec`
3. **For Large Files**: Let AI evolve a custom method
4. **For Speed**: Use Rust implementations when available
5. **For Exploration**: Generate novelty methods and evaluate

## Metrics Tracked

Each method automatically tracks:
- Compression ratio (target size / original size)
- Encode time (milliseconds)
- Decode time (milliseconds)
- Peak memory usage (MB)
- Success rate (valid round-trips)

Access via:
```python
method = engine.registry.get_method("burst")
print(f"Avg Ratio: {method.metrics.compression_ratio:.2%}")
print(f"Encode: {method.metrics.encode_time_ms:.2f}ms")
print(f"Tests: {method.metrics.test_count}")
```

## Demo

Run the showcase:
```bash
python hybrid_compression_demo.py --demo showcase
python hybrid_compression_demo.py --demo test
python hybrid_compression_demo.py --demo novelty
python hybrid_compression_demo.py --demo interactive
```
