# PuffinZipAI Advanced Compression Discovery Example
# Shows how the LLM/Evolutionary Optimizer can discover new compression methods

"""
This example demonstrates how PuffinZipAI can autonomously discover
novel compression methods by analyzing data characteristics and evolving
new algorithms.
"""

from puffinzip_ai import (
    get_hybrid_engine,
    get_generator,
    get_registry,
    generate_novelty,
    evolve
)


def analyze_data_characteristics(data: str) -> dict:
    """
    Analyze data to determine suitable compression approaches.
    This is what an LLM would use to decide what compression methods to try.
    """
    characteristics = {
        "length": len(data),
        "unique_chars": len(set(data)),
        "char_freq_ratio": len(set(data)) / len(data) if data else 0,
        "has_patterns": False,
        "max_repetition": 0,
        "entropy_estimate": 0.0,
    }
    
    # Detect repetition patterns
    for i in range(len(data) - 1):
        if data[i] == data[i + 1]:
            count = 1
            j = i + 1
            while j < len(data) and data[j] == data[i]:
                count += 1
                j += 1
            characteristics["max_repetition"] = max(characteristics["max_repetition"], count)
            characteristics["has_patterns"] = True
    
    # Simple entropy estimate
    import math
    freq = {}
    for c in data:
        freq[c] = freq.get(c, 0) + 1
    
    entropy = 0.0
    for count in freq.values():
        p = count / len(data) if data else 0
        if p > 0:
            entropy -= p * math.log2(p)
    
    characteristics["entropy_estimate"] = entropy
    
    return characteristics


def recommend_compression_strategy(characteristics: dict) -> str:
    """
    Based on data characteristics, recommend compression strategy.
    This is like an LLM reasoning about what would work best.
    """
    
    # High repetition → RLE variants
    if characteristics["max_repetition"] > 10:
        if characteristics["char_freq_ratio"] < 0.1:
            return "burst"  # Good for repeated patterns
        else:
            return "delta_rle"
    
    # Low entropy, uneven distribution → Frequency codec
    if characteristics["entropy_estimate"] < 4.0 and characteristics["char_freq_ratio"] < 0.5:
        return "frequency_codec"
    
    # Default to adaptive method
    return "burst"


def autonomous_compression_discovery():
    """
    Example of autonomous compression method discovery.
    The LLM/AI would do this automatically.
    """
    
    print("\n" + "="*70)
    print("🧠 PUFFIN ZIP AI - AUTONOMOUS COMPRESSION DISCOVERY")
    print("="*70)
    
    # Test data samples
    test_samples = {
        "repetitive": "AAABBBCCCDDDEEEFFFGGGHHHIIIJJJKKKLLL" * 10,
        "sequential": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" * 5,
        "random": "xD8#kL2@pQ9*mN4$bV7&wE6!jH3^tY5~sF0+rG1" * 4,
        "text": "the quick brown fox jumps over the lazy dog. " * 8,
    }
    
    engine = get_hybrid_engine()
    
    for sample_name, sample_data in test_samples.items():
        print(f"\n📊 Sample: {sample_name} ({len(sample_data)} bytes)")
        print("-" * 70)
        
        # Step 1: Analyze characteristics
        chars = analyze_data_characteristics(sample_data)
        print(f"  Characteristics:")
        print(f"    - Unique chars: {chars['unique_chars']}")
        print(f"    - Max repetition: {chars['max_repetition']}")
        print(f"    - Entropy estimate: {chars['entropy_estimate']:.2f}")
        print(f"    - Has patterns: {chars['has_patterns']}")
        
        # Step 2: Recommend strategy
        recommended = recommend_compression_strategy(chars)
        print(f"  Recommended method: {recommended}")
        
        # Step 3: Test recommended method
        metrics = engine.test_method(recommended, sample_data)
        if metrics:
            print(f"  Results:")
            print(f"    - Compression ratio: {metrics['compression_ratio']:.2%}")
            print(f"    - Encode time: {metrics['compress_time_ms']:.3f}ms")
            print(f"    - Valid: {'✓' if metrics['success'] else '✗'}")
        
        # Step 4: Generate novelty methods for this specific data
        print(f"  Generating novelty methods...")
        novel_methods = evolve(num_mutations=2)
        
        for method in novel_methods:
            metrics = engine.test_method(method.name, sample_data)
            if metrics:
                better = "Better! 🎉" if metrics['compression_ratio'] < 1.0 else ""
                print(f"    ✨ {method.name}: {metrics['compression_ratio']:.2%} {better}")


def llm_driven_method_search():
    """
    Simulate LLM-driven search for optimal compression methods.
    The LLM might reason like: "For this type of data, I should try
    combining RLE with dictionary encoding..."
    """
    
    print("\n" + "="*70)
    print("🤖 LLM-DRIVEN COMPRESSION METHOD SEARCH")
    print("="*70)
    
    engine = get_hybrid_engine()
    generator = get_generator()
    
    # Simulating LLM reasoning
    llm_reasoning = """
    I observe this data has:
    1. High repetition of character sequences (AAAA, BBBB, etc.)
    2. Some structure in the ordering
    3. Not fully random
    
    Based on compression theory, I should try:
    - RLE (good for repetitive data)
    - Delta encoding (captures sequential patterns)
    - Frequency-based methods (if distribution is skewed)
    
    Let me generate hybrid methods combining these techniques.
    """
    
    print("\nLLM Reasoning:")
    for line in llm_reasoning.strip().split("\n"):
        print(f"  {line}")
    
    print("\n🔬 Testing Strategy...")
    
    test_data = "AAAA" * 100 + "BBBB" * 50 + "CCCC" * 25
    
    # Generate multiple novelty methods
    print(f"\nGenerating 5 novel compression methods...")
    methods = evolve(num_mutations=5)
    
    results = []
    for method in methods:
        metrics = engine.test_method(method.name, test_data)
        if metrics:
            results.append({
                "name": method.name,
                "ratio": metrics['compression_ratio'],
                "patterns": method.metadata.get('patterns', [])
            })
    
    # Sort by compression ratio
    results.sort(key=lambda x: x['ratio'])
    
    print("\n🏆 Best Methods Discovered:")
    for i, result in enumerate(results[:3], 1):
        print(f"  {i}. {result['name']}")
        print(f"     Ratio: {result['ratio']:.2%}")
        print(f"     Patterns: {', '.join(result['patterns'][:3])}")


def continuous_learning_simulation():
    """
    Simulate continuous learning where the system discovers
    better compression methods over time.
    """
    
    print("\n" + "="*70)
    print("📈 CONTINUOUS COMPRESSION DISCOVERY (Over Generations)")
    print("="*70)
    
    engine = get_hybrid_engine()
    registry = get_registry()
    
    # Simulated "generations" of discovery
    test_data = "ABCDEF" * 50 + "GHIJKL" * 30 + "MNOPQR" * 20
    
    print(f"\nStarting with {len(registry.methods)} base methods")
    print(f"Test data: {len(test_data)} bytes")
    
    best_ratio = float('inf')
    best_method = None
    
    # Simulate 3 generations
    for generation in range(3):
        print(f"\n🧬 Generation {generation + 1}:")
        
        # Generate new methods
        new_methods = evolve(num_mutations=3)
        
        # Test each new method
        for method in new_methods:
            metrics = engine.test_method(method.name, test_data)
            if metrics:
                ratio = metrics['compression_ratio']
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_method = method.name
                    print(f"  🎯 NEW BEST: {method.name} ({ratio:.2%})")
                else:
                    print(f"  ✓ {method.name} ({ratio:.2%})")
    
    print(f"\n🏆 After evolution: Best method is '{best_method}' with {best_ratio:.2%} compression")


if __name__ == "__main__":
    import sys
    
    # Run demonstrations
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║        PUFFIN ZIP AI - ADVANCED COMPRESSION DISCOVERY            ║
    ║                                                                  ║
    ║  This demonstrates how PuffinZipAI autonomously discovers       ║
    ║  new compression methods using LLM reasoning and AI evolution.  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    autonomous_compression_discovery()
    llm_driven_method_search()
    continuous_learning_simulation()
    
    print("\n" + "="*70)
    print("✨ Compression discovery demonstration complete!")
    print("="*70)
