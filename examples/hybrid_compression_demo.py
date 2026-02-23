#!/usr/bin/env python3
"""
Hybrid Compression Showcase
Demonstrates the multi-language compression engine with AI-generated methods.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puffinzip_ai.hybrid_compression_engine import get_hybrid_engine
from puffinzip_ai.compression_method_registry import get_registry


def showcase_methods():
    """Showcase available compression methods"""
    engine = get_hybrid_engine()
    
    print("\n" + "="*70)
    print("🎯 PUFFIN ZIP AI - HYBRID COMPRESSION ENGINE")
    print("="*70)
    
    methods = engine.list_available_methods()
    print("\n📚 Available Compression Methods:")
    for name, desc in methods.items():
        print(f"  {name}")
        print(f"     → {desc}")
    
    print(f"\n  Total Methods: {len(methods)}")


def test_methods():
    """Test all methods with sample data"""
    engine = get_hybrid_engine()
    
    test_data = "AAAAAABBBBCCCCDDDDDEEEEEFFFFFFFFFF" * 3
    print(f"\n📊 Testing with sample data ({len(test_data)} bytes):")
    print(f"   Original: {test_data[:50]}...")
    
    print("\n🧪 Test Results:")
    print(f"{'Method':<20} {'Ratio':<10} {'Comp(ms)':<12} {'Decomp(ms)':<12} {'Valid':<8}")
    print("-" * 70)
    
    for method_name in engine.registry.list_methods():
        metrics = engine.test_method(method_name, test_data)
        if metrics:
            print(
                f"{method_name:<20} {metrics['compression_ratio']:<10.2%} "
                f"{metrics['compress_time_ms']:<12.3f} "
                f"{metrics['decompress_time_ms']:<12.3f} "
                f"{'✓' if metrics['success'] else '✗':<8}"
            )


def generate_novelty():
    """Generate and test novel compression methods"""
    engine = get_hybrid_engine()
    
    print("\n✨ Generating Novel Compression Methods...")
    print("-" * 70)
    
    for i in range(3):
        method = engine.discover_novelty_method(f"novel_{i+1}")
        print(f"\n  Generated: {method.name}")
        print(f"    Description: {method.description}")
        print(f"    Patterns: {method.metadata.get('patterns', [])}")


def compare_languages():
    """Compare compression across different language implementations"""
    engine = get_hybrid_engine()
    
    print("\n🌐 Compression Methods by Implementation Language:")
    print("-" * 70)
    
    registry = get_registry()
    
    for lang in ["PYTHON", "RUST", "HYBRID"]:
        methods = [{
            "name": name,
            "method": method
        } for name, method in registry.methods.items() 
        if method.language.value.upper() == lang]
        
        if methods:
            print(f"\n{lang}:")
            for item in methods:
                emoji = "⚡" if item["method"].language.value == "rust" else ("🔀" if item["method"].language.value == "hybrid" else "🐍")
                print(f"  {emoji} {item['name']}")
                if item["method"].metrics.test_count > 0:
                    print(f"     Avg Ratio: {item['method'].metrics.compression_ratio:.2%}")


def interactive_demo():
    """Interactive compression demo"""
    engine = get_hybrid_engine()
    
    print("\n🎮 Interactive Compression Demo")
    print("-" * 70)
    
    while True:
        print("\nOptions:")
        print("  1. Test a method")
        print("  2. Generate novelty method")
        print("  3. Evolve methods")
        print("  4. Show all methods")
        print("  5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            method_name = input("Enter method name (burst/delta_rle/frequency_codec): ").strip()
            test_data = input("Enter text to compress: ").strip()
            
            if test_data:
                compressed = engine.compress(test_data, method_name)
                decompressed = engine.decompress(compressed, method_name)
                
                print(f"\nResults:")
                print(f"  Original:     {len(test_data)} bytes")
                print(f"  Compressed:   {len(compressed)} bytes")
                print(f"  Ratio:        {len(compressed) / len(test_data):.2%}")
                print(f"  Valid:        {'✓' if test_data == decompressed else '✗'}")
        
        elif choice == "2":
            method = engine.discover_novelty_method()
            print(f"\n✨ Generated: {method.name}")
            print(f"   {method.description}")
        
        elif choice == "3":
            num = int(input("How many methods to evolve? ").strip() or "3")
            methods = engine.evolve_methods(num)
            print(f"\n✓ Evolved {len(methods)} new methods")
        
        elif choice == "4":
            showcase_methods()
        
        else:
            break
    
    print("\nGoodbye! 👋")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid Compression Engine Showcase")
    parser.add_argument("--demo", choices=["showcase", "test", "novelty", "languages", "interactive"],
                        default="showcase",
                        help="Demo to run")
    
    args = parser.parse_args()
    
    if args.demo == "showcase":
        showcase_methods()
    elif args.demo == "test":
        test_methods()
    elif args.demo == "novelty":
        generate_novelty()
    elif args.demo == "languages":
        compare_languages()
    elif args.demo == "interactive":
        interactive_demo()
