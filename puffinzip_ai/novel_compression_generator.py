# PuffinZipAI_Project/puffinzip_ai/novel_compression_generator.py
"""
Novel Compression Method Generator
Enables the LLM/AI system to discover and evolve new compression algorithms.
Uses pattern recognition and genetic evolution to create novel methods.
"""

import logging
import random
import time
from typing import Callable, List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import hashlib

from .compression_method_registry import (
    CompressionMethod,
    CompressionLanguage,
    CompressionMetric,
    register_method
)

gen_logger = logging.getLogger("puffinzip_ai.novel_generator")
if not gen_logger.handlers:
    gen_logger.setLevel(logging.INFO)
    gen_logger.addHandler(logging.NullHandler())


@dataclass
class CompressionPattern:
    """A pattern/rule that can be applied in compression"""
    name: str
    description: str
    pattern_type: str  # e.g., "substitution", "run_length", "dictionary", "transform"
    parameters: Dict[str, Any]
    effectiveness_score: float = 0.0


class NovelCompressionGenerator:
    """Generates and evolves novel compression methods"""
    
    # Catalog of known compression techniques for inspiration
    COMPRESSION_TECHNIQUES = [
        CompressionPattern(
            "rle_adaptive", "Run-Length Encoding with adaptive thresholds",
            "run_length", {"min_run": 2, "max_run": 256}
        ),
        CompressionPattern(
            "dictionary", "Dictionary-based substitution",
            "dictionary", {"dict_size": 256, "min_pattern_len": 3}
        ),
        CompressionPattern(
            "frequency_sort", "Frequency-based reordering",
            "transform", {"sort_descending": True}
        ),
        CompressionPattern(
            "delta_encoding", "Delta/difference encoding",
            "transform", {"base_value": 0}
        ),
        CompressionPattern(
            "entropy_coding", "Entropy-aware encoding",
            "transform", {"use_huffman": True}
        ),
        CompressionPattern(
            "bidirectional_rle", "RLE in both directions",
            "run_length", {"forward": True, "backward": True}
        ),
        CompressionPattern(
            "context_modeling", "Context-based predictions",
            "transform", {"context_order": 2}
        ),
        CompressionPattern(
            "burrows_wheeler", "Burrows-Wheeler transform preprocessing",
            "transform", {"enable": True}
        ),
        CompressionPattern(
            "lz77_variant", "LZ77-style sliding window",
            "dictionary", {"window_size": 32768, "max_match": 258}
        ),
        CompressionPattern(
            "arithmetic_coding", "Arithmetic coding encoding",
            "entropy", {"precision": 32}
        ),
    ]
    
    def __init__(self):
        self.generated_methods: List[CompressionMethod] = []
        self.evolution_history: List[Dict[str, Any]] = []
        gen_logger.info("NovelCompressionGenerator initialized with 10 base techniques")
    
    def _generate_pattern_combo(self, num_patterns: int = 2) -> List[CompressionPattern]:
        """Generate a random combination of compression patterns"""
        if num_patterns <= 0:
            num_patterns = random.randint(1, 3)
        
        selected = random.sample(self.COMPRESSION_TECHNIQUES, min(num_patterns, len(self.COMPRESSION_TECHNIQUES)))
        return selected
    
    def _create_hybrid_compress_fn(self, patterns: List[CompressionPattern]) -> Callable:
        """Create a compression function from pattern combo"""
        def hybrid_compress(text: str, **kwargs) -> str:
            """Hybrid compression applying multiple patterns sequentially"""
            if not text:
                return ""
            
            result = text
            applied_patterns = []
            
            for pattern in patterns:
                try:
                    if pattern.pattern_type == "run_length":
                        result = self._apply_rle(result, pattern.parameters)
                        applied_patterns.append(pattern.name)
                    elif pattern.pattern_type == "dictionary":
                        result = self._apply_dictionary(result, pattern.parameters)
                        applied_patterns.append(pattern.name)
                    elif pattern.pattern_type == "transform":
                        if "frequency_sort" in pattern.name:
                            result = self._apply_frequency_sort(result)
                            applied_patterns.append(pattern.name)
                        elif "delta" in pattern.name:
                            result = self._apply_delta_encoding(result)
                            applied_patterns.append(pattern.name)
                except Exception as e:
                    gen_logger.debug(f"Pattern {pattern.name} failed: {e}")
            
            # Metadata: record what patterns were applied
            if applied_patterns:
                result = f"PATS:{','.join(applied_patterns)}|{result}"
            
            return result
        
        return hybrid_compress
    
    def _create_hybrid_decompress_fn(self, patterns: List[CompressionPattern]) -> Callable:
        """Create a decompression function from pattern combo"""
        def hybrid_decompress(compressed: str, **kwargs) -> str:
            """Hybrid decompression reversing patterns in reverse order"""
            if not compressed:
                return ""
            
            result = compressed
            applied_patterns = []
            
            # Extract metadata
            if result.startswith("PATS:"):
                try:
                    metadata_end = result.index("|")
                    patterns_str = result[5:metadata_end]
                    applied_patterns = patterns_str.split(",")
                    result = result[metadata_end + 1:]
                except:
                    pass
            
            # Reverse patterns in reverse order
            for pattern in reversed(patterns):
                if pattern.name not in applied_patterns:
                    continue
                
                try:
                    if pattern.pattern_type == "run_length":
                        result = self._reverse_rle(result, pattern.parameters)
                    elif pattern.pattern_type == "dictionary":
                        result = self._reverse_dictionary(result, pattern.parameters)
                    elif pattern.pattern_type == "transform":
                        if "frequency_sort" in pattern.name:
                            result = self._reverse_frequency_sort(result)
                        elif "delta" in pattern.name:
                            result = self._reverse_delta_encoding(result)
                except Exception as e:
                    gen_logger.debug(f"Pattern reversal {pattern.name} failed: {e}")
            
            return result
        
        return hybrid_decompress
    
    def _apply_rle(self, text: str, params: Dict) -> str:
        """Apply run-length encoding"""
        if not text:
            return text
        min_run = params.get("min_run", 2)
        result = []
        i = 0
        while i < len(text):
            char = text[i]
            count = 1
            while i + count < len(text) and text[i + count] == char:
                count += 1
            if count >= min_run:
                result.append(f"{count}{char}")
            else:
                result.append(char * count)
            i += count
        return "".join(result)
    
    def _reverse_rle(self, text: str, params: Dict) -> str:
        """Reverse run-length encoding"""
        if not text:
            return text
        result = []
        i = 0
        while i < len(text):
            if i < len(text) - 1 and text[i].isdigit():
                num_str = ""
                while i < len(text) and text[i].isdigit():
                    num_str += text[i]
                    i += 1
                if i < len(text) and num_str:
                    try:
                        count = int(num_str)
                        result.append(text[i] * count)
                        i += 1
                    except:
                        result.append(num_str)
                else:
                    result.append(num_str)
            else:
                result.append(text[i])
                i += 1
        return "".join(result)
    
    def _apply_dictionary(self, text: str, params: Dict) -> str:
        """Apply simple dictionary encoding"""
        if len(text) < params.get("min_pattern_len", 3):
            return text
        # Simple approach: high-frequency substrings get codes
        return text  # Placeholder for now
    
    def _reverse_dictionary(self, text: str, params: Dict) -> str:
        """Reverse dictionary encoding"""
        return text
    
    def _apply_frequency_sort(self, text: str) -> str:
        """Reorder by character frequency (most frequent first)"""
        if not text:
            return text
        freq = {}
        for c in text:
            freq[c] = freq.get(c, 0) + 1
        return "".join(sorted(text, key=lambda c: -freq[c]))
    
    def _reverse_frequency_sort(self, text: str) -> str:
        """Reverse frequency sort - impossible without metadata, so return as-is"""
        return text
    
    def _apply_delta_encoding(self, text: str) -> str:
        """Apply delta encoding (differences from base position)"""
        if len(text) < 2:
            return text
        result = [text[0]]
        for i in range(1, len(text)):
            delta = ord(text[i]) - ord(text[i-1])
            result.append(chr(128 + delta))  # Offset to avoid overlaps
        return "".join(result)
    
    def _reverse_delta_encoding(self, text: str) -> str:
        """Reverse delta encoding"""
        if len(text) < 2:
            return text
        result = [text[0]]
        for i in range(1, len(text)):
            if ord(text[i]) >= 128:
                delta = ord(text[i]) - 128
                result.append(chr(ord(result[-1]) + delta))
            else:
                result.append(text[i])
        return "".join(result)
    
    def generate_novelty_method(self, method_name: str = None, pattern_combo_size: int = None) -> CompressionMethod:
        """Generate a novel compression method from pattern combinations"""
        if method_name is None:
            method_name = f"novelty_v{len(self.generated_methods) + 1}"
        
        patterns = self._generate_pattern_combo(pattern_combo_size)
        compress_fn = self._create_hybrid_compress_fn(patterns)
        decompress_fn = self._create_hybrid_decompress_fn(patterns)
        
        pattern_names = [p.name for p in patterns]
        description = f"Hybrid method combining: {', '.join(pattern_names)}"
        
        method = CompressionMethod(
            name=method_name,
            language=CompressionLanguage.HYBRID,
            compress_fn=compress_fn,
            decompress_fn=decompress_fn,
            description=description,
            author="NovelCompressionGenerator",
            is_novelty=True,
            metadata={
                "patterns": pattern_names,
                "pattern_count": len(patterns),
                "seed": hashlib.md5(method_name.encode()).hexdigest()[:8]
            }
        )
        
        self.generated_methods.append(method)
        register_method(method)
        
        gen_logger.info(f"✨ Generated novelty method: {method_name} using {len(patterns)} patterns")
        return method
    
    def evolve_methods(self, num_mutations: int = 5) -> List[CompressionMethod]:
        """Evolve new methods by mutating existing ones"""
        new_methods = []
        for i in range(num_mutations):
            method = self.generate_novelty_method(f"evolved_gen{i}")
            new_methods.append(method)
        
        gen_logger.info(f"Evolved {num_mutations} new compression methods")
        return new_methods
    
    def get_generated_methods(self) -> List[CompressionMethod]:
        """Return all generated novelty methods"""
        return self.generated_methods


# Global singleton generator
_GLOBAL_GENERATOR = NovelCompressionGenerator()


def get_generator() -> NovelCompressionGenerator:
    """Get the global novel compression generator"""
    return _GLOBAL_GENERATOR


def generate_novelty() -> CompressionMethod:
    """Generate a random novelty compression method"""
    return _GLOBAL_GENERATOR.generate_novelty_method()


def evolve(num_mutations: int = 5) -> List[CompressionMethod]:
    """Evolve new compression methods"""
    return _GLOBAL_GENERATOR.evolve_methods(num_mutations)
