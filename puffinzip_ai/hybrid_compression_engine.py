# PuffinZipAI_Project/puffinzip_ai/hybrid_compression_engine.py
"""
Hybrid Compression Engine
Unified interface for Python, Rust, and novelty compression methods.
Allows the AI to discover, test, and evolve compression algorithms.
"""

import logging
import time
from typing import Callable, Dict, Optional, List, Tuple
from dataclasses import dataclass

from .compression_method_registry import (
    CompressionMethod,
    CompressionLanguage,
    get_registry,
    register_python_method
)
from .novel_compression_generator import get_generator
from .rust_compression_interface import get_rust_interface

engine_logger = logging.getLogger("puffinzip_ai.hybrid_engine")
if not engine_logger.handlers:
    engine_logger.setLevel(logging.INFO)
    engine_logger.addHandler(logging.NullHandler())


class HybridCompressionEngine:
    """Unified compression engine supporting multiple languages and novel methods"""
    
    def __init__(self):
        self.registry = get_registry()
        self.generator = get_generator()
        self.rust = get_rust_interface()
        self._initialize_built_in_methods()
        engine_logger.info("HybridCompressionEngine initialized")
    
    def _initialize_built_in_methods(self):
        """Register built-in compression methods"""
        
        # BURST Method (Bidirectional RLE + Entropy Suppression + Tuple Recognition)
        register_python_method(
            name="burst",
            compress_fn=self.rust.compress_burst,
            decompress_fn=self.rust.decompress_burst,
            description="BURST: Bidirectional RLE with Entropy Suppression and Repeated Tuple Recognition",
            author="hybrid_engine",
            is_novelty=False
        )
        
        # Delta + RLE Hybrid
        def delta_rle_compress(text: str, **kwargs) -> str:
            """Apply delta encoding followed by RLE"""
            if not text or len(text) < 2:
                return text
            
            # Delta pass
            delta = [text[0]]
            for i in range(1, len(text)):
                diff = (ord(text[i]) - ord(text[i-1])) % 256
                delta.append(chr(diff))
            
            # RLE pass
            result = []
            i = 0
            while i < len(delta):
                char = delta[i]
                count = 1
                while i + count < len(delta) and delta[i + count] == char:
                    count += 1
                if count >= 3:
                    result.append(f"{count}:{chr(count)}:")
                else:
                    result.append(char * count)
                i += count
            
            return "DRLE|" + "".join(result)
        
        def delta_rle_decompress(text: str, **kwargs) -> str:
            """Reverse delta + RLE"""
            if not text or not text.startswith("DRLE|"):
                return text
            
            compressed = text[5:]
            
            # Reverse RLE
            result = []
            i = 0
            while i < len(compressed):
                if i < len(compressed) - 2 and compressed[i] == compressed[i+2] == ":":
                    count = ord(compressed[i + 1])
                    char = compressed[i]
                    result.append(char * count)
                    i += 3
                else:
                    result.append(compressed[i])
                    i += 1
            
            delta = "".join(result)
            
            # Reverse delta
            if not delta:
                return delta
            
            original = [delta[0]]
            for i in range(1, len(delta)):
                val = (ord(original[-1]) + ord(delta[i])) % 256
                original.append(chr(val))
            
            return "".join(original)
        
        register_python_method(
            name="delta_rle",
            compress_fn=delta_rle_compress,
            decompress_fn=delta_rle_decompress,
            description="Delta Encoding + RLE for text with patterns",
            author="hybrid_engine",
            is_novelty=False
        )
        
        # Frequency-based codec
        def freq_codec_compress(text: str, **kwargs) -> str:
            """Compress by reordering by frequency"""
            if not text:
                return "FREQ|"
            
            freq = {}
            for c in text:
                freq[c] = freq.get(c, 0) + 1
            
            # Create mapping
            sorted_chars = sorted(freq.items(), key=lambda x: -x[1])
            char_map = {c: chr(i) for i, (c, _) in enumerate(sorted_chars)}
            
            # Encode
            encoded = "".join(char_map[c] for c in text)
            
            # Metadata
            metadata = "".join(f"{c},{count};" for c, count in sorted_chars)
            
            return f"FREQ|{metadata}|{encoded}"
        
        def freq_codec_decompress(text: str, **kwargs) -> str:
            """Decompress frequency codec"""
            if not text or not text.startswith("FREQ|"):
                return text
            
            try:
                parts = text[5:].split("|", 1)
                if len(parts) < 2:
                    return text
                
                metadata, encoded = parts
                sorted_chars = []
                for item in metadata.split(";"):
                    if item:
                        c, count = item.split(",")
                        sorted_chars.append(c)
                
                # Reverse mapping
                char_map = {chr(i): c for i, c in enumerate(sorted_chars)}
                
                return "".join(char_map.get(c, c) for c in encoded)
            except Exception as e:
                engine_logger.warning(f"Frequency codec decompression failed: {e}")
                return text
        
        register_python_method(
            name="frequency_codec",
            compress_fn=freq_codec_compress,
            decompress_fn=freq_codec_decompress,
            description="Frequency-based character remapping",
            author="hybrid_engine",
            is_novelty=False
        )
    
    def compress(self, text: str, method: str = "burst") -> str:
        """Compress using specified method"""
        method_obj = self.registry.get_method(method)
        
        if not method_obj:
            engine_logger.warning(f"Method '{method}' not found, using burst")
            method_obj = self.registry.get_method("burst")
        
        if not method_obj:
            engine_logger.error("No compression methods available!")
            return text
        
        try:
            start_time = time.time()
            compressed = method_obj.compress_fn(text)
            elapsed_ms = (time.time() - start_time) * 1000
            
            ratio = len(compressed) / len(text) if text else 0
            method_obj.metrics.update(ratio, elapsed_ms, 0, 0)
            
            engine_logger.info(f"Compressed with {method}: {len(text)} -> {len(compressed)} bytes ({ratio:.2%})")
            return compressed
        except Exception as e:
            engine_logger.error(f"Compression with {method} failed: {e}")
            return text
    
    def decompress(self, data: str, method: str = "burst") -> str:
        """Decompress using specified method"""
        method_obj = self.registry.get_method(method)
        
        if not method_obj:
            engine_logger.warning(f"Method '{method}' not found, trying burst")
            method_obj = self.registry.get_method("burst")
        
        if not method_obj:
            engine_logger.error("No compression methods available!")
            return data
        
        try:
            start_time = time.time()
            decompressed = method_obj.decompress_fn(data)
            elapsed_ms = (time.time() - start_time) * 1000
            
            method_obj.metrics.encode_time_ms = elapsed_ms  # Update decode time
            
            engine_logger.info(f"Decompressed with {method}: {len(data)} -> {len(decompressed)} bytes")
            return decompressed
        except Exception as e:
            engine_logger.error(f"Decompression with {method} failed: {e}")
            return data
    
    def discover_novelty_method(self, name: str = None) -> CompressionMethod:
        """Discover a new compression method"""
        method = self.generator.generate_novelty_method(name)
        engine_logger.info(f"✨ Discovered novelty method: {method.name}")
        return method
    
    def evolve_methods(self, num_mutations: int = 5) -> List[CompressionMethod]:
        """Evolve compression methods"""
        methods = self.generator.evolve_methods(num_mutations)
        engine_logger.info(f"Evolved {num_mutations} new compression methods")
        return methods
    
    def test_method(self, method_name: str, test_data: str) -> Dict[str, float]:
        """Test a method and return metrics"""
        method = self.registry.get_method(method_name)
        if not method:
            return {}
        
        try:
            # Compress
            start = time.time()
            compressed = method.compress_fn(test_data)
            compress_time = (time.time() - start) * 1000
            
            # Decompress
            start = time.time()
            decompressed = method.decompress_fn(compressed)
            decompress_time = (time.time() - start) * 1000
            
            # Validate
            success = decompressed == test_data
            
            ratio = len(compressed) / len(test_data) if test_data else 0
            method.metrics.update(ratio, compress_time, decompress_time, 0, success)
            
            return {
                "compression_ratio": ratio,
                "compress_time_ms": compress_time,
                "decompress_time_ms": decompress_time,
                "success": success,
                "original_size": len(test_data),
                "compressed_size": len(compressed)
            }
        except Exception as e:
            engine_logger.error(f"Test failed for {method_name}: {e}")
            return {}
    
    def get_best_method(self, metric: str = "compression_ratio") -> Optional[str]:
        """Get the best performing method for a metric"""
        top = self.registry.get_top_methods(metric, 1)
        return top[0][0] if top else None
    
    def list_available_methods(self) -> Dict[str, str]:
        """List all available compression methods"""
        methods = {}
        for name, method in self.registry.methods.items():
            badge = "✨" if method.is_novelty else ""
            methods[f"{badge} {name}"] = method.description
        return methods


# Global singleton
_GLOBAL_ENGINE = None


def get_hybrid_engine() -> HybridCompressionEngine:
    """Get the global hybrid compression engine"""
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = HybridCompressionEngine()
    return _GLOBAL_ENGINE
