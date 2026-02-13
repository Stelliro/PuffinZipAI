# PuffinZipAI_Project/puffinzip_ai/compression_method_registry.py
"""
Hybrid Compression Method Registry
Enables dynamic registration and discovery of compression methods from any language/implementation.
Allows the AI to generate and test novel compression algorithms.
"""

import logging
import json
import time
import hashlib
from typing import Callable, Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

registry_logger = logging.getLogger("puffinzip_ai.compression_registry")
if not registry_logger.handlers:
    registry_logger.setLevel(logging.INFO)
    registry_logger.addHandler(logging.NullHandler())


class CompressionLanguage(Enum):
    """Supported implementation languages for compression methods"""
    PYTHON = "python"
    RUST = "rust"
    CPP = "cpp"
    CUDA = "cuda"
    HYBRID = "hybrid"  # Multi-language


class CompressionMetric:
    """Tracks performance metrics for a compression method"""
    def __init__(self):
        self.compression_ratio = 0.0
        self.encode_time_ms = 0.0
        self.decode_time_ms = 0.0
        self.memory_peak_mb = 0.0
        self.test_count = 0
        self.avg_success_rate = 1.0
        
    def update(self, ratio: float, encode_ms: float, decode_ms: float, mem_mb: float, success: bool = True):
        """Update metrics with new observation"""
        self.compression_ratio = (self.compression_ratio * self.test_count + ratio) / (self.test_count + 1)
        self.encode_time_ms = (self.encode_time_ms * self.test_count + encode_ms) / (self.test_count + 1)
        self.decode_time_ms = (self.decode_time_ms * self.test_count + decode_ms) / (self.test_count + 1)
        self.memory_peak_mb = max(self.memory_peak_mb, mem_mb)
        self.test_count += 1
        if not success:
            self.avg_success_rate = self.avg_success_rate * 0.99


@dataclass
class CompressionMethod:
    """Descriptor for a compression method"""
    name: str
    language: CompressionLanguage
    compress_fn: Callable
    decompress_fn: Callable
    version: str = "1.0"
    author: str = "system"
    description: str = ""
    created_at: float = field(default_factory=time.time)
    is_novelty: bool = False  # True if AI-generated
    metrics: CompressionMetric = field(default_factory=CompressionMetric)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_signature(self) -> str:
        """Generate unique signature for this method"""
        sig_str = f"{self.name}_{self.version}_{self.language.value}"
        return hashlib.md5(sig_str.encode()).hexdigest()[:16]


class CompressionRegistry:
    """Central registry for all compression methods"""
    
    def __init__(self):
        self.methods: Dict[str, CompressionMethod] = {}
        self.method_history: List[Dict[str, Any]] = []
        self.novelty_history: List[CompressionMethod] = []
        registry_logger.info("CompressionRegistry initialized")
    
    def register(self, method: CompressionMethod) -> bool:
        """Register a new compression method"""
        if method.name in self.methods:
            registry_logger.warning(f"Method '{method.name}' already registered. Updating.")
        
        self.methods[method.name] = method
        self.method_history.append({
            "name": method.name,
            "language": method.language.value,
            "timestamp": time.time(),
            "is_novelty": method.is_novelty,
            "signature": method.get_signature()
        })
        
        if method.is_novelty:
            self.novelty_history.append(method)
            registry_logger.info(f"✨ New novelty method registered: {method.name} (by {method.author})")
        else:
            registry_logger.info(f"Registered compression method: {method.name}")
        
        return True
    
    def get_method(self, name: str) -> Optional[CompressionMethod]:
        """Retrieve a registered method by name"""
        return self.methods.get(name)
    
    def list_methods(self, language: Optional[CompressionLanguage] = None) -> List[str]:
        """List all registered methods, optionally filtered by language"""
        if language is None:
            return list(self.methods.keys())
        return [name for name, method in self.methods.items() if method.language == language]
    
    def get_top_methods(self, metric: str = "compression_ratio", limit: int = 5) -> List[Tuple[str, float]]:
        """Get top methods by specified metric"""
        scored = []
        for name, method in self.methods.items():
            if metric == "compression_ratio":
                score = method.metrics.compression_ratio
            elif metric == "speed":
                score = 1.0 / (method.metrics.encode_time_ms + method.metrics.decode_time_ms + 0.001)
            elif metric == "efficiency":
                # Balance compression ratio and speed
                score = (method.metrics.compression_ratio * 100) / (method.metrics.encode_time_ms + 1)
            else:
                score = 0.0
            scored.append((name, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]
    
    def export_methods(self) -> Dict[str, Any]:
        """Export all methods as JSON-serializable dict"""
        exported = {
            "methods": {},
            "history": self.method_history,
            "novelty_count": len(self.novelty_history)
        }
        
        for name, method in self.methods.items():
            exported["methods"][name] = {
                "language": method.language.value,
                "version": method.version,
                "author": method.author,
                "description": method.description,
                "is_novelty": method.is_novelty,
                "metrics": {
                    "compression_ratio": method.metrics.compression_ratio,
                    "encode_time_ms": method.metrics.encode_time_ms,
                    "decode_time_ms": method.metrics.decode_time_ms,
                    "memory_peak_mb": method.metrics.memory_peak_mb,
                    "test_count": method.metrics.test_count,
                    "success_rate": method.metrics.avg_success_rate
                }
            }
        
        return exported
    
    def get_novelty_methods(self) -> List[CompressionMethod]:
        """Get all AI-generated novelty methods"""
        return self.novelty_history


# Global singleton registry
_GLOBAL_REGISTRY = CompressionRegistry()


def get_registry() -> CompressionRegistry:
    """Get the global compression registry"""
    return _GLOBAL_REGISTRY


def register_method(method: CompressionMethod) -> bool:
    """Convenience function to register with global registry"""
    return _GLOBAL_REGISTRY.register(method)


def register_python_method(
    name: str,
    compress_fn: Callable,
    decompress_fn: Callable,
    description: str = "",
    author: str = "user",
    is_novelty: bool = False
) -> CompressionMethod:
    """Convenience for registering Python compression methods"""
    method = CompressionMethod(
        name=name,
        language=CompressionLanguage.PYTHON,
        compress_fn=compress_fn,
        decompress_fn=decompress_fn,
        description=description,
        author=author,
        is_novelty=is_novelty
    )
    _GLOBAL_REGISTRY.register(method)
    return method
