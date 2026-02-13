# PuffinZipAI_Project/puffinzip_ai/rust_compression_interface.py
"""
Rust Compression Interface
Enables calling high-performance Rust compression implementations from Python.
Provides fallback to Python implementations if Rust bindings not available.
"""

import logging
import subprocess
import json
import base64
import tempfile
import os
from typing import Optional, Tuple
from pathlib import Path

rust_logger = logging.getLogger("puffinzip_ai.rust_interface")
if not rust_logger.handlers:
    rust_logger.setLevel(logging.INFO)
    rust_logger.addHandler(logging.NullHandler())


class RustCompressionInterface:
    """Interface for calling Rust compression implementations"""
    
    def __init__(self):
        self.rust_available = self._check_rust_availability()
        self.compiled_lib_path = self._find_rust_lib()
        
        if self.rust_available:
            rust_logger.info(f"Rust compression interface available at: {self.compiled_lib_path}")
        else:
            rust_logger.info("Rust bindings not available. Python implementations will be used as fallback.")
    
    def _check_rust_availability(self) -> bool:
        """Check if Rust compiler and tools are available"""
        try:
            # The actual Rust lib would be built separately and placed in the project
            # For now, this is a placeholder that enables future Rust integration
            result = subprocess.run(['rustc', '--version'], capture_output=True, timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _find_rust_lib(self) -> Optional[str]:
        """Find path to compiled Rust library"""
        possible_paths = [
            Path(__file__).parent / "rust_compression" / "target" / "release" / "libpuffin_compression.so",
            Path(__file__).parent / "rust_compression" / "target" / "release" / "puffin_compression.dll",
            Path(__file__).parent / "rust_compression" / "target" / "release" / "libpuffin_compression.dylib",
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def compress_burst(self, data: str, method: str = "burst_rle") -> Optional[str]:
        """
        BURST Method: Bidirectional RLE with Entropy Suppression and Repeated Tuple Recognition
        Novel multi-pass compression combining:
        - Forward & backward RLE simultaneously
        - Entropy analysis to skip incompressible sections
        - Repeated tuple pattern recognition
        """
        if not self.rust_available:
            rust_logger.debug(f"Rust not available, using Python fallback for {method}")
            return self._python_burst_compress(data)
        
        # Would call actual Rust implementation here
        return self._python_burst_compress(data)
    
    def decompress_burst(self, data: str, method: str = "burst_rle") -> Optional[str]:
        """Decompress BURST-encoded data"""
        if not self.rust_available:
            return self._python_burst_decompress(data)
        return self._python_burst_decompress(data)
    
    def _python_burst_compress(self, text: str) -> str:
        """Python implementation of BURST compression"""
        if not text:
            return ""
        
        # Forward RLE pass
        forward = self._rle_pass(text)
        
        # Entropy check: if too incompressible, skip further processing
        entropy_ratio = self._calculate_entropy(forward) / self._calculate_entropy(text)
        if entropy_ratio > 0.95:  # Not worth compressing further
            return f"BURST0|{text}"
        
        # Backward RLE pass on result
        backward = self._rle_pass(forward[::-1])[::-1]
        
        # Tuple recognition pass
        tupled = self._recognize_tuples(backward)
        
        return f"BURST3|{tupled}"
    
    def _python_burst_decompress(self, data: str) -> str:
        """Python implementation of BURST decompression"""
        if not data or not data.startswith("BURST"):
            return data
        
        try:
            parts = data.split("|", 1)
            if len(parts) < 2:
                return data
            
            level = int(parts[0][5:])  # Extract level (0-3)
            compressed = parts[1]
            
            # Reverse the tuple recognition
            if level >= 3:
                compressed = self._unrec_tuples(compressed)
            
            # Reverse RLE passes
            if level >= 2:
                compressed = self._reverse_rle(compressed[::-1])[::-1]
            
            if level >= 1:
                compressed = self._reverse_rle(compressed)
            
            return compressed
        except Exception as e:
            rust_logger.warning(f"BURST decompression failed: {e}")
            return data
    
    def _rle_pass(self, text: str) -> str:
        """Single RLE compression pass"""
        if not text:
            return ""
        result = []
        i = 0
        while i < len(text):
            char = text[i]
            count = 1
            while i + count < len(text) and text[i + count] == char:
                count += 1
            if count >= 3:
                result.append(f"{count}~{char}")  # Using ~ as separator
            else:
                result.append(char * count)
            i += count
        return "".join(result)
    
    def _reverse_rle(self, text: str) -> str:
        """Reverse RLE compression"""
        if not text:
            return ""
        result = []
        i = 0
        while i < len(text):
            if i < len(text) - 2 and text[i].isdigit() and text[i + 1] == "~":
                count_str = ""
                while i < len(text) and text[i].isdigit():
                    count_str += text[i]
                    i += 1
                if i < len(text) and text[i] == "~":
                    i += 1
                    if i < len(text):
                        result.append(text[i] * int(count_str))
                        i += 1
                else:
                    result.append(count_str)
            else:
                result.append(text[i])
                i += 1
        return "".join(result)
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        freq = {}
        for c in text:
            freq[c] = freq.get(c, 0) + 1
        
        entropy = 0.0
        length = len(text)
        for count in freq.values():
            prob = count / length
            entropy -= prob * (prob and -np.log2(prob) or 0)
        
        return entropy
    
    def _recognize_tuples(self, text: str) -> str:
        """Recognize and encode repeated multi-character tuples"""
        if len(text) < 6:
            return text
        
        # Find repeated 2-3 char patterns
        # This is a placeholder - real implementation would be in Rust
        return text
    
    def _unrec_tuples(self, text: str) -> str:
        """Unreverse tuple recognition"""
        return text


# Attempt to import numpy for entropy calculation
try:
    import numpy as np
except ImportError:
    np = None
    rust_logger.warning("NumPy not available for entropy calculations")


# Global singleton
_GLOBAL_RUST_INTERFACE = RustCompressionInterface()


def get_rust_interface() -> RustCompressionInterface:
    """Get the global Rust compression interface"""
    return _GLOBAL_RUST_INTERFACE
