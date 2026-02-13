# PuffinZipAI - Compression Benchmark & Validation
"""
Benchmark compression performance against standard methods to validate AI improvements.
Provides success criteria based on comparing AI compression vs established baselines.
"""

import gzip
import bz2
import lzma
import zlib
import os
import logging
from typing import Dict, Tuple
from pathlib import Path

class CompressionBenchmark:
    """Benchmark and compare compression methods"""
    
    def __init__(self, logger=None):
        self.logger = logger or self._get_default_logger()
        self.baseline_results = {}
        self.ai_results = {}
        
    @staticmethod
    def _get_default_logger():
        """Create a basic logger if none provided"""
        logger = logging.getLogger('CompressionBenchmark')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def compress_with_gzip(self, data: bytes) -> bytes:
        """Compress using gzip (baseline method 1)"""
        return gzip.compress(data, compresslevel=9)
    
    def compress_with_bz2(self, data: bytes) -> bytes:
        """Compress using bz2 (baseline method 2)"""
        return bz2.compress(data, compresslevel=9)
    
    def compress_with_lzma(self, data: bytes) -> bytes:
        """Compress using LZMA (baseline method 3 - usually best)"""
        return lzma.compress(data, preset=9)
    
    def compress_with_zlib(self, data: bytes) -> bytes:
        """Compress using zlib (baseline method 4)"""
        return zlib.compress(data, level=9)
    
    def get_baseline_compression(self, data: bytes, use_best_only: bool = True) -> Dict:
        """
        Get compression results from all standard methods.
        
        Args:
            data: Raw bytes to compress
            use_best_only: If True, return only the best performing method
        
        Returns:
            Dict with compression results including:
            - method: Name of compression method
            - original_size: Size of input data (bytes)
            - compressed_size: Size of compressed data (bytes)
            - compression_ratio: Percentage compressed (0-100)
            - savings: Bytes saved
        """
        original_size = len(data)
        results = []
        
        methods = {
            'gzip': self.compress_with_gzip,
            'bz2': self.compress_with_bz2,
            'lzma': self.compress_with_lzma,
            'zlib': self.compress_with_zlib,
        }
        
        for method_name, compress_func in methods.items():
            try:
                compressed = compress_func(data)
                compressed_size = len(compressed)
                compression_ratio = (1 - compressed_size / original_size) * 100
                savings = original_size - compressed_size
                
                result = {
                    'method': method_name,
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'compression_ratio': compression_ratio,
                    'savings': savings,
                }
                results.append(result)
                
                self.logger.debug(
                    f"Method: {method_name:6s} | "
                    f"Compressed: {compressed_size:10d} bytes | "
                    f"Ratio: {compression_ratio:6.2f}%"
                )
            except Exception as e:
                self.logger.warning(f"Failed to compress with {method_name}: {e}")
        
        if not results:
            raise RuntimeError("No compression methods succeeded")
        
        # Find best method
        best_result = min(results, key=lambda x: x['compressed_size'])
        
        if use_best_only:
            return best_result
        else:
            return {
                'all_methods': results,
                'best_method': best_result
            }
    
    def compare_compression(self, 
                          original_data: bytes,
                          ai_compressed_size: int) -> Dict:
        """
        Compare AI compression against baseline methods.
        
        Args:
            original_data: Original uncompressed data
            ai_compressed_size: Size of data compressed by AI
        
        Returns:
            Dict with comparison results including success flag
        """
        original_size = len(original_data)
        
        # Get baseline (best standard method)
        baseline = self.get_baseline_compression(original_data, use_best_only=True)
        
        # Calculate metrics
        ai_ratio = (1 - ai_compressed_size / original_size) * 100
        baseline_ratio = baseline['compression_ratio']
        improvement = ai_ratio - baseline_ratio
        success = ai_compressed_size < baseline['compressed_size']
        
        comparison = {
            'original_size': original_size,
            'ai_compressed_size': ai_compressed_size,
            'ai_compression_ratio': ai_ratio,
            'baseline_method': baseline['method'],
            'baseline_compressed_size': baseline['compressed_size'],
            'baseline_compression_ratio': baseline_ratio,
            'improvement_percentage': improvement,
            'success': success,
            'savings_vs_baseline': baseline['compressed_size'] - ai_compressed_size,
        }
        
        return comparison
    
    def validate_ai_success(self,
                           original_data: bytes,
                           ai_compressed_data: bytes) -> Tuple[bool, Dict]:
        """
        Validate if AI compression is better than standard methods.
        
        Args:
            original_data: Original uncompressed data
            ai_compressed_data: AI-compressed data
        
        Returns:
            Tuple of (success: bool, metrics: dict)
        """
        comparison = self.compare_compression(original_data, len(ai_compressed_data))
        
        success_message = (
            f"\n{'='*70}\n"
            f"COMPRESSION VALIDATION REPORT\n"
            f"{'='*70}\n"
            f"Original Size:           {comparison['original_size']:,} bytes\n"
            f"\nAI Compression:\n"
            f"  Compressed Size:       {comparison['ai_compressed_size']:,} bytes\n"
            f"  Compression Ratio:     {comparison['ai_compression_ratio']:.2f}%\n"
            f"\nBaseline ({comparison['baseline_method'].upper()}):\n"
            f"  Compressed Size:       {comparison['baseline_compressed_size']:,} bytes\n"
            f"  Compression Ratio:     {comparison['baseline_compression_ratio']:.2f}%\n"
            f"\nResult:\n"
            f"  Improvement:           {comparison['improvement_percentage']:+.2f}%\n"
            f"  Additional Savings:    {comparison['savings_vs_baseline']:,} bytes\n"
            f"  Status:                {'✅ SUCCESS' if comparison['success'] else '❌ FAILED'}\n"
            f"{'='*70}\n"
        )
        
        log_level = logging.INFO if comparison['success'] else logging.WARNING
        self.logger.log(log_level, success_message)
        
        return comparison['success'], comparison
    
    def format_comparison_report(self, comparison: Dict) -> str:
        """Format comparison results as readable report"""
        status = '✅ SUCCESS' if comparison['success'] else '❌ FAILED'
        
        report = (
            f"AI vs {comparison['baseline_method'].upper()} Compression:\n"
            f"  Original: {comparison['original_size']:,} bytes\n"
            f"  AI:       {comparison['ai_compressed_size']:,} bytes ({comparison['ai_compression_ratio']:.2f}%)\n"
            f"  Baseline: {comparison['baseline_compressed_size']:,} bytes ({comparison['baseline_compression_ratio']:.2f}%)\n"
            f"  Gain:     {comparison['improvement_percentage']:+.2f}% ({comparison['savings_vs_baseline']:+,} bytes)\n"
            f"  Status:   {status}"
        )
        
        return report
