# PuffinZipAI_Project/puffinzip_ai/gpu_core/gpu_rle_interface.py
import logging

logger = logging.getLogger("PuffinZipAI_GPU_RLE_Interface")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

CUPY_AVAILABLE = False
cp = None
NUMBA_AVAILABLE = False
nb_cuda = None

try:
    from ..rle_utils import rle_compress as cpu_rle_compress, rle_decompress as cpu_rle_decompress
except ImportError:
    # This fallback is for catastrophic cases; the main app should not hit this.
    def cpu_rle_compress(d, **k): return "ERROR_CPU_RLE_UNAVAILABLE_IN_GPU_IFACE"
    def cpu_rle_decompress(d, **k): return "ERROR_CPU_RLE_UNAVAILABLE_IN_GPU_IFACE"
    logger.critical("Could not import CPU RLE functions for GPU interface fallback. This is a critical error.")


try:
    import cupy as cp
    CUPY_AVAILABLE = True
    logger.debug("CuPy available for GPU RLE.")
except ImportError:
    logger.debug("CuPy not available for GPU RLE.")
    pass

try:
    from numba import cuda as nb_cuda_mod
    if nb_cuda_mod.is_available():
        NUMBA_AVAILABLE = True
        nb_cuda = nb_cuda_mod
        logger.debug("Numba CUDA available for GPU RLE.")
    else:
        logger.debug("Numba CUDA not available (no compatible GPU or drivers).")
except ImportError:
    logger.debug("Numba not available for GPU RLE.")
    pass


def gpu_accelerated_rle_compress(text_data: str, method: str = "simple", min_run_len_override: int = None,
                                 use_gpu: bool = True, gpu_id: int = 0) -> str:
    if not use_gpu or (not CUPY_AVAILABLE and not NUMBA_AVAILABLE):
        logger.warning("GPU RLE compress called but no GPU lib is available. This should have been caught earlier. Using CPU as last resort.")
        return cpu_rle_compress(text_data, method=method, min_run_len_override=min_run_len_override)

    # In a future implementation, this would contain custom CUDA kernels.
    # For now, it explicitly returns an error to signal that the GPU path is not complete.
    # This prevents silent fallbacks to CPU.
    logger.warning(f"GPU RLE compress called, but a real GPU implementation is not available. Returning error code.")
    return "ERROR_GPU_RLE_C_NOT_IMPLEMENTED"


def gpu_accelerated_rle_decompress(compressed_text_data: str, method: str = "simple",
                                   min_run_len_override: int = None, expected_output_size_hint: int = 0,
                                   use_gpu: bool = True, gpu_id: int = 0) -> str:
    if not use_gpu or (not CUPY_AVAILABLE and not NUMBA_AVAILABLE):
        logger.warning("GPU RLE decompress called but no GPU lib is available. This should have been caught earlier. Using CPU as last resort.")
        return cpu_rle_decompress(compressed_text_data, method=method, min_run_len_override=min_run_len_override)

    # In a future implementation, this would contain custom CUDA kernels.
    # For now, it explicitly returns an error.
    logger.warning(f"GPU RLE decompress called, but a real GPU implementation is not available. Returning error code.")
    return "ERROR_GPU_RLE_D_NOT_IMPLEMENTED"

if __name__ == '__main__':
    print("--- Testing GPU RLE Interface (Explicit Errors) ---")
    print(f"CuPy Available: {CUPY_AVAILABLE}")
    print(f"Numba CUDA Available: {NUMBA_AVAILABLE}")

    test_string = "AAAAABBBCCCDDDDEEEEE"
    print(f"\nOriginal: {test_string}")

    compressed_simple_gpu = gpu_accelerated_rle_compress(test_string, method="simple", use_gpu=True)
    print(f"Compressed (Simple GPU, expect error): {compressed_simple_gpu}")
    compressed_adv_gpu = gpu_accelerated_rle_compress(test_string, method="advanced", use_gpu=True)
    print(f"Compressed (Advanced GPU, expect error): {compressed_adv_gpu}")

    decompressed_simple_gpu = gpu_accelerated_rle_decompress(compressed_simple_gpu, method="simple", use_gpu=True)
    print(f"Decompressed (Simple GPU, expect error): {decompressed_simple_gpu}")