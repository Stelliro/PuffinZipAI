# PuffinZipAI_Project/puffinzip_ai/gpu_core/gpu_rle_interface.py
import logging
import threading
from typing import List, Tuple, Union, Optional

import numpy as np

logger = logging.getLogger("PuffinZipAI_GPU_RLE_Interface")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

CUPY_AVAILABLE = False
cp = None
CuPyOutOfMemoryError = RuntimeError
NUMBA_AVAILABLE = False
nb_cuda = None

try:
    from ..rle_utils import (
        ABSOLUTE_MAX_PARSED_COUNT,
        ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE,
        MAX_COUNT_STRING_DIGITS,
        MIN_ENCODABLE_RUN_LENGTH,
        RLE_DECOMPRESSION_ERRORS,
        RLE_RUN_MARKER,
        rle_compress as cpu_rle_compress,
        rle_decompress as cpu_rle_decompress,
    )
except ImportError:
    # This fallback is for catastrophic cases; the main app should not hit this.
    def cpu_rle_compress(d, **k): return "ERROR_CPU_RLE_UNAVAILABLE_IN_GPU_IFACE"
    def cpu_rle_decompress(d, **k): return "ERROR_CPU_RLE_UNAVAILABLE_IN_GPU_IFACE"
    ABSOLUTE_MAX_PARSED_COUNT = 100 * 1024 * 1024
    ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE = 200 * 1024 * 1024
    MAX_COUNT_STRING_DIGITS = 9
    MIN_ENCODABLE_RUN_LENGTH = 3
    RLE_RUN_MARKER = '\x02'
    RLE_DECOMPRESSION_ERRORS = {
        "ERROR_INVALID_RLE_FORMAT_NO_COUNT",
        "ERROR_INVALID_RLE_FORMAT_BAD_COUNT",
        "ERROR_INVALID_RLE_FORMAT_NO_CHAR_AFTER_COUNT",
        "ERROR_MALFORMED_RLE_STRING",
        "ERROR_COUNT_TOO_LARGE_FOR_SAFETY",
        "ERROR_TOTAL_SIZE_LIMIT_EXCEEDED",
        "ERROR_MEMORY_DURING_CHUNK_ALLOCATION",
        "ERROR_MEMORY_DURING_FINAL_JOIN",
    }
    logger.critical("Could not import CPU RLE functions for GPU interface fallback. This is a critical error.")




try:
    from ..config import (
        GPU_RLE_TARGET_VRAM_USAGE_FRACTION,
        GPU_RLE_WORKSPACE_MIN_MB,
        GPU_RLE_WORKSPACE_MAX_MB,
        GPU_RLE_WORKSPACE_TARGET_MB,
    )
except ImportError:
    GPU_RLE_TARGET_VRAM_USAGE_FRACTION = 0.0
    GPU_RLE_WORKSPACE_MIN_MB = 0
    GPU_RLE_WORKSPACE_MAX_MB = 0
    GPU_RLE_WORKSPACE_TARGET_MB = 0


try:
    import cupy as cp
    CUPY_AVAILABLE = True
    from cupy.cuda.memory import OutOfMemoryError as CuPyOutOfMemoryError  # type: ignore
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


_GPU_WORKSPACE_BUFFERS = {}
_GPU_WORKSPACE_LOCK = threading.Lock()


def _get_total_mem_bytes(props) -> int:
    total_mem = 0
    if isinstance(props, dict):
        total_mem = int(props.get("totalGlobalMem", 0))
    else:
        total_mem = int(getattr(props, "totalGlobalMem", 0) or 0)
    return max(total_mem, 0)


def _ensure_workspace_allocation(gpu_id: int, required_elements: int = 0) -> Optional["cp.ndarray"]:  # type: ignore[name-defined]
    if not CUPY_AVAILABLE:
        return None

    target_fraction = float(max(0.0, min(1.0, GPU_RLE_TARGET_VRAM_USAGE_FRACTION)))
    min_bytes_cfg = int(max(0, GPU_RLE_WORKSPACE_MIN_MB) * 1024 * 1024)
    max_bytes_cfg = int(max(0, GPU_RLE_WORKSPACE_MAX_MB) * 1024 * 1024) if GPU_RLE_WORKSPACE_MAX_MB else 0
    target_bytes_cfg = int(max(0, GPU_RLE_WORKSPACE_TARGET_MB) * 1024 * 1024)

    with _GPU_WORKSPACE_LOCK:
        existing = _GPU_WORKSPACE_BUFFERS.get(gpu_id)
        if existing is not None and required_elements and existing.size >= required_elements:
            return existing

    with cp.cuda.Device(gpu_id):
        props = cp.cuda.runtime.getDeviceProperties(gpu_id)
        total_mem_bytes = _get_total_mem_bytes(props)
        if total_mem_bytes <= 0 and required_elements <= 0:
            return None

        target_bytes = int(total_mem_bytes * target_fraction) if total_mem_bytes > 0 else 0
        if target_bytes_cfg > 0:
            target_bytes = max(target_bytes, target_bytes_cfg)
        required_bytes = int(required_elements) * np.dtype(np.uint32).itemsize if required_elements else 0
        if required_bytes > 0:
            target_bytes = max(target_bytes, required_bytes)
        if min_bytes_cfg > 0:
            target_bytes = max(target_bytes, min_bytes_cfg)
        if max_bytes_cfg > 0:
            target_bytes = min(target_bytes, max_bytes_cfg)
        if total_mem_bytes > 0:
            target_bytes = min(target_bytes, total_mem_bytes)

        if target_bytes <= 0:
            if required_bytes > 0:
                target_bytes = required_bytes
            else:
                return None

        target_elements = max(1, target_bytes // np.dtype(np.uint32).itemsize)

        with _GPU_WORKSPACE_LOCK:
            existing = _GPU_WORKSPACE_BUFFERS.get(gpu_id)
            if existing is not None and existing.size >= target_elements:
                return existing

        attempt_elements = max(target_elements, required_elements)
        desired_allocation_bytes = target_bytes
        min_elements_allowed = max(
            1,
            required_elements if required_elements else (min_bytes_cfg // np.dtype(np.uint32).itemsize if min_bytes_cfg else 1),
        )

        logger.info(
            "GPU RLE workspace allocation started on device %s: Total GPU VRAM=%.1f MB, Target Fraction=%.1f%%, Target Allocation=%.1f MB",
            gpu_id,
            total_mem_bytes / (1024 * 1024) if total_mem_bytes > 0 else 0,
            target_fraction * 100,
            desired_allocation_bytes / (1024 * 1024),
        )

        while attempt_elements >= min_elements_allowed:
            try:
                workspace = cp.empty(int(attempt_elements), dtype=cp.uint32)
                with _GPU_WORKSPACE_LOCK:
                    _GPU_WORKSPACE_BUFFERS[gpu_id] = workspace
                logger.info(
                    "GPU RLE reserved %.1f MB (requested %.1f MB) on device %s for workspace (elements=%d).",
                    workspace.nbytes / (1024 * 1024),
                    desired_allocation_bytes / (1024 * 1024),
                    gpu_id,
                    workspace.size,
                )
                return workspace
            except CuPyOutOfMemoryError:
                next_attempt = int(attempt_elements * 0.90)
                if next_attempt < min_elements_allowed:
                    if attempt_elements == min_elements_allowed:
                        break
                    next_attempt = min_elements_allowed
                if next_attempt == attempt_elements:
                    break
                attempt_elements = next_attempt

        logger.warning(
            "GPU RLE could not reserve workspace on device %s (required_elements=%d, requested_bytes=%d).",
            gpu_id,
            required_elements,
            desired_allocation_bytes,
        )
        with _GPU_WORKSPACE_LOCK:
            _GPU_WORKSPACE_BUFFERS.pop(gpu_id, None)
        return None


def _acquire_workspace_slice(gpu_id: int, required_length: int) -> Optional["cp.ndarray"]:  # type: ignore[name-defined]
    if required_length <= 0:
        return None
    workspace = _ensure_workspace_allocation(gpu_id, required_length)
    if workspace is None or workspace.size < required_length:
        return None
    return workspace[:required_length]


def _encode_string_to_codepoints(text: str) -> np.ndarray:
    """Return an array of UTF-32 codepoints for *text* (empty array for empty input)."""
    if not text:
        return np.empty(0, dtype=np.uint32)
    # UTF-32 little endian keeps one codepoint per 4-byte chunk, making slicing predictable.
    return np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)


def _codepoints_to_string(codepoints: np.ndarray) -> str:
    if codepoints.size == 0:
        return ""
    # Convert lazily to avoid allocating intermediate large Python lists.
    return "".join(chr(int(code)) for code in codepoints)


def _build_run_boundaries_gpu(gpu_codepoints):
    """Return (run_lengths, run_values) from a CuPy array of codepoints."""
    if gpu_codepoints.size == 0:
        return cp.array([], dtype=cp.int64), cp.array([], dtype=gpu_codepoints.dtype)

    if gpu_codepoints.size == 1:
        return cp.array([1], dtype=cp.int64), gpu_codepoints.copy()

    diffs = cp.diff(gpu_codepoints)
    change_indices = cp.where(diffs != 0)[0] + 1
    # Ensure int64 for diff stability when dealing with large arrays.
    run_boundaries = cp.concatenate(
        [
            cp.array([0], dtype=cp.int64),
            change_indices.astype(cp.int64),
            cp.array([gpu_codepoints.size], dtype=cp.int64),
        ]
    )
    run_lengths = cp.diff(run_boundaries)
    run_values = gpu_codepoints[run_boundaries[:-1]]
    return run_lengths, run_values


def _parse_compressed_segments(
    compressed_text: str,
) -> Union[str, Tuple[List[Tuple[bool, Union[str, Tuple[int, str]]]], int]]:
    """
    Parse *compressed_text* (marker-framed format) into segments.

    Format:  MARKER + count_digits + MARKER + char   →  run
             MARKER + MARKER                          →  literal marker
             anything else                            →  literal

    Returns either an error string or (segments, total_output_length).
    """

    M = RLE_RUN_MARKER
    segments: List[Tuple[bool, Union[str, Tuple[int, str]]]] = []
    total_output_length = 0
    i = 0
    n = len(compressed_text)

    while i < n:
        char = compressed_text[i]

        if char == M:
            i += 1
            if i >= n:
                # Trailing orphan marker
                segments.append((False, M))
                total_output_length += 1
                break

            next_char = compressed_text[i]

            if next_char == M:
                # Escaped marker → literal marker char
                segments.append((False, M))
                total_output_length += 1
                i += 1

            elif next_char.isdigit():
                # Run: MARKER + digits + MARKER + char
                count_str = ""
                digit_count = 0
                while i < n and compressed_text[i].isdigit() and digit_count < MAX_COUNT_STRING_DIGITS:
                    count_str += compressed_text[i]
                    i += 1
                    digit_count += 1

                if digit_count == MAX_COUNT_STRING_DIGITS and i < n and compressed_text[i].isdigit():
                    return "ERROR_COUNT_TOO_LARGE_FOR_SAFETY"

                # Expect closing marker
                if i >= n or compressed_text[i] != M:
                    return "ERROR_MALFORMED_RLE_STRING"
                i += 1  # consume closing marker

                # Expect char to repeat
                if i >= n:
                    return "ERROR_INVALID_RLE_FORMAT_NO_CHAR_AFTER_COUNT"
                char_to_repeat = compressed_text[i]
                i += 1

                try:
                    parsed_count = int(count_str)
                except ValueError:
                    return "ERROR_INVALID_RLE_FORMAT_BAD_COUNT"

                if parsed_count > ABSOLUTE_MAX_PARSED_COUNT:
                    return "ERROR_COUNT_TOO_LARGE_FOR_SAFETY"

                total_output_length += parsed_count
                if total_output_length > ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE:
                    return "ERROR_TOTAL_SIZE_LIMIT_EXCEEDED"

                segments.append((True, (parsed_count, char_to_repeat)))
            else:
                # Marker followed by non-digit, non-marker — tolerate as literal
                segments.append((False, M))
                total_output_length += 1
                # Don't consume next_char; it'll be handled next iteration

            continue

        # Any other character is literal (including digits)
        segments.append((False, char))
        total_output_length += 1
        i += 1

        if total_output_length > ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE:
            return "ERROR_TOTAL_SIZE_LIMIT_EXCEEDED"

    return segments, total_output_length


def gpu_accelerated_rle_compress(
    text_data: str,
    method: str = "simple",
    min_run_len_override: int = None,
    use_gpu: bool = True,
    gpu_id: int = 0,
) -> str:
    if not use_gpu or not CUPY_AVAILABLE:
        logger.debug(
            "GPU RLE compress falling back to CPU (GPU disabled or CuPy unavailable)."
        )
        return cpu_rle_compress(text_data, method=method, min_run_len_override=min_run_len_override)

    if method != "simple":
        logger.debug(
            "GPU RLE compress currently supports only the 'simple' method. Falling back to CPU."
        )
        return cpu_rle_compress(text_data, method=method, min_run_len_override=min_run_len_override)

    min_run = min_run_len_override if min_run_len_override is not None else MIN_ENCODABLE_RUN_LENGTH
    if min_run < 1:
        min_run = 1

    if not isinstance(text_data, str):
        raise TypeError("Input data for GPU RLE compression must be a string.")

    if not text_data:
        return ""

    try:
        host_codepoints = _encode_string_to_codepoints(text_data)
        with cp.cuda.Device(gpu_id):
            workspace_view = _acquire_workspace_slice(gpu_id, host_codepoints.size)
            if workspace_view is not None:
                workspace_view[:host_codepoints.size] = host_codepoints
                gpu_codepoints = workspace_view
            else:
                gpu_codepoints = cp.asarray(host_codepoints)

            run_lengths_gpu, run_values_gpu = _build_run_boundaries_gpu(gpu_codepoints)
            run_lengths = cp.asnumpy(run_lengths_gpu)
            run_values = cp.asnumpy(run_values_gpu)
    except CuPyOutOfMemoryError:
        logger.error("GPU RLE compress ran out of memory; falling back to CPU implementation.")
        return cpu_rle_compress(text_data, method=method, min_run_len_override=min_run_len_override)
    except Exception as exc:
        logger.error(
            "GPU RLE compress encountered an unexpected error. Falling back to CPU.",
            exc_info=True,
        )
        return cpu_rle_compress(text_data, method=method, min_run_len_override=min_run_len_override)

    M = RLE_RUN_MARKER
    result_parts: List[str] = []
    for run_length, run_value in zip(run_lengths, run_values):
        char = chr(int(run_value))
        rl = int(run_length)
        if rl >= min_run:
            # Framed run: MARKER + count + MARKER + char
            result_parts.append(M)
            result_parts.append(str(rl))
            result_parts.append(M)
            result_parts.append(char)
        else:
            # Literal — escape any marker chars
            for _ in range(rl):
                if char == M:
                    result_parts.append(M)
                    result_parts.append(M)
                else:
                    result_parts.append(char)

    return "".join(result_parts)


def gpu_accelerated_rle_decompress(compressed_text_data: str, method: str = "simple",
                                   min_run_len_override: int = None, expected_output_size_hint: int = 0,
                                   use_gpu: bool = True, gpu_id: int = 0) -> str:
    if not use_gpu or not CUPY_AVAILABLE:
        logger.debug(
            "GPU RLE decompress falling back to CPU (GPU disabled or CuPy unavailable)."
        )
        return cpu_rle_decompress(compressed_text_data, method=method, min_run_len_override=min_run_len_override)

    if method != "simple":
        logger.debug(
            "GPU RLE decompress currently supports only the 'simple' method. Falling back to CPU."
        )
        return cpu_rle_decompress(compressed_text_data, method=method, min_run_len_override=min_run_len_override)

    if not isinstance(compressed_text_data, str):
        raise TypeError("Input data for GPU RLE decompression must be a string.")

    if not compressed_text_data:
        return ""

    parsed_segments = _parse_compressed_segments(compressed_text_data)
    if isinstance(parsed_segments, str):
        if parsed_segments in RLE_DECOMPRESSION_ERRORS:
            return parsed_segments
        return "ERROR_MALFORMED_RLE_STRING"

    segments, total_length = parsed_segments
    if total_length == 0:
        return ""

    try:
        with cp.cuda.Device(gpu_id):
            workspace_view = _acquire_workspace_slice(gpu_id, total_length)
            gpu_result = None

            if workspace_view is not None:
                combined_gpu = workspace_view[:total_length]
                offset = 0
                for is_run, payload in segments:
                    if is_run:
                        count, char = payload  # type: ignore[misc]
                        try:
                            count_int = int(count)
                        except (TypeError, ValueError):
                            return "ERROR_INVALID_RLE_FORMAT_BAD_COUNT"

                        if count_int <= 0:
                            continue

                        end_idx = min(combined_gpu.size, offset + count_int)
                        combined_gpu[offset:end_idx] = ord(char)
                        offset = end_idx
                    else:
                        literal = payload  # type: ignore[misc]
                        if not literal:
                            continue
                        literal_codes = _encode_string_to_codepoints(literal)
                        if literal_codes.size == 0:
                            continue
                        end_idx = min(combined_gpu.size, offset + literal_codes.size)
                        slice_len = end_idx - offset
                        if slice_len <= 0:
                            continue
                        combined_gpu[offset:end_idx] = literal_codes[:slice_len]
                        offset = end_idx

                if offset < total_length:
                    logger.warning(
                        "GPU RLE workspace fill undershot expected output (expected %s, wrote %s).",
                        total_length,
                        offset,
                    )
                    gpu_result = combined_gpu[:offset]
                else:
                    gpu_result = combined_gpu
            else:
                gpu_segments: List = []
                for is_run, payload in segments:
                    if is_run:
                        count, char = payload  # type: ignore[misc]
                        try:
                            count_int = int(count)
                        except (TypeError, ValueError):
                            return "ERROR_INVALID_RLE_FORMAT_BAD_COUNT"

                        if count_int <= 0:
                            continue

                        char_code = ord(char)
                        gpu_segments.append(
                            cp.repeat(cp.array([char_code], dtype=cp.uint32), count_int)
                        )
                    else:
                        literal = payload  # type: ignore[misc]
                        if not literal:
                            continue
                        literal_codes = _encode_string_to_codepoints(literal)
                        if literal_codes.size == 0:
                            continue
                        gpu_segments.append(cp.asarray(literal_codes))

                if not gpu_segments:
                    return ""

                gpu_result = cp.concatenate(gpu_segments)
                if gpu_result.size != total_length:
                    logger.warning(
                        "GPU RLE decompress size mismatch (expected %s, got %s).",
                        total_length,
                        gpu_result.size,
                    )

            if gpu_result is None or gpu_result.size == 0:
                return ""

            host_codepoints = cp.asnumpy(gpu_result)
    except CuPyOutOfMemoryError:
        logger.error("GPU RLE decompress ran out of memory; falling back to CPU implementation.")
        return cpu_rle_decompress(compressed_text_data, method=method, min_run_len_override=min_run_len_override)
    except Exception:
        logger.error(
            "GPU RLE decompress encountered an unexpected error. Falling back to CPU.",
            exc_info=True,
        )
        return cpu_rle_decompress(compressed_text_data, method=method, min_run_len_override=min_run_len_override)

    try:
        if expected_output_size_hint and host_codepoints.size != expected_output_size_hint:
            logger.debug(
                "GPU RLE decompress output size (%s) differs from expected hint (%s).",
                host_codepoints.size,
                expected_output_size_hint,
            )
        return _codepoints_to_string(host_codepoints)
    except MemoryError:
        return "ERROR_MEMORY_DURING_FINAL_JOIN"

if __name__ == '__main__':
    print("--- Testing GPU RLE Interface ---")
    print(f"CuPy Available: {CUPY_AVAILABLE}")
    print(f"Numba CUDA Available: {NUMBA_AVAILABLE}")

    sample = "AAAAABBBCCCDDDDEEEEE``XYZ"
    print(f"\nOriginal: {sample}")

    compressed_gpu = gpu_accelerated_rle_compress(sample, method="simple", use_gpu=True)
    print(f"Compressed (GPU attempt): {compressed_gpu}")

    decompressed_gpu = gpu_accelerated_rle_decompress(compressed_gpu, method="simple", use_gpu=True)
    print(f"Decompressed (GPU attempt): {decompressed_gpu}")

    if decompressed_gpu == sample:
        print("GPU interface round-trip: PASS")
    else:
        print("GPU interface round-trip: FAIL")

