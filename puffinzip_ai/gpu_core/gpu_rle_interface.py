# PuffinZipAI_Project/puffinzip_ai/gpu_core/gpu_rle_interface.py
import logging
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
compress_kernel = None
decompress_kernel = None

try:
    from ..rle_utils import (
        rle_compress as cpu_rle_compress,
        rle_decompress as cpu_rle_decompress,
        RLE_DECOMPRESSION_ERRORS,
        MIN_ENCODABLE_RUN_LENGTH,
        ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE
    )
except ImportError:
    def cpu_rle_compress(d, **k): return "ERROR"
    def cpu_rle_decompress(d, **k): return "ERROR"
    RLE_DECOMPRESSION_ERRORS = set()
    MIN_ENCODABLE_RUN_LENGTH = 3
    ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE = 200 * 1024 * 1024

# Raw C++ CUDA Code - Unicode Safe (uint32_t) Batched Global Memory processing
CUDA_RLE_CODE = r'''
extern "C" {

__global__ void batchedRleCompressKernel(
    const unsigned int* inputData, const int* inputOffsets, const int* inputLengths,
    unsigned int* outputData, const int* maxOutputLens, const int* outputOffsets,
    int* actualOutputLens, const int* minRunLengths, int numItems)
{
    int itemIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (itemIdx >= numItems) return;

    int length = inputLengths[itemIdx];
    int offset = inputOffsets[itemIdx];
    int maxOut = maxOutputLens[itemIdx];
    int minRun = minRunLengths[itemIdx];

    if (length == 0) {
        actualOutputLens[itemIdx] = 0;
        return;
    }

    const unsigned int* in = inputData + offset;
    unsigned int* out = outputData + outputOffsets[itemIdx];

    int outIdx = 0;
    unsigned int currentChar = in[0];
    int count = 1;
    const unsigned int MARKER = 2; // '\x02'

    for (int i = 1; i < length; ++i) {
        if (in[i] == currentChar) {
            count++;
        } else {
            if (count >= minRun) {
                if (outIdx + 15 > maxOut) { actualOutputLens[itemIdx] = -1; return; }
                out[outIdx++] = MARKER;
                // Convert integer count to string chars inline
                unsigned int buf[10]; int len = 0; int temp = count;
                do { buf[len++] = 48 + (temp % 10); temp /= 10; } while(temp > 0);
                for(int k = len - 1; k >= 0; k--) out[outIdx++] = buf[k];
                out[outIdx++] = MARKER;
                out[outIdx++] = currentChar;
            } else {
                for (int k = 0; k < count; k++) {
                    if (outIdx + 2 > maxOut) { actualOutputLens[itemIdx] = -1; return; }
                    if (currentChar == MARKER) { out[outIdx++] = MARKER; out[outIdx++] = MARKER; }
                    else { out[outIdx++] = currentChar; }
                }
            }
            currentChar = in[i];
            count = 1;
        }
    }

    if (count > 0) {
        if (count >= minRun) {
            if (outIdx + 15 > maxOut) { actualOutputLens[itemIdx] = -1; return; }
            out[outIdx++] = MARKER;
            unsigned int buf[10]; int len = 0; int temp = count;
            do { buf[len++] = 48 + (temp % 10); temp /= 10; } while(temp > 0);
            for(int k = len - 1; k >= 0; k--) out[outIdx++] = buf[k];
            out[outIdx++] = MARKER;
            out[outIdx++] = currentChar;
        } else {
            for (int k = 0; k < count; k++) {
                if (outIdx + 2 > maxOut) { actualOutputLens[itemIdx] = -1; return; }
                if (currentChar == MARKER) { out[outIdx++] = MARKER; out[outIdx++] = MARKER; }
                else { out[outIdx++] = currentChar; }
            }
        }
    }
    actualOutputLens[itemIdx] = outIdx;
}

__global__ void batchedRleDecompressKernel(
    const unsigned int* inputData, const int* inputOffsets, const int* inputLengths,
    unsigned int* outputData, const int* maxOutputLens, const int* outputOffsets,
    int* actualOutputLens, int numItems)
{
    int itemIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (itemIdx >= numItems) return;

    int length = inputLengths[itemIdx];
    int offset = inputOffsets[itemIdx];
    int maxOut = maxOutputLens[itemIdx];

    if (length == 0) {
        actualOutputLens[itemIdx] = 0;
        return;
    }

    const unsigned int* in = inputData + offset;
    unsigned int* out = outputData + outputOffsets[itemIdx];

    const unsigned int MARKER = 2;
    int inIdx = 0;
    int outIdx = 0;
    bool error = false;

    while (inIdx < length && !error) {
        unsigned int c = in[inIdx++];
        if (c == MARKER) {
            if (inIdx >= length) { error = true; break; }
            unsigned int next_c = in[inIdx];
            if (next_c == MARKER) {
                if (outIdx < maxOut) out[outIdx++] = MARKER; else { error = true; break; }
                inIdx++;
            } else if (next_c >= 48 && next_c <= 57) { // '0'-'9'
                int count = 0;
                int digit_count = 0;
                while (inIdx < length && in[inIdx] >= 48 && in[inIdx] <= 57 && digit_count < 10) {
                    count = count * 10 + (in[inIdx] - 48);
                    inIdx++;
                    digit_count++;
                }
                if (inIdx >= length || in[inIdx] != MARKER) { error = true; break; }
                inIdx++;
                if (inIdx >= length) { error = true; break; }
                unsigned int char_to_repeat = in[inIdx++];
                
                if (outIdx + count > maxOut) { error = true; break; }
                for (int k = 0; k < count; k++) {
                    out[outIdx++] = char_to_repeat;
                }
            } else {
                if (outIdx < maxOut) out[outIdx++] = MARKER; else { error = true; break; }
            }
        } else {
            if (outIdx < maxOut) out[outIdx++] = c;
            else { error = true; break; }
        }
    }
    actualOutputLens[itemIdx] = error ? -1 : outIdx;
}

} // extern "C"
'''

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    rle_module = cp.RawModule(code=CUDA_RLE_CODE)
    compress_kernel = rle_module.get_function("batchedRleCompressKernel")
    decompress_kernel = rle_module.get_function("batchedRleDecompressKernel")
    logger.debug("Successfully compiled Raw C++ CUDA Kernels.")
except ImportError:
    pass
except Exception as e:
    logger.warning(f"Failed to compile CUDA kernels: {e}")
    CUPY_AVAILABLE = False


def _encode_to_uint32(texts):
    """Safely converts Python strings to continuous UTF-32 1D numpy arrays."""
    joined = "".join(texts)
    if not joined:
        return np.array([], dtype=np.uint32), [], []
    arr = np.frombuffer(joined.encode("utf-32-le"), dtype=np.uint32)
    lengths = [len(t) for t in texts]
    offsets = np.zeros(len(texts), dtype=np.int32)
    current = 0
    for i, l in enumerate(lengths):
        offsets[i] = current
        current += l
    return arr, offsets, np.array(lengths, dtype=np.int32)


def gpu_accelerated_rle_compress_batch(texts, min_runs, gpu_id=0):
    if not CUPY_AVAILABLE:
        return [cpu_rle_compress(t, method="simple", min_run_len_override=mr) for t, mr in zip(texts, min_runs)]

    num_items = len(texts)
    if num_items == 0: return []

    arr, offsets, lengths = _encode_to_uint32(texts)
    # Max expansion buffer: 2x the size in worst case (e.g. escaping markers)
    max_outs = lengths * 2 + 16
    # Exclusive prefix sum: per-item start offset into the flat output buffer.
    # Must match the host's cumulative read below and the input-side layout —
    # items have variable max_outs, so a fixed itemIdx*maxOut stride is wrong.
    output_offsets = np.zeros(num_items, dtype=np.int32)
    if num_items > 1:
        np.cumsum(max_outs[:-1], out=output_offsets[1:])

    try:
        with cp.cuda.Device(gpu_id):
            d_in = cp.asarray(arr)
            d_offsets = cp.asarray(offsets)
            d_lengths = cp.asarray(lengths)
            d_min_runs = cp.asarray(np.array(min_runs, dtype=np.int32))
            d_max_outs = cp.asarray(max_outs)
            d_output_offsets = cp.asarray(output_offsets)
            d_actual_lens = cp.zeros(num_items, dtype=cp.int32)

            total_out_size = int(np.sum(max_outs))
            d_out = cp.empty(total_out_size, dtype=cp.uint32)

            threads_per_block = 256
            blocks = (num_items + threads_per_block - 1) // threads_per_block

            compress_kernel((blocks,), (threads_per_block,),
                            (d_in, d_offsets, d_lengths, d_out, d_max_outs, d_output_offsets, d_actual_lens, d_min_runs, num_items))

            h_actual_lens = d_actual_lens.get()
            h_out = d_out.get()

        results = []
        out_offset = 0
        for i in range(num_items):
            act_len = h_actual_lens[i]
            max_o = max_outs[i]
            if act_len == -1:
                results.append("ERROR_GPU_RLE_MEMORY")
            else:
                res_arr = h_out[out_offset : out_offset + act_len]
                results.append(res_arr.tobytes().decode('utf-32-le'))
            out_offset += max_o
        return results
    except cp.cuda.memory.OutOfMemoryError:
        logger.warning("GPU OOM during batched compress. Falling back to CPU multiprocessing.")
        return [cpu_rle_compress(t, method="simple", min_run_len_override=mr) for t, mr in zip(texts, min_runs)]
    except Exception as e:
        logger.error(f"GPU batch compression failed: {e}")
        return [cpu_rle_compress(t, method="simple", min_run_len_override=mr) for t, mr in zip(texts, min_runs)]


def gpu_accelerated_rle_decompress_batch(compressed_texts, expected_lengths, gpu_id=0):
    if not CUPY_AVAILABLE:
        return [cpu_rle_decompress(t, method="simple") for t in compressed_texts]

    num_items = len(compressed_texts)
    if num_items == 0: return []

    arr, offsets, lengths = _encode_to_uint32(compressed_texts)

    # Use original input sizes to guide expected decompressed lengths safely
    max_outs = np.clip(np.array(expected_lengths, dtype=np.int32) + 32, 0, ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE)
    # Exclusive prefix sum: per-item start offset into the flat output buffer
    # (variable max_outs → cumulative layout, matching the host read below).
    output_offsets = np.zeros(num_items, dtype=np.int32)
    if num_items > 1:
        np.cumsum(max_outs[:-1], out=output_offsets[1:])

    try:
        with cp.cuda.Device(gpu_id):
            d_in = cp.asarray(arr)
            d_offsets = cp.asarray(offsets)
            d_lengths = cp.asarray(lengths)
            d_max_outs = cp.asarray(max_outs)
            d_output_offsets = cp.asarray(output_offsets)
            d_actual_lens = cp.zeros(num_items, dtype=cp.int32)

            total_out_size = int(np.sum(max_outs))
            d_out = cp.empty(total_out_size, dtype=cp.uint32)

            threads_per_block = 256
            blocks = (num_items + threads_per_block - 1) // threads_per_block

            decompress_kernel((blocks,), (threads_per_block,),
                            (d_in, d_offsets, d_lengths, d_out, d_max_outs, d_output_offsets, d_actual_lens, num_items))

            h_actual_lens = d_actual_lens.get()
            h_out = d_out.get()

        results = []
        out_offset = 0
        for i in range(num_items):
            act_len = h_actual_lens[i]
            max_o = max_outs[i]
            if act_len == -1:
                results.append("ERROR_MALFORMED_RLE_STRING")
            else:
                res_arr = h_out[out_offset : out_offset + act_len]
                results.append(res_arr.tobytes().decode('utf-32-le'))
            out_offset += max_o
        return results
    except cp.cuda.memory.OutOfMemoryError:
        logger.warning("GPU OOM during batched decompress. Falling back to CPU.")
        return [cpu_rle_decompress(t, method="simple") for t in compressed_texts]
    except Exception as e:
        logger.error(f"GPU batch decompression failed: {e}")
        return [cpu_rle_decompress(t, method="simple") for t in compressed_texts]


# Maintain backwards compatibility for single-item GUI processing
def gpu_accelerated_rle_compress(text_data: str, method: str = "simple", min_run_len_override: int = None, use_gpu: bool = True, gpu_id: int = 0) -> str:
    if method != "simple" or not use_gpu:
        return cpu_rle_compress(text_data, method=method, min_run_len_override=min_run_len_override)
    mr = min_run_len_override if min_run_len_override else 3
    return gpu_accelerated_rle_compress_batch([text_data], [mr], gpu_id)[0]

def gpu_accelerated_rle_decompress(compressed_text_data: str, method: str = "simple", min_run_len_override: int = None, expected_output_size_hint: int = 0, use_gpu: bool = True, gpu_id: int = 0) -> str:
    if method != "simple" or not use_gpu:
        return cpu_rle_decompress(compressed_text_data, method=method, min_run_len_override=min_run_len_override)
    sz = expected_output_size_hint if expected_output_size_hint > 0 else len(compressed_text_data) * 5
    return gpu_accelerated_rle_decompress_batch([compressed_text_data], [sz], gpu_id)[0]