# PuffinZipAI_Project/puffinzip_ai/gpu_core/gpu_model_utils.py
import logging

CUPY_AVAILABLE = False
cp = None
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    # print("INFO (gpu_model_utils.py): CuPy library found and imported.") # Comment removed
except ImportError:
    # print("INFO (gpu_model_utils.py): CuPy library not found. GPU model utilities will be limited.") # Comment removed
    CUPY_AVAILABLE = False

logger = logging.getLogger("PuffinZipAI_GPU_ModelUtils")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def get_gpu_memory_info(gpu_id=0):
    if not CUPY_AVAILABLE:
        logger.debug("get_gpu_memory_info: CuPy not available.")
        return None
    try:
        with cp.cuda.Device(gpu_id):
            mem_info = cp.cuda.runtime.memGetInfo()
            free_memory_bytes = mem_info[0]
            total_memory_bytes = mem_info[1]
            used_memory_bytes = total_memory_bytes - free_memory_bytes
            return {
                "total_bytes": total_memory_bytes,
                "free_bytes": free_memory_bytes,
                "used_bytes": used_memory_bytes
            }
    except Exception as e:
        logger.error(f"Error getting GPU memory info for device {gpu_id}: {e}", exc_info=True)
        return None

def array_to_gpu(np_array, gpu_id=0):
    if not CUPY_AVAILABLE:
        logger.debug("array_to_gpu: CuPy not available. Cannot transfer array.")
        return None
    if np_array is None:
        logger.warning("array_to_gpu: Input NumPy array is None.")
        return None
    try:
        with cp.cuda.Device(gpu_id):
            gpu_array = cp.asarray(np_array)
        logger.debug(f"Array (shape: {np_array.shape}, dtype: {np_array.dtype}) transferred to GPU {gpu_id}.")
        return gpu_array
    except Exception as e:
        logger.error(f"Error transferring NumPy array to GPU {gpu_id}: {e}", exc_info=True)
        return None

def array_to_cpu(gpu_array):
    if not CUPY_AVAILABLE:
        logger.debug("array_to_cpu: CuPy not available.")
        return None
    if gpu_array is None:
        logger.warning("array_to_cpu: Input GPU array is None.")
        return None
    if not isinstance(gpu_array, cp.ndarray):
        logger.warning(f"array_to_cpu: Input is not a CuPy array (type: {type(gpu_array)}). Cannot transfer.")
        return None

    try:
        np_array = cp.asnumpy(gpu_array)
        logger.debug(f"CuPy array (shape: {gpu_array.shape}, dtype: {gpu_array.dtype}) transferred to CPU.")
        return np_array
    except Exception as e:
        logger.error(f"Error transferring CuPy array to CPU: {e}", exc_info=True)
        return None

def get_best_available_gpu_id():
    if not CUPY_AVAILABLE:
        return -1
    try:
        num_gpus = cp.cuda.runtime.getDeviceCount()
        if num_gpus > 0:
            logger.debug(f"get_best_available_gpu_id: Found {num_gpus} GPU(s). Returning ID 0.")
            return 0
        else:
            logger.debug("get_best_available_gpu_id: No CuPy-compatible GPUs found.")
            return -1
    except Exception as e:
        logger.error(f"Error detecting GPU for get_best_available_gpu_id: {e}", exc_info=True)
        return -1

if __name__ == "__main__":
    import numpy as np
    logger.setLevel(logging.DEBUG)
    print("\n--- Testing GPU Model Utilities ---")

    if CUPY_AVAILABLE:
        print("\n--- GPU Memory Info Test ---")
        device_id_to_test = 0
        try:
            count = cp.cuda.runtime.getDeviceCount()
            if count == 0:
                print(f"No CUDA devices found by CuPy. Skipping memory info test for device {device_id_to_test}.")
            else:
                if device_id_to_test >= count :
                    print(f"Test device ID {device_id_to_test} is out of range (found {count} devices). Defaulting to 0.")
                    device_id_to_test = 0

                mem_info = get_gpu_memory_info(gpu_id=device_id_to_test)
                if mem_info:
                    print(f"GPU {device_id_to_test} Memory (MB):")
                    print(f"  Total: {mem_info['total_bytes'] / (1024**2):.2f} MB")
                    print(f"  Free:  {mem_info['free_bytes'] / (1024**2):.2f} MB")
                    print(f"  Used:  {mem_info['used_bytes'] / (1024**2):.2f} MB")
                else:
                    print(f"Could not get memory info for GPU {device_id_to_test}.")
        except Exception as e_mem_test:
             print(f"Error during memory test: {e_mem_test}")

        print("\n--- Array Transfer Test (NumPy <-> CuPy) ---")
        np_data = np.random.rand(1000, 1000).astype(np.float32)
        print(f"Original NumPy array created. Shape: {np_data.shape}, Dtype: {np_data.dtype}")

        gpu_data = array_to_gpu(np_data, gpu_id=device_id_to_test)
        if gpu_data is not None:
            print(f"Array transferred to GPU. Shape: {gpu_data.shape}, Dtype: {gpu_data.dtype}")
            gpu_data_sum = cp.sum(gpu_data)
            print(f"Sum of GPU array (calculated on GPU): {gpu_data_sum}")

            np_data_check = array_to_cpu(gpu_data)
            if np_data_check is not None:
                print(f"Array transferred back to CPU. Shape: {np_data_check.shape}, Dtype: {np_data_check.dtype}")
                np_data_sum_original = np.sum(np_data)
                np_data_sum_check = np.sum(np_data_check)
                print(f"Sum of original NumPy array: {np_data_sum_original}")
                print(f"Sum of NumPy array after GPU roundtrip: {np_data_sum_check}")
                if np.allclose(np_data_sum_original, np_data_sum_check):
                    print("Array sums match after roundtrip: PASS")
                else:
                    print("Array sums MISMATCH after roundtrip: FAIL")
                if np.allclose(np_data, np_data_check):
                     print("Full array content matches after roundtrip: PASS")
                else:
                     print("Full array content MISMATCH after roundtrip: FAIL")
            del gpu_data
            if CUPY_AVAILABLE: cp.cuda.Stream.null.synchronize()
            print("GPU data deleted.")
        else:
            print("Failed to transfer array to GPU.")
    else:
        print("CuPy not available. Skipping GPU-specific tests.")

    print("\n--- Testing get_best_available_gpu_id ---")
    best_gpu = get_best_available_gpu_id()
    if best_gpu != -1:
        print(f"Best available GPU ID determined as: {best_gpu}")
    else:
        print("No suitable GPU found or CuPy not available.")

    print("\n--- GPU Model Utilities Test Finished ---")