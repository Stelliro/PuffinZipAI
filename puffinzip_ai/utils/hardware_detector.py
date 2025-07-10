# PuffinZipAI_Project/puffinzip_ai/utils/hardware_detector.py
import logging
import platform
import subprocess
import re
import traceback

try:
    from puffinzip_ai.gpu_core.gpu_ai_agent import cuda
except ImportError:
    cuda = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# --- Enhanced GPU Library Checks ---
# More detailed check for CuPy
try:
    import cupy as cp
    # Test if CuPy can actually communicate with the driver
    try:
        if cp.cuda.runtime.getDeviceCount() >= 0:
            CUPY_AVAILABLE = True
        else: # Should not happen, getDeviceCount raises on error
             CUPY_AVAILABLE = False
    except cp.cuda.runtime.CUDARuntimeError as e_runtime:
        # This is a common error if CUDA toolkit/driver is missing or incompatible
        logging.getLogger("puffinzip_ai.hardware_detector").warning(f"CuPy is installed, but a CUDA runtime error occurred: {e_runtime}. This often means a missing or incompatible CUDA driver/toolkit. CuPy will be disabled.")
        CUPY_AVAILABLE = False
    except Exception as e_cupy_init:
        logging.getLogger("puffinzip_ai.hardware_detector").warning(f"CuPy is installed, but failed to initialize: {e_cupy_init}. CuPy will be disabled.")
        CUPY_AVAILABLE = False
except ImportError:
    # This is fine, just means CuPy is not installed.
    CUPY_AVAILABLE = False
except Exception as e_cupy_imp:
    # This is more unusual
    logging.getLogger("puffinzip_ai.hardware_detector").error(f"An unexpected error occurred during CuPy import: {e_cupy_imp}", exc_info=True)
    CUPY_AVAILABLE = False


# More detailed check for Numba
try:
    from numba import cuda as nb_cuda
    try:
        if nb_cuda.is_available():
            NUMBA_CUDA_AVAILABLE = True
            if cuda is None: cuda = nb_cuda
        else:
            # is_available() returns False if no GPU is found or driver is bad.
            # The reason is often logged by Numba itself internally.
            logging.getLogger("puffinzip_ai.hardware_detector").info("Numba CUDA is installed, but `numba.cuda.is_available()` returned False. No Numba GPU will be detected.")
            NUMBA_CUDA_AVAILABLE = False
    except Exception as e_numba_init:
        logging.getLogger("puffinzip_ai.hardware_detector").warning(f"Numba is installed, but failed on `is_available()` check: {e_numba_init}. Numba CUDA will be disabled.")
        NUMBA_CUDA_AVAILABLE = False
except ImportError:
    NUMBA_CUDA_AVAILABLE = False
except Exception as e_numba_imp:
    logging.getLogger("puffinzip_ai.hardware_detector").error(f"An unexpected error occurred during Numba import: {e_numba_imp}", exc_info=True)
    NUMBA_CUDA_AVAILABLE = False
# --- End Enhanced GPU Library Checks ---

logger = logging.getLogger("puffinzip_ai.hardware_detector")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.NullHandler())

def get_cpu_info():
    logger.debug("Detecting CPU info...")
    try:
        if platform.system() == "Windows":
            if PSUTIL_AVAILABLE:
                pass
            try:
                process = subprocess.Popen(
                    ['wmic', 'cpu', 'get', 'name'],
                    stdout=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW
                )
                stdout, _ = process.communicate(timeout=5)
                if process.returncode == 0 and stdout:
                    lines = stdout.strip().splitlines()
                    if len(lines) > 1:
                        cpu_name = lines[1].strip()
                        logger.info(f"CPU Info (WMIC): {cpu_name}")
                        return f"CPU: {cpu_name}"
            except Exception as e_wmic:
                logger.debug(f"WMIC CPU detection failed: {e_wmic}. Trying platform.processor().")
        elif platform.system() == "Linux":
            try:
                process = subprocess.Popen(
                    ['lscpu'], stdout=subprocess.PIPE, text=True
                )
                stdout, _ = process.communicate(timeout=5)
                if process.returncode == 0 and stdout:
                    match = re.search(r"Model name:\s*(.*)", stdout)
                    if match:
                        cpu_name = match.group(1).strip()
                        logger.info(f"CPU Info (lscpu): {cpu_name}")
                        return f"CPU: {cpu_name}"
            except Exception as e_lscpu:
                logger.debug(f"lscpu CPU detection failed: {e_lscpu}. Trying platform.processor().")
        elif platform.system() == "Darwin":
            try:
                cpu_name = subprocess.check_output(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    text=True, timeout=2
                ).strip()
                if cpu_name:
                    logger.info(f"CPU Info (sysctl): {cpu_name}")
                    return f"CPU: {cpu_name}"
            except Exception as e_sysctl:
                logger.debug(f"sysctl CPU detection failed: {e_sysctl}. Trying platform.processor().")
        processor_name = platform.processor()
        if processor_name:
            logger.info(f"CPU Info (platform.processor): {processor_name}")
            return f"CPU: {processor_name}"
        else:
            logger.warning("CPU detection: platform.processor() returned empty. Using generic CPU string.")
            return "CPU (Generic)"
    except Exception as e:
        logger.error(f"Generic error detecting CPU info: {e}", exc_info=True)
        return "CPU (Detection Error)"

def get_available_gpus_info():
    logger.debug(f"Detecting GPU info... (CuPy available: {CUPY_AVAILABLE}, Numba CUDA available: {NUMBA_CUDA_AVAILABLE})")
    gpus = []
    if CUPY_AVAILABLE:
        try:
            num_cupy_gpus = cp.cuda.runtime.getDeviceCount()
            logger.debug(f"CuPy detected {num_cupy_gpus} CUDA devices.")
            for i in range(num_cupy_gpus):
                try:
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    gpu_name = "Unknown CuPy GPU"
                    if isinstance(props, dict) and 'name' in props:
                        name_val = props['name']
                        gpu_name = name_val.decode().strip() if isinstance(name_val, bytes) else str(name_val).strip()
                    elif hasattr(props, 'name'): # Fallback for older/different CuPy versions
                        name_val = props.name
                        gpu_name = name_val.decode().strip() if isinstance(name_val, bytes) else str(name_val).strip()

                    gpus.append({
                        "id_str": f"GPU_ID:{i}",
                        "name": f"GPU {i}: {gpu_name} (CuPy)",
                        "type": "cupy",
                        "device_id_int": i
                    })
                    logger.info(f"Found CuPy GPU {i}: {gpu_name}")
                except Exception as e_cupy_prop:
                    logger.warning(f"Error getting properties for CuPy GPU ID {i}: {e_cupy_prop}")
        except Exception as e_cupy_count:
            logger.warning(f"Error with CuPy getDeviceCount: {e_cupy_count}")

    if NUMBA_CUDA_AVAILABLE and cuda is not None and hasattr(cuda, 'gpus') and hasattr(cuda.gpus, 'lst'):
        try:
            numba_devices = cuda.gpus.lst
            logger.debug(f"Numba detected {len(numba_devices)} CUDA devices.")
            for i, device in enumerate(numba_devices):
                gpu_name = "Unknown Numba GPU"
                try:
                    gpu_name = device.name.decode().strip() if isinstance(device.name, bytes) else str(device.name).strip()
                except Exception:
                    pass
                already_listed_by_cupy = False
                if CUPY_AVAILABLE:
                    for cupy_gpu in gpus:
                        if cupy_gpu["device_id_int"] == i and (gpu_name in cupy_gpu["name"] or cupy_gpu["name"] in gpu_name):
                            already_listed_by_cupy = True
                            logger.debug(f"Numba GPU {i} ({gpu_name}) seems to be already listed by CuPy as {cupy_gpu['name']}. Skipping Numba entry for this ID.")
                            break
                if not already_listed_by_cupy:
                    gpus.append({
                        "id_str": f"GPU_ID:{i}",
                        "name": f"GPU {i}: {gpu_name} (Numba)",
                        "type": "numba",
                        "device_id_int": i
                    })
                    logger.info(f"Found Numba CUDA GPU {i}: {gpu_name}")
        except Exception as e_numba:
            logger.warning(f"Error with Numba GPU detection: {e_numba}")
    elif NUMBA_CUDA_AVAILABLE and (cuda is None or not hasattr(cuda, 'gpus') or not hasattr(cuda.gpus, 'lst')):
        logger.warning("Numba CUDA available but 'cuda.gpus.lst' attribute not found. Skipping Numba GPU detection.")


    if not gpus:
        logger.info("No CUDA-capable GPUs detected by CuPy or Numba.")
    return gpus

def get_processing_device_options():
    options = []
    cpu_display_name = get_cpu_info()
    options.append((cpu_display_name, "CPU"))
    detected_gpus = get_available_gpus_info()
    if detected_gpus:
        options.append(("GPU (Auto-Select Best Available)", "GPU_AUTO"))
        for gpu in detected_gpus:
            options.append((gpu["name"], gpu["id_str"]))
    logger.info(f"Generated processing device options for GUI: {options}")
    return options

if __name__ == "__main__":
    if not logger.handlers or isinstance(logger.handlers[0], logging.NullHandler):
        logger.handlers.clear()
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        logger.addHandler(ch)
        logger.setLevel(logging.DEBUG)
    print("--- Testing Hardware Detector ---")
    print("\nCPU Info:")
    cpu_info = get_cpu_info()
    print(f"  {cpu_info}")
    print("\nGPU Info:")
    gpus = get_available_gpus_info()
    if gpus:
        for gpu in gpus:
            print(f"  ID String: {gpu['id_str']}, Name: {gpu['name']}, Type: {gpu['type']}")
    else:
        print("  No compatible GPUs detected.")
    print("\nProcessing Device Options for GUI Combobox:")
    gui_options = get_processing_device_options()
    for display, value in gui_options:
        print(f"  Display: \"{display}\", Value: \"{value}\"")
    print("\n--- Hardware Detector Test Finished ---")