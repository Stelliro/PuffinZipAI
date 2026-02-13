# PuffinZipAI_Project/puffinzip_ai/rle_utils.py
import logging
import time

rle_logger = logging.getLogger("puffinzip_ai.rle_utils")
if not rle_logger.handlers:
    rle_logger.setLevel(logging.WARNING)
    rle_logger.addHandler(logging.NullHandler())

RLE_ERROR_NO_COUNT = "ERROR_INVALID_RLE_FORMAT_NO_COUNT"
RLE_ERROR_BAD_COUNT = "ERROR_INVALID_RLE_FORMAT_BAD_COUNT"
RLE_ERROR_NO_CHAR = "ERROR_INVALID_RLE_FORMAT_NO_CHAR_AFTER_COUNT"
RLE_ERROR_MALFORMED = "ERROR_MALFORMED_RLE_STRING"
RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY = "ERROR_COUNT_TOO_LARGE_FOR_SAFETY"
RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED = "ERROR_TOTAL_SIZE_LIMIT_EXCEEDED"
RLE_ERROR_MEMORY_ON_CHUNK = "ERROR_MEMORY_DURING_CHUNK_ALLOCATION"
RLE_ERROR_MEMORY_ON_JOIN = "ERROR_MEMORY_DURING_FINAL_JOIN"
RLE_DECOMPRESSION_ERRORS = {
    RLE_ERROR_NO_COUNT, RLE_ERROR_BAD_COUNT, RLE_ERROR_NO_CHAR, RLE_ERROR_MALFORMED,
    RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY, RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED,
    RLE_ERROR_MEMORY_ON_CHUNK, RLE_ERROR_MEMORY_ON_JOIN,
}
ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE = 200 * 1024 * 1024
ABSOLUTE_MAX_PARSED_COUNT = 100 * 1024 * 1024
MAX_COUNT_STRING_DIGITS = 9
RLE_DELIMITER = '`'  # New unambiguous format delimiter

_constants_imported_successfully = False
try:
    from .rle_constants import (
        RLE_ERROR_NO_COUNT as RC_RLE_ERROR_NO_COUNT,
        RLE_ERROR_BAD_COUNT as RC_RLE_ERROR_BAD_COUNT,
        RLE_ERROR_NO_CHAR as RC_RLE_ERROR_NO_CHAR,
        RLE_ERROR_MALFORMED as RC_RLE_ERROR_MALFORMED,
        RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY as RC_RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY,
        RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED as RC_RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED,
        RLE_ERROR_MEMORY_ON_CHUNK as RC_RLE_ERROR_MEMORY_ON_CHUNK,
        RLE_ERROR_MEMORY_ON_JOIN as RC_RLE_ERROR_MEMORY_ON_JOIN,
        RLE_DECOMPRESSION_ERRORS as RC_RLE_DECOMPRESSION_ERRORS,
        ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE as RC_ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE,
        ABSOLUTE_MAX_PARSED_COUNT as RC_ABSOLUTE_MAX_PARSED_COUNT,
        MAX_COUNT_STRING_DIGITS as RC_MAX_COUNT_STRING_DIGITS
    )

    RLE_ERROR_NO_COUNT = RC_RLE_ERROR_NO_COUNT
    RLE_ERROR_BAD_COUNT = RC_RLE_ERROR_BAD_COUNT
    RLE_ERROR_NO_CHAR = RC_RLE_ERROR_NO_CHAR
    RLE_ERROR_MALFORMED = RC_RLE_ERROR_MALFORMED
    RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY = RC_RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY
    RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED = RC_RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED
    RLE_ERROR_MEMORY_ON_CHUNK = RC_RLE_ERROR_MEMORY_ON_CHUNK
    RLE_ERROR_MEMORY_ON_JOIN = RC_RLE_ERROR_MEMORY_ON_JOIN
    RLE_DECOMPRESSION_ERRORS = RC_RLE_DECOMPRESSION_ERRORS
    ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE = RC_ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE
    ABSOLUTE_MAX_PARSED_COUNT = RC_ABSOLUTE_MAX_PARSED_COUNT
    MAX_COUNT_STRING_DIGITS = RC_MAX_COUNT_STRING_DIGITS
    _constants_imported_successfully = True
    rle_logger.info("Constants successfully imported into rle_utils from .rle_constants.")
except ImportError:
    rle_logger.warning("Relative import from .rle_constants failed in rle_utils. Trying package-absolute import.")
    try:
        from puffinzip_ai.rle_constants import (
            RLE_ERROR_NO_COUNT as RC_RLE_ERROR_NO_COUNT,
            RLE_ERROR_BAD_COUNT as RC_RLE_ERROR_BAD_COUNT,
            RLE_ERROR_NO_CHAR as RC_RLE_ERROR_NO_CHAR,
            RLE_ERROR_MALFORMED as RC_RLE_ERROR_MALFORMED,
            RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY as RC_RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY,
            RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED as RC_RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED,
            RLE_ERROR_MEMORY_ON_CHUNK as RC_RLE_ERROR_MEMORY_ON_CHUNK,
            RLE_ERROR_MEMORY_ON_JOIN as RC_RLE_ERROR_MEMORY_ON_JOIN,
            RLE_DECOMPRESSION_ERRORS as RC_RLE_DECOMPRESSION_ERRORS,
            ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE as RC_ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE,
            ABSOLUTE_MAX_PARSED_COUNT as RC_ABSOLUTE_MAX_PARSED_COUNT,
            MAX_COUNT_STRING_DIGITS as RC_MAX_COUNT_STRING_DIGITS
        )

        RLE_ERROR_NO_COUNT = RC_RLE_ERROR_NO_COUNT
        RLE_ERROR_BAD_COUNT = RC_RLE_ERROR_BAD_COUNT
        RLE_ERROR_NO_CHAR = RC_RLE_ERROR_NO_CHAR
        RLE_ERROR_MALFORMED = RC_RLE_ERROR_MALFORMED
        RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY = RC_RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY
        RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED = RC_RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED
        RLE_ERROR_MEMORY_ON_CHUNK = RC_RLE_ERROR_MEMORY_ON_CHUNK
        RLE_ERROR_MEMORY_ON_JOIN = RC_RLE_ERROR_MEMORY_ON_JOIN
        RLE_DECOMPRESSION_ERRORS = RC_RLE_DECOMPRESSION_ERRORS
        ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE = RC_ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE
        ABSOLUTE_MAX_PARSED_COUNT = RC_ABSOLUTE_MAX_PARSED_COUNT
        MAX_COUNT_STRING_DIGITS = RC_MAX_COUNT_STRING_DIGITS
        _constants_imported_successfully = True
        rle_logger.info("Constants successfully imported into rle_utils from puffinzip_ai.rle_constants.")
    except ImportError:
        rle_logger.critical(
            "CRITICAL: ALL import attempts for rle_constants failed in rle_utils. Using internal hardcoded fallback constants.")

MIN_ENCODABLE_RUN_LENGTH = 3
THROTTLE_RUN_LENGTH_THRESHOLD = 1 * 1024 * 1024
THROTTLE_CHUNK_SIZE = 256 * 1024
THROTTLE_SLEEP_DURATION = 0.001

_advanced_rle_module_cache = None
_advanced_rle_module_available_cache = None
_adv_module_print_info_done = False

try:
    from .utils import performance_tuner

    _rle_tuned_params = performance_tuner.get_tuned_parameters()
    THROTTLE_RUN_LENGTH_THRESHOLD = _rle_tuned_params.get("RLE_THROTTLE_RUN_LENGTH_THRESHOLD",
                                                          THROTTLE_RUN_LENGTH_THRESHOLD)
    THROTTLE_CHUNK_SIZE = _rle_tuned_params.get("RLE_THROTTLE_CHUNK_SIZE", THROTTLE_CHUNK_SIZE)
    THROTTLE_SLEEP_DURATION = _rle_tuned_params.get("RLE_THROTTLE_SLEEP_DURATION", THROTTLE_SLEEP_DURATION)
    rle_logger.info("RLE utils using dynamically tuned throttle parameters for simple RLE.")
except ImportError:
    rle_logger.info("Performance tuner not found for RLE utils, using default throttle parameters for simple RLE.")
except Exception as e_rle_tune:
    rle_logger.warning(f"Error applying tuned params to RLE utils: {e_rle_tune}. Using defaults for simple RLE.")


def _get_advanced_rle_module():
    global _advanced_rle_module_cache, _advanced_rle_module_available_cache, _adv_module_print_info_done
    if _advanced_rle_module_available_cache is None:
        try:
            from . import advanced_rle_methods
            _advanced_rle_module_cache = advanced_rle_methods
            _advanced_rle_module_available_cache = True
            if not _adv_module_print_info_done:
                rle_logger.info("Successfully imported advanced_rle_methods.")
                _adv_module_print_info_done = True
        except ImportError:
            _advanced_rle_module_available_cache = False
            if not _adv_module_print_info_done:
                rle_logger.info("advanced_rle_methods module not found. Fallback to simple RLE.")
                _adv_module_print_info_done = True
        except Exception as e_adv_import_generic:
            _advanced_rle_module_available_cache = False
            if not _adv_module_print_info_done:
                rle_logger.warning(
                    f"Unexpected error importing advanced_rle_methods: {e_adv_import_generic}. Fallback to simple RLE.")
                _adv_module_print_info_done = True
    return _advanced_rle_module_cache, _advanced_rle_module_available_cache


def simple_rle_compress(text_data: str, min_run_len_override: int = None) -> str:
    if not isinstance(text_data, str):
        raise TypeError("Input data for RLE compression must be a string.")
    if not text_data:
        return ""

    current_min_run = min_run_len_override if min_run_len_override is not None else MIN_ENCODABLE_RUN_LENGTH
    if current_min_run < 1:
        current_min_run = 1

    n = len(text_data)
    # Optimization: if length is less than min run, no compression possible (unless using escaped delimiter, handled below)
    # but escapes are rare in normal text

    result_parts = []
    i = 0
    while i < n:
        current_char = text_data[i]
        count = 1
        i += 1
        while i < n and text_data[i] == current_char:
            count += 1
            i += 1

        if count >= current_min_run:
            result_parts.append(str(count))
            result_parts.append(RLE_DELIMITER)
            result_parts.append(current_char)
        else:
            literal_block = current_char * count
            # Escape the delimiter if it appears in a literal block
            if RLE_DELIMITER in literal_block:
                literal_block = literal_block.replace(RLE_DELIMITER, RLE_DELIMITER + RLE_DELIMITER)
            result_parts.append(literal_block)

    return "".join(result_parts)


def simple_rle_decompress(compressed_text_data: str, min_run_len_override: int = None) -> str:
    if not isinstance(compressed_text_data, str):
        raise TypeError("Input data for RLE decompression must be a string.")
    if not compressed_text_data:
        return ""

    result_parts = []
    i = 0
    n = len(compressed_text_data)
    total_decompressed_size = 0

    while i < n:
        char = compressed_text_data[i]
        if char.isdigit():
            count_str = ""
            # start_of_count_idx = i # unused
            while i < n and compressed_text_data[i].isdigit():
                count_str += compressed_text_data[i]
                i += 1

            # Check for delimiter right after the number
            if i < n and compressed_text_data[i] == RLE_DELIMITER:
                # This is a run
                i += 1  # Consume delimiter
                if i >= n: return RLE_ERROR_NO_CHAR  # No character after delimiter

                char_to_repeat = compressed_text_data[i]
                i += 1  # Consume character

                # Check for escaped delimiter
                if char_to_repeat == RLE_DELIMITER and i < n and compressed_text_data[i] == RLE_DELIMITER:
                    # This was an escaped delimiter, not a run of delimiters
                    result_parts.append(count_str)
                    total_decompressed_size += len(count_str)
                    # We already consumed the first delimiter, now append the second and continue
                    result_parts.append(RLE_DELIMITER)
                    total_decompressed_size += 1
                    continue  # The `i` is already pointing at the char after the second delimiter

                try:
                    parsed_count = int(count_str)
                except ValueError:
                    return RLE_ERROR_BAD_COUNT

                if parsed_count > ABSOLUTE_MAX_PARSED_COUNT: return RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY
                if total_decompressed_size + parsed_count > ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE: return RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED

                try:
                    result_parts.append(char_to_repeat * parsed_count)
                    total_decompressed_size += parsed_count
                except MemoryError:
                    return RLE_ERROR_MEMORY_ON_CHUNK
            else:
                # No delimiter, so the number is a literal
                result_parts.append(count_str)
                total_decompressed_size += len(count_str)
        else:
            # Not a digit, so it's a literal or an escaped delimiter
            if char == RLE_DELIMITER and i + 1 < n and compressed_text_data[i + 1] == RLE_DELIMITER:
                result_parts.append(RLE_DELIMITER)  # It was an escaped delimiter
                total_decompressed_size += 1
                i += 2  # Skip both
            else:
                result_parts.append(char)
                total_decompressed_size += 1
                i += 1

        if total_decompressed_size > ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE:
            return RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED

    try:
        final_result = "".join(result_parts)
    except MemoryError:
        return RLE_ERROR_MEMORY_ON_JOIN

    return final_result


def rle_compress(text_data: str, method: str = "simple", **kwargs) -> str:
    min_run_len_param = kwargs.get('min_run_len_override')
    adv_rle_mod, adv_is_avail = _get_advanced_rle_module()
    if method == "simple":
        return simple_rle_compress(text_data, min_run_len_override=min_run_len_param)
    elif method == "advanced" and adv_is_avail and hasattr(adv_rle_mod, 'advanced_rle_compress'):
        try:
            return adv_rle_mod.advanced_rle_compress(text_data)
        except Exception as e_adv_comp:
            rle_logger.error(f"Error during advanced_rle_compress: {e_adv_comp}. Fallback to simple.", exc_info=True)
            return simple_rle_compress(text_data, min_run_len_override=min_run_len_param)
    else:
        if method == "advanced" and not adv_is_avail:
            rle_logger.warning("Advanced RLE requested but module not available. Falling back to simple RLE.")
        return simple_rle_compress(text_data, min_run_len_override=min_run_len_param)


def rle_decompress(compressed_text_data: str, method: str = "simple", **kwargs) -> str:
    min_run_len_param = kwargs.get('min_run_len_override')
    adv_rle_mod, adv_is_avail = _get_advanced_rle_module()
    if method == "simple":
        return simple_rle_decompress(compressed_text_data, min_run_len_override=min_run_len_param)
    elif method == "advanced" and adv_is_avail and hasattr(adv_rle_mod, 'advanced_rle_decompress'):
        try:
            return adv_rle_mod.advanced_rle_decompress(compressed_text_data)
        except Exception as e_adv_decomp:
            rle_logger.error(f"Error during advanced_rle_decompress: {e_adv_decomp}. Fallback to simple.",
                             exc_info=True)
            return simple_rle_decompress(compressed_text_data, min_run_len_override=min_run_len_param)
    else:
        if method == "advanced" and not adv_is_avail:
            rle_logger.warning(
                "Advanced RLE decompress requested but module not available. Falling back to simple RLE.")
        return simple_rle_decompress(compressed_text_data, min_run_len_override=min_run_len_param)