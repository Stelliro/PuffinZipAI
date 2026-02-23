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
# STX control char used as frame marker for runs.  Format:
#   Run  (count >= min_run):  RLE_RUN_MARKER + count_digits + RLE_RUN_MARKER + char
#   Literal marker in data:  RLE_RUN_MARKER + RLE_RUN_MARKER
#   Everything else:         literal char
# This is completely unambiguous: digit characters outside marker frames are always literal.
RLE_RUN_MARKER = '\x02'

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
    """Standard Run-Length Encoding with unambiguous marker-framed format.

    Encoding rules:
      - A run of `count` copies of `char` (count >= min_run) is encoded as
        RLE_RUN_MARKER + str(count) + RLE_RUN_MARKER + char.
      - A literal RLE_RUN_MARKER in the data is escaped as
        RLE_RUN_MARKER + RLE_RUN_MARKER.
      - All other characters (including digits) are emitted literally.

    SAFE MODE: Breaks runs > 10,000 characters into chunks to prevent thread
    hanging on massive files.
    """
    if not isinstance(text_data, str):
        try:
            text_data = str(text_data)
        except Exception:
            raise TypeError("Input data for RLE compression must be a string.")

    if not text_data:
        return ""

    current_min_run = min_run_len_override if min_run_len_override is not None else MIN_ENCODABLE_RUN_LENGTH
    if current_min_run < 1:
        current_min_run = 1

    M = RLE_RUN_MARKER
    n = len(text_data)
    result_parts = []
    i = 0

    while i < n:
        current_char = text_data[i]
        count = 1
        i += 1

        while i < n and text_data[i] == current_char:
            count += 1
            i += 1
            # SAFETY BREAK: chunk massive runs to avoid CPU freeze
            if count > 10000:
                break

        if count >= current_min_run:
            # Framed run: MARKER + count + MARKER + char
            result_parts.append(M)
            result_parts.append(str(count))
            result_parts.append(M)
            result_parts.append(current_char)
        else:
            # Literal — escape any marker chars
            for _ in range(count):
                if current_char == M:
                    result_parts.append(M)
                    result_parts.append(M)
                else:
                    result_parts.append(current_char)

    return "".join(result_parts)


def simple_rle_decompress(compressed_text_data: str, min_run_len_override: int = None) -> str:
    """Decompress marker-framed RLE.

    Decoding rules:
      - MARKER + digits + MARKER + char  →  repeat char `digits` times.
      - MARKER + MARKER                  →  literal MARKER character.
      - Any other character               →  literal.
    """
    if not isinstance(compressed_text_data, str):
        raise TypeError("Input data for RLE decompression must be a string.")
    if not compressed_text_data:
        return ""

    M = RLE_RUN_MARKER
    result_parts = []
    i = 0
    n = len(compressed_text_data)
    total_decompressed_size = 0

    while i < n:
        char = compressed_text_data[i]

        if char == M:
            i += 1
            if i >= n:
                # Trailing orphan marker — tolerate as literal
                result_parts.append(M)
                total_decompressed_size += 1
                break

            next_char = compressed_text_data[i]

            if next_char == M:
                # Escaped marker → literal marker char
                result_parts.append(M)
                total_decompressed_size += 1
                i += 1

            elif next_char.isdigit():
                # Run encoding: MARKER + count_digits + MARKER + char
                count_str = ""
                digit_read_count = 0
                while i < n and compressed_text_data[i].isdigit() and digit_read_count < MAX_COUNT_STRING_DIGITS:
                    count_str += compressed_text_data[i]
                    i += 1
                    digit_read_count += 1

                # Too many digits — safety limit
                if digit_read_count == MAX_COUNT_STRING_DIGITS and i < n and compressed_text_data[i].isdigit():
                    return RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY

                # Expect closing MARKER
                if i >= n or compressed_text_data[i] != M:
                    return RLE_ERROR_MALFORMED
                i += 1  # consume closing marker

                # Expect the character to repeat
                if i >= n:
                    return RLE_ERROR_NO_CHAR
                char_to_repeat = compressed_text_data[i]
                i += 1

                try:
                    parsed_count = int(count_str)
                except ValueError:
                    return RLE_ERROR_BAD_COUNT

                if parsed_count > ABSOLUTE_MAX_PARSED_COUNT:
                    return RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY
                if total_decompressed_size + parsed_count > ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE:
                    return RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED

                if parsed_count > THROTTLE_RUN_LENGTH_THRESHOLD:
                    remaining = parsed_count
                    while remaining > 0:
                        chunk_len = min(remaining, THROTTLE_CHUNK_SIZE)
                        try:
                            result_parts.append(char_to_repeat * chunk_len)
                        except MemoryError:
                            return RLE_ERROR_MEMORY_ON_CHUNK
                        total_decompressed_size += chunk_len
                        remaining -= chunk_len
                        if remaining > 0:
                            time.sleep(THROTTLE_SLEEP_DURATION)
                else:
                    try:
                        result_parts.append(char_to_repeat * parsed_count)
                    except MemoryError:
                        return RLE_ERROR_MEMORY_ON_CHUNK
                    total_decompressed_size += parsed_count
            else:
                # MARKER followed by non-digit, non-MARKER — tolerate as literal
                result_parts.append(M)
                total_decompressed_size += 1
                # Don't consume next_char; it will be processed on next iteration
        else:
            # Literal character (digits, letters, anything — literal outside marker frames)
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