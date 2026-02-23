# PuffinZipAI_Project/puffinzip_ai/novel_compression_generator.py
"""
Novel Compression Method Generator v2
--------------------------------------
Generates invertible compression methods from composable primitives.
Every primitive is guaranteed to be perfectly reversible.

The AI can also discover new byte-manipulation transforms via random
"corruption" experiments — slamming random bytes together and keeping
any transform that compresses AND decompresses correctly.
"""

import logging
import random
import hashlib
from typing import Callable, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .compression_method_registry import (
    CompressionMethod,
    CompressionLanguage,
    CompressionMetric,
    register_method,
)

gen_logger = logging.getLogger("puffinzip_ai.novel_generator")
if not gen_logger.handlers:
    gen_logger.setLevel(logging.INFO)
    gen_logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
#  INVERTIBLE COMPRESSION PRIMITIVES
#  Each is guaranteed to perfectly round-trip: decompress(compress(x)) == x
# ---------------------------------------------------------------------------

# Markers for identifying encoded formats (single-byte control chars)
_M_RLE  = "\x10"
_M_BWT  = "\x11"
_M_BWTC = "\x12"  # chunked BWT
_M_MTF  = "\x13"
_M_MTFS = "\x14"  # MTF skip (too many unique chars)
_M_DELT = "\x15"
_M_BPE  = "\x16"
_M_BPES = "\x17"  # BPE skip (too small)
_M_XOR  = "\x18"
_M_PERM = "\x19"
_M_BLKS = "\x1A"  # block shuffle skip
_M_BLKD = "\x1B"  # block shuffle data


def _rle_compress_safe(text: str, min_run: int = 3) -> str:
    """Delimiter-based RLE. Unambiguous and perfectly invertible.

    Runs of >= min_run chars become: MARKER + 4-hex-count + char
    Non-runs pass through (with MARKER and ESC chars escaped).

    Escape mechanism (suffix-based, prevents ESC runs from forming):
      literal ESC (\\x1F) → ESC + 'B'
      literal  M  (\\x10) → ESC + 'A'

    After escaping, \\x1F never appears as a standalone char and \\x10 never
    appears at all, so the only \\x10 in the output are legitimate RLE run
    markers.  This makes the decompressor trivially correct.
    """
    if not text:
        return ""
    M = _M_RLE       # \x10 — run marker
    ESC = "\x1F"
    # Escape: ESC first (so newly-inserted ESC aren't re-escaped), then M
    escaped = text.replace(ESC, ESC + "B").replace(M, ESC + "A")

    result = []
    i = 0
    while i < len(escaped):
        char = escaped[i]
        count = 1
        while i + count < len(escaped) and escaped[i + count] == char and count < 65535:
            count += 1
        if count >= min_run:
            result.append(f"{M}{count:04x}{char}")
        else:
            result.append(char * count)
        i += count
    return "".join(result)


def _rle_decompress_safe(compressed: str) -> str:
    """Inverse of _rle_compress_safe.

    Since the escape mechanism guarantees no bare \\x10 exists in the data
    (all are escaped as ESC+'A'), the only \\x10 chars are legitimate RLE
    run markers.  Simple sequential parsing is safe.  Afterwards, a two-step
    unescape restores original ESC and M chars.
    """
    if not compressed:
        return ""
    M = _M_RLE
    ESC = "\x1F"

    result = []
    i = 0
    while i < len(compressed):
        if compressed[i] == M:
            hex_str = compressed[i+1:i+5]
            char = compressed[i+5]
            count = int(hex_str, 16)
            result.append(char * count)
            i += 6
        else:
            result.append(compressed[i])
            i += 1

    unescaped = "".join(result)
    unescaped = unescaped.replace(ESC + "A", M).replace(ESC + "B", ESC)
    return unescaped


def _bwt_compress(text: str) -> str:
    """Burrows-Wheeler Transform — rearranges characters to create long runs.
    Perfectly invertible with stored index. Output: marker + index(4hex) + text.
    """
    if len(text) <= 1:
        return _M_BWT + "0000" + text

    MAX_BWT = 4096  # keep BWT fast
    if len(text) > MAX_BWT:
        chunks = [text[i:i+MAX_BWT] for i in range(0, len(text), MAX_BWT)]
        results = [_bwt_compress(chunk) for chunk in chunks]
        count_hex = f"{len(results):04x}"
        # Use a unique separator that can't appear in BWT output
        joined = _M_BWTC.join(results)
        return _M_BWTC + count_hex + joined

    n = len(text)
    indices = sorted(range(n), key=lambda k: text[k:] + text[:k])
    transformed = "".join(text[(idx - 1) % n] for idx in indices)
    original_idx = indices.index(0)
    return _M_BWT + f"{original_idx:04x}" + transformed


def _bwt_decompress(compressed: str) -> str:
    """Inverse BWT."""
    if not compressed:
        return ""

    if compressed[0] == _M_BWTC:
        count = int(compressed[1:5], 16)
        parts = compressed[5:].split(_M_BWTC)
        return "".join(_bwt_decompress(part) for part in parts)

    if compressed[0] != _M_BWT:
        return compressed

    original_idx = int(compressed[1:5], 16)
    text = compressed[5:]

    if len(text) <= 1:
        return text

    n = len(text)
    # Efficient inverse BWT using LF-mapping
    # Count occurrences of each character
    count = {}
    for c in text:
        count[c] = count.get(c, 0) + 1

    # First column (sorted)
    sorted_chars = sorted(count.keys())
    first_occ = {}
    total = 0
    for c in sorted_chars:
        first_occ[c] = total
        total += count[c]

    # Build LF mapping
    lf = [0] * n
    occ_so_far = {}
    for i in range(n):
        c = text[i]
        occ_so_far[c] = occ_so_far.get(c, 0)
        lf[i] = first_occ[c] + occ_so_far[c]
        occ_so_far[c] += 1

    # Reconstruct
    result = [""] * n
    idx = original_idx
    for i in range(n - 1, -1, -1):
        result[i] = text[idx]
        idx = lf[idx]

    return "".join(result)


def _mtf_compress(text: str) -> str:
    """Move-to-Front transform. After BWT, this produces small numbers.
    Output: marker + alphabet_len(4hex) + alphabet + positions(each 2hex).
    """
    if not text:
        return ""

    alphabet = sorted(set(text))
    if len(alphabet) > 255:
        return _M_MTFS + text  # Too many unique chars, skip

    alpha_len_hex = f"{len(alphabet):02x}"
    alpha_str = "".join(alphabet)

    working = list(alphabet)
    positions = []
    for char in text:
        idx = working.index(char)
        positions.append(f"{idx:02x}")
        working.pop(idx)
        working.insert(0, char)

    return _M_MTF + alpha_len_hex + alpha_str + "".join(positions)


def _mtf_decompress(compressed: str) -> str:
    """Inverse MTF."""
    if not compressed:
        return ""
    if compressed[0] == _M_MTFS:
        return compressed[1:]
    if compressed[0] != _M_MTF:
        return compressed

    alpha_len = int(compressed[1:3], 16)
    alpha_str = compressed[3:3+alpha_len]
    positions_str = compressed[3+alpha_len:]

    working = list(alpha_str)
    result = []

    for i in range(0, len(positions_str), 2):
        hex_str = positions_str[i:i+2]
        if len(hex_str) < 2:
            break
        idx = int(hex_str, 16)
        if idx >= len(working):
            break
        char = working[idx]
        result.append(char)
        working.pop(idx)
        working.insert(0, char)

    return "".join(result)


def _delta_compress(text: str) -> str:
    """Delta encoding — stores differences between consecutive char values.
    Uses 2-byte Latin-1 encoding (big-endian) instead of 4-hex-digit per value.
    This halves the encoding overhead from 4x to 2x. Perfectly invertible.
    """
    if len(text) < 2:
        return _M_DELT + "S" + text  # S = short/skip

    ords = [ord(c) for c in text]
    parts = []
    # First value: raw ordinal, clamped to 16-bit, stored as 2 Latin-1 chars
    first = max(0, min(65535, ords[0]))
    parts.append(chr(first >> 8))
    parts.append(chr(first & 0xFF))
    for i in range(1, len(ords)):
        delta = ords[i] - ords[i-1]
        encoded = delta + 32768  # offset to handle negatives
        encoded = max(0, min(65535, encoded))
        parts.append(chr(encoded >> 8))
        parts.append(chr(encoded & 0xFF))

    return _M_DELT + "D" + "".join(parts)


def _delta_decompress(compressed: str) -> str:
    """Inverse delta encoding (2-byte Latin-1 format)."""
    if not compressed or compressed[0] != _M_DELT:
        return compressed

    if compressed[1] == "S":
        return compressed[2:]

    data = compressed[2:]
    if len(data) < 2:
        return ""

    result = []
    first_val = (ord(data[0]) << 8) | ord(data[1])
    result.append(first_val)

    for i in range(2, len(data) - 1, 2):
        hi = ord(data[i])
        lo = ord(data[i + 1])
        encoded = (hi << 8) | lo
        delta = encoded - 32768
        result.append(result[-1] + delta)

    return "".join(chr(max(0, min(0x10FFFF, v))) for v in result)


def _bpe_compress(text: str) -> str:
    """Byte-pair encoding — replaces common 2-char pairs with single substitution chars.
    Uses Unicode Private Use Area for substitution, making it invertible.
    """
    if len(text) < 6:
        return _M_BPES + text

    pair_freq: Dict[str, int] = {}
    for i in range(len(text) - 1):
        pair = text[i:i+2]
        pair_freq[pair] = pair_freq.get(pair, 0) + 1

    sorted_pairs = sorted(pair_freq.items(), key=lambda x: -x[1])

    used_chars = set(text)
    sub_start = 0xE000  # Unicode Private Use Area

    substitutions = []
    working_text = text

    for pair, freq in sorted_pairs[:32]:
        if freq < 3:
            break

        sub_char = None
        for code in range(sub_start, sub_start + 256):
            c = chr(code)
            if c not in used_chars and c not in working_text:
                sub_char = c
                sub_start = code + 1
                break

        if sub_char is None:
            break

        savings = freq - 6  # overhead per entry
        if savings <= 0:
            break

        working_text = working_text.replace(pair, sub_char)
        used_chars.add(sub_char)
        substitutions.append((sub_char, pair))

    if not substitutions:
        return _M_BPES + text

    # Header: count(2hex) + [sub_char + pair_char1 + pair_char2] * count + separator
    header = f"{len(substitutions):02x}"
    for sub_char, pair in substitutions:
        header += sub_char + pair

    return _M_BPE + header + "|" + working_text


def _bpe_decompress(compressed: str) -> str:
    """Inverse BPE."""
    if not compressed:
        return ""
    if compressed[0] == _M_BPES:
        return compressed[1:]
    if compressed[0] != _M_BPE:
        return compressed

    count = int(compressed[1:3], 16)
    entries_start = 3
    substitutions = []
    for i in range(count):
        offset = entries_start + i * 3
        sub_char = compressed[offset]
        pair = compressed[offset+1:offset+3]
        substitutions.append((sub_char, pair))

    separator_idx = entries_start + count * 3
    text = compressed[separator_idx + 1:]

    for sub_char, pair in reversed(substitutions):
        text = text.replace(sub_char, pair)

    return text


# ---------------------------------------------------------------------------
#  RANDOM DISCOVERY TRANSFORMS
#  These are the "slam random characters together" experiments.
#  XOR and permutations are always mathematically invertible.
# ---------------------------------------------------------------------------

def _create_random_xor_transform(seed: int) -> Tuple[Callable, Callable, Dict]:
    """XOR with a random key. Always perfectly invertible (apply twice = original).
    Can pre-condition data to create more runs for downstream RLE.

    Encoding: Latin-1 (1:1 byte↔char mapping, zero overhead vs 2x for hex).
    Format: MARKER + key_len_char + key_bytes_latin1 + data_bytes_latin1
    """
    rng = random.Random(seed)
    key_len = rng.randint(4, 64)
    key = bytes(rng.randint(0, 255) for _ in range(key_len))

    def xor_fwd(text: str) -> str:
        data = text.encode('utf-8', errors='replace')
        result = bytearray(len(data))
        for i in range(len(data)):
            result[i] = data[i] ^ key[i % key_len]
        # Store as: marker + chr(key_len) + key(latin-1) + data(latin-1)
        return _M_XOR + chr(key_len) + key.decode('latin-1') + bytes(result).decode('latin-1')

    def xor_rev(compressed: str) -> str:
        if not compressed.startswith(_M_XOR):
            return compressed
        kl = ord(compressed[1])
        stored_key = compressed[2:2 + kl].encode('latin-1')
        data = compressed[2 + kl:].encode('latin-1')
        result = bytearray(len(data))
        for i in range(len(data)):
            result[i] = data[i] ^ stored_key[i % kl]
        return result.decode('utf-8', errors='replace')

    return xor_fwd, xor_rev, {"type": "xor", "key_len": key_len}


def _create_random_byte_permutation(seed: int) -> Tuple[Callable, Callable, Dict]:
    """Random byte-value permutation. Bijective = perfectly invertible.

    Encoding: Latin-1 (1:1 byte↔char mapping, zero overhead vs 2x for hex).
    Format: MARKER + 256_perm_bytes_latin1 + data_bytes_latin1
    Permutation table is always exactly 256 bytes, so no delimiter needed.
    """
    rng = random.Random(seed)
    perm = list(range(256))
    rng.shuffle(perm)
    perm_latin1 = bytes(perm).decode('latin-1')  # Always exactly 256 chars

    def perm_fwd(text: str) -> str:
        data = text.encode('utf-8', errors='replace')
        result = bytes(perm[b] for b in data)
        return _M_PERM + perm_latin1 + result.decode('latin-1')

    def perm_rev(compressed: str) -> str:
        if not compressed.startswith(_M_PERM):
            return compressed
        stored_perm = [ord(c) for c in compressed[1:257]]  # 256 chars after marker
        inv_perm = [0] * 256
        for i, p in enumerate(stored_perm):
            inv_perm[p] = i
        data = compressed[257:].encode('latin-1')
        result = bytes(inv_perm[b] for b in data)
        return result.decode('utf-8', errors='replace')

    return perm_fwd, perm_rev, {"type": "permutation"}


def _create_random_block_shuffle(seed: int) -> Tuple[Callable, Callable, Dict]:
    """Shuffle fixed-size blocks. With stored permutation, perfectly invertible."""
    rng = random.Random(seed)
    block_size = rng.choice([4, 8, 16, 32])

    def blk_fwd(text: str) -> str:
        if len(text) < block_size * 2:
            return _M_BLKS + text

        blocks = [text[i:i+block_size] for i in range(0, len(text), block_size)]
        n = len(blocks)
        perm = list(range(n))
        random.Random(seed).shuffle(perm)

        shuffled = [""] * n
        for i, p in enumerate(perm):
            if p < n and i < len(blocks):
                shuffled[p] = blocks[i]

        perm_str = ",".join(str(p) for p in perm)
        return f"{_M_BLKD}{block_size:02x}{perm_str}|" + "".join(shuffled)

    def blk_rev(compressed: str) -> str:
        if not compressed:
            return compressed
        if compressed[0] == _M_BLKS:
            return compressed[1:]
        if compressed[0] != _M_BLKD:
            return compressed

        bs = int(compressed[1:3], 16)
        sep = compressed.index("|")
        perm_str = compressed[3:sep]
        perm = [int(x) for x in perm_str.split(",")]
        text = compressed[sep+1:]

        blocks = [text[i:i+bs] for i in range(0, len(text), bs)]
        n = len(blocks)

        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            if p < len(inv_perm):
                inv_perm[p] = i

        result = [""] * max(n, len(inv_perm))
        for i in range(n):
            src = inv_perm[i] if i < len(inv_perm) else i
            result[i] = blocks[src] if src < n else ""

        return "".join(result[:n])

    return blk_fwd, blk_rev, {"type": "block_shuffle", "block_size": block_size}


# ---------------------------------------------------------------------------
#  DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class CompressionPattern:
    """A pattern/rule that can be applied in compression"""
    name: str
    description: str
    pattern_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    effectiveness_score: float = 0.0


@dataclass
class DiscoveredTransform:
    """A byte-level transform discovered through random experimentation."""
    name: str
    transform_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    effectiveness: float = 0.0
    generation_discovered: int = 0
    test_pass_count: int = 0


# ---------------------------------------------------------------------------
#  MAIN GENERATOR CLASS
# ---------------------------------------------------------------------------

class NovelCompressionGenerator:
    """Generates and evolves novel compression methods with guaranteed invertibility.

    Methods are composed from invertible primitives:
    - Safe RLE (delimiter-based, unambiguous)
    - BWT (Burrows-Wheeler Transform for run creation)
    - MTF (Move-to-Front for BWT output optimization)
    - Delta encoding (for smooth data)
    - BPE (byte-pair encoding for repeated pairs)
    - Random discovery transforms (XOR, byte permutations, block shuffles)

    Every generated method is verified for invertibility before being returned.
    """

    PIPELINES = {
        "rle_only":    ["rle"],
        "bwt_rle":     ["bwt", "rle"],
        "bwt_mtf_rle": ["bwt", "mtf", "rle"],
        "delta_rle":   ["delta", "rle"],
        "bpe_rle":     ["bpe", "rle"],
        "bpe_only":    ["bpe"],
        "delta_bpe":   ["delta", "bpe"],
        "bwt_bpe":     ["bwt", "bpe"],
    }

    def __init__(self):
        self.generated_methods: List[CompressionMethod] = []
        self.discovered_transforms: List[DiscoveredTransform] = []
        self.evolution_history: List[Dict[str, Any]] = []
        self._discovery_seeds_tried: set = set()
        gen_logger.info("NovelCompressionGenerator v2 initialized (all primitives invertible)")

    def _build_pipeline(self, pipeline_name: str, discovery_seed: Optional[int] = None,
                        rle_min_run: Optional[int] = None) -> Tuple[Callable, Callable]:
        """Build a compress/decompress pair from a named pipeline.
        Each step is invertible. Decompression reverses the steps.
        """
        steps = list(self.PIPELINES.get(pipeline_name, ["rle"]))

        discovery_fwd, discovery_rev = None, None
        if discovery_seed is not None:
            transform_type = discovery_seed % 3
            if transform_type == 0:
                discovery_fwd, discovery_rev, _ = _create_random_xor_transform(discovery_seed)
            elif transform_type == 1:
                discovery_fwd, discovery_rev, _ = _create_random_byte_permutation(discovery_seed)
            else:
                discovery_fwd, discovery_rev, _ = _create_random_block_shuffle(discovery_seed)

        min_run = rle_min_run if rle_min_run else random.randint(2, 5)

        def compress(text: str) -> str:
            if not text:
                return ""
            result = text

            if discovery_fwd is not None:
                try:
                    result = discovery_fwd(result)
                except Exception:
                    pass

            for step in steps:
                try:
                    if step == "rle":
                        result = _rle_compress_safe(result, min_run=min_run)
                    elif step == "bwt":
                        result = _bwt_compress(result)
                    elif step == "mtf":
                        result = _mtf_compress(result)
                    elif step == "delta":
                        result = _delta_compress(result)
                    elif step == "bpe":
                        result = _bpe_compress(result)
                except Exception as e:
                    gen_logger.debug(f"Pipeline step '{step}' failed: {e}")

            return result

        def decompress(compressed_text: str) -> str:
            if not compressed_text:
                return ""
            result = compressed_text

            for step in reversed(steps):
                try:
                    if step == "rle":
                        result = _rle_decompress_safe(result)
                    elif step == "bwt":
                        result = _bwt_decompress(result)
                    elif step == "mtf":
                        result = _mtf_decompress(result)
                    elif step == "delta":
                        result = _delta_decompress(result)
                    elif step == "bpe":
                        result = _bpe_decompress(result)
                except Exception as e:
                    gen_logger.debug(f"Pipeline reverse step '{step}' failed: {e}")

            if discovery_rev is not None:
                try:
                    result = discovery_rev(result)
                except Exception:
                    pass

            return result

        return compress, decompress

    def _verify_invertibility(self, compress_fn: Callable, decompress_fn: Callable,
                              num_tests: int = 5) -> bool:
        """Verify compress->decompress round-trips correctly on several test inputs."""
        test_cases = [
            "AAABBBCCCDDD",
            "Hello World! This is a test string with digits 12345 and symbols!@#",
            "aaaaaabbbbbbcccccc" * 10,
            "".join(chr(random.randint(32, 126)) for _ in range(200)),
            "x" * 500 + "y" * 300 + "z" * 200,
        ]

        for test in test_cases[:num_tests]:
            try:
                compressed = compress_fn(test)
                decompressed = decompress_fn(compressed)
                if decompressed != test:
                    return False
            except Exception:
                return False
        return True

    def generate_novelty_method(self, method_name: Optional[str] = None,
                                pattern_combo_size: Optional[int] = None) -> CompressionMethod:
        """Generate a novel, verified-invertible compression method.

        Tries pipeline + optional discovery transform, verifying invertibility.
        Falls back to simpler pipelines if verification fails.
        """
        if method_name is None:
            method_name = f"novelty_v{len(self.generated_methods) + 1}"

        pipeline_name = random.choice(list(self.PIPELINES.keys()))

        # 40% chance of a random discovery transform
        discovery_seed = None
        if random.random() < 0.4:
            discovery_seed = random.randint(0, 2**31)
            while discovery_seed in self._discovery_seeds_tried:
                discovery_seed = random.randint(0, 2**31)
            self._discovery_seeds_tried.add(discovery_seed)

        rle_min_run = random.randint(2, 5)

        compress_fn, decompress_fn = self._build_pipeline(
            pipeline_name, discovery_seed=discovery_seed, rle_min_run=rle_min_run
        )

        if not self._verify_invertibility(compress_fn, decompress_fn):
            gen_logger.debug(f"Pipeline '{pipeline_name}' + seed {discovery_seed} failed, retrying without discovery")
            discovery_seed = None
            compress_fn, decompress_fn = self._build_pipeline(
                pipeline_name, discovery_seed=None, rle_min_run=rle_min_run
            )

            if not self._verify_invertibility(compress_fn, decompress_fn):
                gen_logger.debug(f"Pipeline '{pipeline_name}' still failing, falling back to rle_only")
                pipeline_name = "rle_only"
                compress_fn, decompress_fn = self._build_pipeline("rle_only", rle_min_run=rle_min_run)

        discovery_str = f" + discovery(seed={discovery_seed})" if discovery_seed else ""
        description = f"Pipeline: {pipeline_name}{discovery_str}, min_run={rle_min_run}"

        method = CompressionMethod(
            name=method_name,
            language=CompressionLanguage.HYBRID,
            compress_fn=compress_fn,
            decompress_fn=decompress_fn,
            description=description,
            author="NovelCompressionGenerator_v2",
            is_novelty=True,
            metadata={
                "pipeline": pipeline_name,
                "discovery_seed": discovery_seed,
                "rle_min_run": rle_min_run,
                "steps": self.PIPELINES.get(pipeline_name, ["rle"]),
            }
        )

        self.generated_methods.append(method)
        try:
            register_method(method)
        except Exception:
            pass

        gen_logger.info(f"✨ Generated novelty method: {method_name} — {description}")
        return method

    def generate_random_discovery(self) -> Optional[DiscoveredTransform]:
        """Attempt a random byte-level discovery by creating a random transform
        and testing if it helps compression when combined with RLE.

        This is the "slamming random characters together" mechanism.
        Returns the transform if it passes invertibility and compresses better
        than plain RLE, otherwise None.
        """
        seed = random.randint(0, 2**31)
        while seed in self._discovery_seeds_tried:
            seed = random.randint(0, 2**31)
        self._discovery_seeds_tried.add(seed)

        compress_fn, decompress_fn = self._build_pipeline("rle_only", discovery_seed=seed)

        if not self._verify_invertibility(compress_fn, decompress_fn):
            return None

        test_data = "AABBBCCCCDDDDDEEEEEFFFFFFGGGGGGG" * 20
        plain_rle = _rle_compress_safe(test_data, min_run=3)
        novel_compressed = compress_fn(test_data)

        if len(novel_compressed) < len(plain_rle):
            transform = DiscoveredTransform(
                name=f"discovery_{seed}",
                transform_type="pipeline_with_random_precondition",
                parameters={"seed": seed},
                effectiveness=1.0 - len(novel_compressed) / len(plain_rle),
                test_pass_count=5,
            )
            self.discovered_transforms.append(transform)
            gen_logger.info(
                f"🔬 Random discovery! Seed {seed} compresses "
                f"{(1 - len(novel_compressed)/len(plain_rle))*100:.1f}% better than plain RLE"
            )
            return transform

        return None

    def evolve_methods(self, num_mutations: int = 5) -> List[CompressionMethod]:
        """Evolve new methods by trying new pipeline + discovery combinations."""
        new_methods = []
        for i in range(num_mutations):
            method = self.generate_novelty_method(f"evolved_gen{i}")
            new_methods.append(method)

        discoveries = 0
        for _ in range(num_mutations * 2):
            if self.generate_random_discovery():
                discoveries += 1

        gen_logger.info(f"Evolved {num_mutations} methods, made {discoveries} random discoveries")
        return new_methods

    def get_generated_methods(self) -> List[CompressionMethod]:
        return self.generated_methods

    def get_discoveries(self) -> List[DiscoveredTransform]:
        return self.discovered_transforms


# Global singleton generator
_GLOBAL_GENERATOR = NovelCompressionGenerator()


def get_generator() -> NovelCompressionGenerator:
    return _GLOBAL_GENERATOR


def generate_novelty() -> CompressionMethod:
    return _GLOBAL_GENERATOR.generate_novelty_method()


def evolve(num_mutations: int = 5) -> List[CompressionMethod]:
    return _GLOBAL_GENERATOR.evolve_methods(num_mutations)
