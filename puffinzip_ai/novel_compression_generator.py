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
import os
import json
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
#  RECIPE-BASED NOVEL METHOD EVOLUTION  (v0.9.9)
#
#  Novel methods are NO LONGER randomly generated for each agent.  Instead
#  they evolve incrementally through partial mutations:
#
#    1.  All agents start with the same simple base recipe (RLE-only).
#    2.  Each generation, one small mutation is applied to the child's
#        inherited recipe (add/remove/tweak a single transform step).
#    3.  If the mutation improves compression: the change is recorded in
#        the recipe's improvement log and contribution scores are updated.
#    4.  Children inherit ALL accumulated improvements from their parent.
#    5.  When two parents with structurally different recipes breed, a
#        *sub-novel method* is created that blends steps from both.
#    6.  Only recipes with >=2 proven improvements count as "novel".
#
# v0.9.10 additions:
#    7.  **Strength** — each recipe has a LoRA-like strength (0.0–1.0) that
#        increases on improvement and decays when stagnant.  Strength
#        influences breeding selection probability and method application
#        intensity.
#    8.  **Method Registry / Graveyard** — ALL recipe families ever seen are
#        catalogued.  When a recipe dies (strength → 0), it moves to the
#        graveyard but persists so it can be recognized if it re-emerges
#        (like a recessive gene resurfacing).
#    9.  **Breeding-out** — weak recipes are weakened further; when strength
#        hits 0, the recipe is replaced with a base recipe and the old one
#        is archived to the graveyard.
#   10.  **Dormant re-emergence** — during mutations, a small chance to
#        resurrect a recipe from the graveyard.  Re-emerged recipes are
#        tagged and their ``times_rediscovered`` counter increments.
# ---------------------------------------------------------------------------

@dataclass
class RecipeStep:
    """A single transform step within a novel method recipe."""
    step_type: str   # "rle", "bwt", "mtf", "delta", "bpe", "xor", "perm", "block_shuffle"
    params: Dict[str, Any] = field(default_factory=dict)
    contribution_score: float = 0.0   # Tracks how much this step helps
    generation_added: int = 0


@dataclass
class NovelMethodRecipe:
    """A mutable, evolvable compression recipe built incrementally.

    Recipes start simple (just RLE) and grow through small, tested
    mutations.  A recipe is only considered genuinely *novel* once it
    has accumulated ``MATURITY_THRESHOLD`` proven improvements — until
    then it is just a variant of the baseline.

    v0.9.10: Recipes also carry a ``strength`` field (0.0–1.0) modeled
    after LoRA weights.  Strength boosts when the recipe improves and
    decays each generation it stagnates.  When strength reaches 0, the
    recipe is "dead" and moves to the method graveyard.
    """
    MATURITY_THRESHOLD: int = field(default=2, repr=False)

    recipe_id: str = ""
    steps: List[RecipeStep] = field(default_factory=list)
    rle_min_run: int = 3
    discovery_seed: Optional[int] = None
    generation_created: int = 0
    generation_last_improved: int = 0
    best_fitness: float = 0.0
    parent_recipe_ids: List[str] = field(default_factory=list)
    improvement_log: List[Dict[str, Any]] = field(default_factory=list)

    # v0.9.10: LoRA-like strength — how influential this recipe currently is
    # Starts at a moderate baseline (0.5), increases on improvement, decays
    # when stagnant.  Used for breeding selection weight and method intensity.
    strength: float = 0.5

    # v0.9.10: Lifecycle tracking
    is_alive: bool = True                      # False = in graveyard
    death_generation: Optional[int] = None     # Generation when strength hit 0
    times_rediscovered: int = 0                # How many times this family re-emerged
    total_generations_active: int = 0          # Cumulative gens this recipe was alive

    # ---- queries ----
    @property
    def is_mature(self) -> bool:
        """True when this recipe has enough proven improvements to be
        considered a genuinely novel method (not just a baseline variant)."""
        return len(self.improvement_log) >= self.MATURITY_THRESHOLD

    @property
    def is_weak(self) -> bool:
        """True when this recipe's strength has dropped below the
        breeding-out threshold (0.1).  Weak recipes are candidates for
        replacement during the next breeding cycle."""
        return self.strength < 0.1

    @property
    def step_types(self) -> tuple:
        """Ordered tuple of step type strings — used for family comparison."""
        return tuple(s.step_type for s in self.steps)

    @property
    def family_key(self) -> str:
        """Unique string identifying this recipe's structural family.
        Used for deduplication in the registry and vault."""
        return "_".join(s.step_type for s in self.steps) or "empty"

    def get_pipeline_name(self) -> str:
        """Return a pipeline name string compatible with the old system."""
        types = [s.step_type for s in self.steps]
        return "_".join(types) if types else "rle_only"

    @property
    def aggregate_contribution(self) -> float:
        """Sum of all steps' contribution scores — a quick proxy for
        how proven this recipe's pipeline is."""
        return sum(s.contribution_score for s in self.steps)

    # ---- serialization ----
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for heritage / pickle / JSON."""
        return {
            "recipe_id": self.recipe_id,
            "steps": [
                {
                    "step_type": s.step_type,
                    "params": dict(s.params),
                    "contribution_score": s.contribution_score,
                    "generation_added": s.generation_added,
                }
                for s in self.steps
            ],
            "rle_min_run": self.rle_min_run,
            "discovery_seed": self.discovery_seed,
            "generation_created": self.generation_created,
            "generation_last_improved": self.generation_last_improved,
            "best_fitness": self.best_fitness,
            "parent_recipe_ids": list(self.parent_recipe_ids),
            "improvement_log": list(self.improvement_log),
            # v0.9.10: strength + lifecycle
            "strength": self.strength,
            "is_alive": self.is_alive,
            "death_generation": self.death_generation,
            "times_rediscovered": self.times_rediscovered,
            "total_generations_active": self.total_generations_active,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NovelMethodRecipe":
        """Reconstruct from a serialized dict."""
        steps = [
            RecipeStep(
                step_type=s["step_type"],
                params=dict(s.get("params", {})),
                contribution_score=s.get("contribution_score", 0.0),
                generation_added=s.get("generation_added", 0),
            )
            for s in d.get("steps", [])
        ]
        return cls(
            recipe_id=d.get("recipe_id", ""),
            steps=steps,
            rle_min_run=d.get("rle_min_run", 3),
            discovery_seed=d.get("discovery_seed"),
            generation_created=d.get("generation_created", 0),
            generation_last_improved=d.get("generation_last_improved", 0),
            best_fitness=d.get("best_fitness", 0.0),
            parent_recipe_ids=d.get("parent_recipe_ids", []),
            improvement_log=d.get("improvement_log", []),
            # v0.9.10: strength + lifecycle
            strength=d.get("strength", 0.5),
            is_alive=d.get("is_alive", True),
            death_generation=d.get("death_generation"),
            times_rediscovered=d.get("times_rediscovered", 0),
            total_generations_active=d.get("total_generations_active", 0),
        )

    # ---- backward-compat helper: build old-style metadata dict ----
    def to_legacy_metadata(self) -> Dict[str, Any]:
        """Return a metadata dict that downstream code expecting the old
        ``CompressionMethod.metadata`` format can consume unchanged."""
        core_steps = [s.step_type for s in self.steps
                      if s.step_type in ("rle", "bwt", "mtf", "delta", "bpe")]
        return {
            "pipeline": self.get_pipeline_name(),
            "discovery_seed": self.discovery_seed,
            "rle_min_run": self.rle_min_run,
            "steps": core_steps,
            "recipe": self.to_dict(),      # New: full recipe data
            "is_mature": self.is_mature,   # New: maturity flag
        }


class RecipeEvolver:
    """Evolves novel method recipes incrementally through partial mutations.

    Design principles:
    * Start simple — every family begins with ``["rle"]``.
    * One small change per generation — add/remove/tweak one step.
    * Track what works — successful mutations are logged so descendants
      inherit proven improvements.
    * Cross-family breeding — when parents have structurally different
      recipes, combine the best-contributing steps from each.
    """

    CORE_STEPS = ("rle", "bwt", "mtf", "delta", "bpe")
    DISCOVERY_STEPS = ("xor", "perm", "block_shuffle")
    ALL_STEPS = CORE_STEPS + DISCOVERY_STEPS

    MUTATION_TYPES = (
        "add_step",
        "remove_step",
        "modify_param",
        "swap_step",
        "reorder_steps",
    )

    MAX_PIPELINE_LEN = 6  # Safety cap

    # ------------------------------------------------------------------ #
    #  Recipe creation
    # ------------------------------------------------------------------ #
    @staticmethod
    def create_base_recipe(generation: int = 0) -> NovelMethodRecipe:
        """Create the minimal base recipe that ALL agents start with.

        This is intentionally simple — complexity must be *earned* through
        proven improvements over subsequent generations.
        """
        import uuid as _uuid
        return NovelMethodRecipe(
            recipe_id=str(_uuid.uuid4()),
            steps=[RecipeStep(step_type="rle", params={"min_run": 3},
                              generation_added=generation)],
            rle_min_run=3,
            generation_created=generation,
        )

    # ------------------------------------------------------------------ #
    #  Partial mutation  (one small change)
    # ------------------------------------------------------------------ #
    @staticmethod
    def mutate_recipe(recipe: NovelMethodRecipe,
                      generation: int) -> Tuple["NovelMethodRecipe", Dict[str, Any]]:
        """Apply **one** small random mutation to *recipe*.

        Returns ``(mutated_copy, change_desc)`` where *change_desc* is a
        dict describing what was changed (stored in improvement_log if the
        mutation later proves beneficial).
        """
        import copy as _copy
        mutated = _copy.deepcopy(recipe)

        # --- choose available mutations based on recipe state ---
        available = list(RecipeEvolver.MUTATION_TYPES)
        if len(mutated.steps) <= 1:
            # Can't remove or reorder the only step
            available = [m for m in available if m not in ("remove_step", "reorder_steps")]
        if len(mutated.steps) >= RecipeEvolver.MAX_PIPELINE_LEN:
            available = [m for m in available if m != "add_step"]

        mutation_type = random.choice(available)
        change: Dict[str, Any] = {"mutation_type": mutation_type, "generation": generation}

        if mutation_type == "add_step":
            # 25 % chance of a discovery transform (rarer, more exotic)
            if random.random() < 0.25 and len(mutated.steps) >= 2:
                new_type = random.choice(RecipeEvolver.DISCOVERY_STEPS)
                params: Dict[str, Any] = {"seed": random.randint(0, 2**31)}
            else:
                new_type = random.choice(RecipeEvolver.CORE_STEPS)
                params = {}
                if new_type == "rle":
                    params["min_run"] = random.randint(2, 7)
            pos = random.randint(0, len(mutated.steps))
            mutated.steps.insert(
                pos,
                RecipeStep(step_type=new_type, params=params,
                           generation_added=generation),
            )
            change.update(added_step=new_type, position=pos)

        elif mutation_type == "remove_step":
            # Prefer removing low-contribution steps
            weights = [1.0 / (0.1 + abs(s.contribution_score))
                       for s in mutated.steps]
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            idx = random.choices(range(len(mutated.steps)), weights=weights, k=1)[0]
            removed = mutated.steps.pop(idx)
            change.update(removed_step=removed.step_type, position=idx)

        elif mutation_type == "modify_param":
            idx = random.randint(0, len(mutated.steps) - 1)
            step = mutated.steps[idx]
            change.update(step_index=idx, step_type=step.step_type)
            if step.step_type == "rle":
                old_v = step.params.get("min_run", 3)
                new_v = max(2, min(7, old_v + random.choice([-2, -1, 1, 2])))
                step.params["min_run"] = new_v
                mutated.rle_min_run = new_v
                change.update(param="min_run", old_value=old_v, new_value=new_v)
            elif step.step_type in RecipeEvolver.DISCOVERY_STEPS:
                old_seed = step.params.get("seed", 0)
                # Flip 1-4 random bits → small neighbourhood search
                new_seed = old_seed
                for _ in range(random.randint(1, 4)):
                    new_seed ^= (1 << random.randint(0, 30))
                step.params["seed"] = new_seed
                change.update(param="seed", old_value=old_seed, new_value=new_seed)
            else:
                change.update(param="none")  # BWT/MTF/delta/BPE have no tuneable params

        elif mutation_type == "swap_step":
            idx = random.randint(0, len(mutated.steps) - 1)
            old_type = mutated.steps[idx].step_type
            candidates = [s for s in RecipeEvolver.ALL_STEPS if s != old_type]
            new_type = random.choice(candidates)
            params = {}
            if new_type == "rle":
                params["min_run"] = random.randint(2, 7)
            elif new_type in RecipeEvolver.DISCOVERY_STEPS:
                params["seed"] = random.randint(0, 2**31)
            mutated.steps[idx] = RecipeStep(
                step_type=new_type, params=params, generation_added=generation,
            )
            change.update(position=idx, old_step=old_type, new_step=new_type)

        elif mutation_type == "reorder_steps":
            idx = random.randint(0, len(mutated.steps) - 2)
            mutated.steps[idx], mutated.steps[idx + 1] = (
                mutated.steps[idx + 1], mutated.steps[idx])
            change.update(swapped_positions=[idx, idx + 1])

        return mutated, change

    # ------------------------------------------------------------------ #
    #  Improvement tracking
    # ------------------------------------------------------------------ #
    @staticmethod
    def record_improvement(recipe: NovelMethodRecipe,
                           change: Dict[str, Any],
                           fitness: float,
                           generation: int):
        """Record a proven improvement to the recipe.

        Called after evaluation when the child's fitness exceeds the
        recipe's ``best_fitness``.  The change is logged, all steps
        present during the improvement get a contribution boost, and
        the recipe's strength is increased (capped at 1.0).
        """
        old_best = recipe.best_fitness
        if fitness <= old_best:
            return  # Not an improvement — nothing to record

        delta = fitness - old_best
        recipe.improvement_log.append({
            "change": change,
            "fitness_before": old_best,
            "fitness_after": fitness,
            "delta": round(delta, 6),
            "generation": generation,
        })
        recipe.best_fitness = fitness
        recipe.generation_last_improved = generation

        # Boost contribution scores — weight by relative delta magnitude
        # so big jumps reward steps more than tiny tweaks.
        boost_factor = min(delta * 0.15, 0.5)  # Capped per-step boost
        for step in recipe.steps:
            step.contribution_score += boost_factor

        # v0.9.10: Boost recipe strength on improvement
        # Strength gain is proportional to delta, capped at 1.0
        strength_gain = min(delta * 0.25, 0.3)
        recipe.strength = min(1.0, recipe.strength + strength_gain)

        gen_logger.info(
            f"📈 Recipe {recipe.recipe_id[:8]} improved: "
            f"{old_best:.4f} → {fitness:.4f} (+{delta:.4f}) "
            f"via {change.get('mutation_type', '?')} "
            f"[mature={recipe.is_mature}, strength={recipe.strength:.2f}]"
        )

    # ------------------------------------------------------------------ #
    #  Strength decay (called each generation for all living recipes)
    # ------------------------------------------------------------------ #
    @staticmethod
    def decay_strength(recipe: NovelMethodRecipe,
                       current_generation: int,
                       decay_rate: float = 0.03) -> bool:
        """Apply per-generation strength decay to a recipe.

        Recipes that haven't improved recently lose strength faster.
        Returns True if the recipe's strength has dropped to 0 (dead).

        Decay schedule:
        * Base decay: ``decay_rate`` per generation (default 0.03).
        * Stagnation penalty: additional decay if the recipe hasn't
          improved in the last 5 generations.
        * Immature penalty: recipes that never matured decay 2x faster.
        """
        if not recipe.is_alive:
            return True  # Already dead

        recipe.total_generations_active += 1

        # Base decay
        actual_decay = decay_rate

        # Stagnation penalty — no improvement in 5+ generations
        gens_since_improvement = current_generation - recipe.generation_last_improved
        if gens_since_improvement > 5:
            actual_decay += 0.02 * min(gens_since_improvement - 5, 10)

        # Immature recipes decay faster — haven't proven themselves
        if not recipe.is_mature:
            actual_decay *= 2.0

        recipe.strength = max(0.0, recipe.strength - actual_decay)

        if recipe.strength <= 0.0:
            recipe.is_alive = False
            recipe.death_generation = current_generation
            gen_logger.info(
                f"💀 Recipe {recipe.recipe_id[:8]} died at gen {current_generation} "
                f"(active for {recipe.total_generations_active} gens, "
                f"best={recipe.best_fitness:.4f}, improvements={len(recipe.improvement_log)})"
            )
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Dormant re-emergence — resurrect a recipe from the graveyard
    # ------------------------------------------------------------------ #
    @staticmethod
    def resurrect_recipe(dead_recipe: NovelMethodRecipe,
                         generation: int) -> NovelMethodRecipe:
        """Resurrect a dead recipe from the graveyard ('distant gene' pattern).

        The recipe is given a fresh ID but retains its structural DNA (steps).
        Strength is reset to a moderate baseline.  The ``times_rediscovered``
        counter is incremented to track how often this family resurfaces.
        """
        import copy as _copy
        import uuid as _uuid

        reborn = _copy.deepcopy(dead_recipe)
        old_id = reborn.recipe_id
        reborn.recipe_id = str(_uuid.uuid4())
        reborn.is_alive = True
        reborn.death_generation = None
        reborn.strength = 0.4  # Moderate start — must re-prove itself
        reborn.times_rediscovered += 1
        reborn.generation_last_improved = generation
        # Keep improvement_log from its past life for context
        reborn.improvement_log.append({
            "type": "resurrection",
            "previous_recipe_id": old_id,
            "resurrection_generation": generation,
            "times_rediscovered": reborn.times_rediscovered,
        })
        reborn.parent_recipe_ids = [old_id]

        gen_logger.info(
            f"🧟 Recipe {old_id[:8]} resurrected as {reborn.recipe_id[:8]} "
            f"at gen {generation} (rediscovered {reborn.times_rediscovered}x, "
            f"family={reborn.family_key})"
        )
        return reborn

    # ------------------------------------------------------------------ #
    #  Cross-family combination  (sub-novel methods)
    # ------------------------------------------------------------------ #
    @staticmethod
    def should_combine(recipe_a: NovelMethodRecipe,
                       recipe_b: NovelMethodRecipe) -> bool:
        """True when two recipes are structurally different enough to
        warrant combination rather than simple inheritance."""
        return recipe_a.step_types != recipe_b.step_types

    @staticmethod
    def combine_recipes(recipe_a: NovelMethodRecipe,
                        recipe_b: NovelMethodRecipe,
                        generation: int) -> NovelMethodRecipe:
        """Create a sub-novel method blending steps from two recipe families.

        The algorithm:
        1. Pick a blend strategy (A-dominant / balanced / B-dominant).
        2. Walk through steps from both recipes and probabilistically
           include them based on the blend weight.
        3. Prefer steps with higher contribution scores.
        4. Cap total pipeline length at ``MAX_PIPELINE_LEN``.

        The resulting recipe must still prove itself — contribution scores
        are reset and it starts with no improvements logged.
        """
        import copy as _copy
        import uuid as _uuid

        # Sort each recipe's steps by contribution (best first)
        steps_a = sorted(recipe_a.steps,
                         key=lambda s: s.contribution_score, reverse=True)
        steps_b = sorted(recipe_b.steps,
                         key=lambda s: s.contribution_score, reverse=True)

        # Random blend strategy
        strategy, a_weight = random.choice([
            ("a_dominant", 0.70),
            ("balanced",   0.50),
            ("b_dominant", 0.30),
        ])

        combined_steps: List[RecipeStep] = []
        max_steps = min(max(len(steps_a), len(steps_b)),
                        RecipeEvolver.MAX_PIPELINE_LEN)

        for i in range(max_steps):
            pick_a = random.random() < a_weight
            if pick_a and i < len(steps_a):
                step = _copy.deepcopy(steps_a[i])
            elif i < len(steps_b):
                step = _copy.deepcopy(steps_b[i])
            elif i < len(steps_a):
                step = _copy.deepcopy(steps_a[i])
            else:
                continue
            # Reset contribution — must prove itself in new context
            step.contribution_score = 0.0
            step.generation_added = generation
            combined_steps.append(step)

        if not combined_steps:
            combined_steps = [RecipeStep(step_type="rle",
                                        params={"min_run": 3},
                                        generation_added=generation)]

        # Inherit discovery seed from the higher-fitness parent
        ds = (recipe_a.discovery_seed
              if recipe_a.best_fitness >= recipe_b.best_fitness
              else recipe_b.discovery_seed)
        mrr = (recipe_a.rle_min_run
               if recipe_a.best_fitness >= recipe_b.best_fitness
               else recipe_b.rle_min_run)

        combined = NovelMethodRecipe(
            recipe_id=str(_uuid.uuid4()),
            steps=combined_steps,
            rle_min_run=mrr,
            discovery_seed=ds,
            generation_created=generation,
            parent_recipe_ids=[recipe_a.recipe_id, recipe_b.recipe_id],
            # v0.9.10: Inherit averaged strength from parents — must still
            # prove itself but starts with genetic momentum.
            strength=min(1.0, (recipe_a.strength + recipe_b.strength) / 2.0),
            improvement_log=[{
                "type": "combination",
                "strategy": strategy,
                "a_weight": a_weight,
                "parent_a_id": recipe_a.recipe_id,
                "parent_b_id": recipe_b.recipe_id,
                "parent_a_fitness": recipe_a.best_fitness,
                "parent_b_fitness": recipe_b.best_fitness,
                "parent_a_strength": recipe_a.strength,
                "parent_b_strength": recipe_b.strength,
                "generation": generation,
            }],
        )
        gen_logger.info(
            f"🧬 Combined recipes {recipe_a.recipe_id[:8]} + "
            f"{recipe_b.recipe_id[:8]} → {combined.recipe_id[:8]} "
            f"({strategy}, weight_a={a_weight})"
        )
        return combined

    # ------------------------------------------------------------------ #
    #  Build closures from recipe
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_from_recipe(recipe: NovelMethodRecipe,
                          generator: "NovelCompressionGenerator"
                          ) -> Tuple[Callable, Callable]:
        """Convert a recipe's step list into compress/decompress closures.

        Reuses the existing ``_build_pipeline`` infrastructure but feeds
        an arbitrary steps list (not restricted to pre-defined pipeline
        names).
        """
        return generator._build_pipeline_from_steps(
            [s.step_type for s in recipe.steps],
            discovery_seed=recipe.discovery_seed,
            rle_min_run=recipe.rle_min_run,
        )


# ---------------------------------------------------------------------------
#  METHOD REGISTRY / GRAVEYARD  (v0.9.10)
# ---------------------------------------------------------------------------

class MethodRegistry:
    """Persistent catalogue of ALL recipe families ever discovered.

    Every recipe family (identified by its ``family_key`` — the ordered
    step-type signature) gets an entry.  Living recipes have their stats
    updated each generation; dead recipes remain in the graveyard so they
    can be recognized if they re-emerge (distant-gene pattern).

    Storage format: ``data/method_registry.json``
    """

    def __init__(self, registry_path: Optional[str] = None):
        if registry_path is None:
            try:
                from .config import DATA_DIR
                registry_path = os.path.join(DATA_DIR, "method_registry.json")
            except ImportError:
                registry_path = os.path.join("data", "method_registry.json")
        self._path = registry_path
        # family_key → registry entry dict
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ---- persistence ----
    def _load(self):
        """Load the registry from disk (non-fatal on failure)."""
        if not os.path.isfile(self._path):
            return
        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._entries = data.get('families', {})
        except Exception:
            self._entries = {}

    def save(self):
        """Write the registry to disk."""
        import datetime as _dt
        try:
            os.makedirs(os.path.dirname(self._path) or '.', exist_ok=True)
            data = {
                'version': 1,
                'updated_at': _dt.datetime.now().isoformat(),
                'total_families': len(self._entries),
                'alive_count': sum(1 for e in self._entries.values() if e.get('is_alive')),
                'dead_count': sum(1 for e in self._entries.values() if not e.get('is_alive')),
                'families': self._entries,
            }
            with open(self._path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            gen_logger.debug(f"Method registry save failed (non-fatal): {e}")

    # ---- registration ----
    def register(self, recipe: NovelMethodRecipe):
        """Register or update a recipe family in the catalogue.

        If this family has been seen before, update its stats.
        If it was dead and is now alive, increment ``times_rediscovered``.
        """
        fk = recipe.family_key
        existing = self._entries.get(fk)

        if existing is None:
            # New family — first sighting
            self._entries[fk] = {
                'family_key': fk,
                'first_recipe_id': recipe.recipe_id,
                'latest_recipe_id': recipe.recipe_id,
                'pipeline_name': recipe.get_pipeline_name(),
                'step_count': len(recipe.steps),
                'generation_first_seen': recipe.generation_created,
                'generation_last_seen': recipe.generation_last_improved or recipe.generation_created,
                'peak_fitness': recipe.best_fitness,
                'peak_strength': recipe.strength,
                'total_improvements': len(recipe.improvement_log),
                'total_agents_used': 1,
                'is_alive': recipe.is_alive,
                'death_generation': recipe.death_generation,
                'times_rediscovered': recipe.times_rediscovered,
                'total_generations_active': recipe.total_generations_active,
                'recipe_snapshot': recipe.to_dict(),
            }
        else:
            # Known family — update stats
            existing['latest_recipe_id'] = recipe.recipe_id
            existing['generation_last_seen'] = max(
                existing.get('generation_last_seen', 0),
                recipe.generation_last_improved or recipe.generation_created,
            )
            existing['peak_fitness'] = max(existing.get('peak_fitness', 0), recipe.best_fitness)
            existing['peak_strength'] = max(existing.get('peak_strength', 0), recipe.strength)
            existing['total_improvements'] = max(
                existing.get('total_improvements', 0), len(recipe.improvement_log)
            )
            existing['is_alive'] = recipe.is_alive
            existing['death_generation'] = recipe.death_generation
            existing['times_rediscovered'] = max(
                existing.get('times_rediscovered', 0), recipe.times_rediscovered
            )
            existing['total_generations_active'] = max(
                existing.get('total_generations_active', 0),
                recipe.total_generations_active,
            )
            # Update snapshot if this recipe is better
            if recipe.best_fitness >= existing.get('peak_fitness', 0):
                existing['recipe_snapshot'] = recipe.to_dict()

    def register_death(self, recipe: NovelMethodRecipe, generation: int):
        """Mark a recipe family as dead in the registry."""
        recipe.is_alive = False
        recipe.death_generation = generation
        self.register(recipe)

    def increment_agent_count(self, family_key: str):
        """Bump the total_agents_used counter for a family."""
        entry = self._entries.get(family_key)
        if entry:
            entry['total_agents_used'] = entry.get('total_agents_used', 0) + 1

    # ---- graveyard queries ----
    def get_dead_recipes(self) -> List[NovelMethodRecipe]:
        """Return all dead recipe families as NovelMethodRecipe objects."""
        results = []
        for entry in self._entries.values():
            if entry.get('is_alive'):
                continue
            snapshot = entry.get('recipe_snapshot')
            if snapshot:
                try:
                    results.append(NovelMethodRecipe.from_dict(snapshot))
                except Exception:
                    pass
        return results

    def get_all_entries(self) -> Dict[str, Dict[str, Any]]:
        """Return the full registry dict (for API/UI consumption)."""
        return dict(self._entries)

    def get_summary(self) -> Dict[str, Any]:
        """Quick stats for API responses."""
        alive = [e for e in self._entries.values() if e.get('is_alive')]
        dead = [e for e in self._entries.values() if not e.get('is_alive')]
        return {
            'total_families': len(self._entries),
            'alive_families': len(alive),
            'dead_families': len(dead),
            'total_rediscoveries': sum(e.get('times_rediscovered', 0) for e in self._entries.values()),
            'peak_fitness_ever': max((e.get('peak_fitness', 0) for e in self._entries.values()), default=0),
            'top_families': sorted(
                [{'family': e.get('family_key', '?'),
                  'peak_fitness': e.get('peak_fitness', 0),
                  'is_alive': e.get('is_alive', False),
                  'times_rediscovered': e.get('times_rediscovered', 0)}
                 for e in self._entries.values()],
                key=lambda x: x['peak_fitness'], reverse=True
            )[:10],
        }

    def pick_graveyard_recipe(self) -> Optional[NovelMethodRecipe]:
        """Randomly select a dead recipe for potential re-emergence.

        Biased toward recipes that had higher peak fitness — they're
        more likely to be useful if re-introduced.  Returns None if the
        graveyard is empty.
        """
        dead = self.get_dead_recipes()
        if not dead:
            return None
        # Weight by peak fitness (higher = more likely to be resurrected)
        weights = [max(r.best_fitness, 0.01) for r in dead]
        total_w = sum(weights)
        if total_w <= 0:
            return random.choice(dead)
        weights = [w / total_w for w in weights]
        return random.choices(dead, weights=weights, k=1)[0]


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

    def _build_pipeline_from_steps(self, step_list: List[str],
                                   discovery_seed: Optional[int] = None,
                                   rle_min_run: Optional[int] = None) -> Tuple[Callable, Callable]:
        """Build compress/decompress closures from an arbitrary steps list.

        Works like ``_build_pipeline`` but accepts a raw list of step names
        instead of a pipeline name.  Used by ``RecipeEvolver.build_from_recipe``
        for recipes whose step list is not a pre-defined pipeline.

        Discovery transforms (xor/perm/block_shuffle) appearing in
        *step_list* are extracted and converted to a discovery_seed-style
        pre-/post-processor; core steps (rle/bwt/mtf/delta/bpe) go through
        the normal compress/decompress loop.
        """
        core_steps = [s for s in step_list
                      if s in ("rle", "bwt", "mtf", "delta", "bpe")]
        if not core_steps:
            core_steps = ["rle"]

        # If a discovery step is in the list, derive seed from it
        disc_step = next((s for s in step_list
                          if s in ("xor", "perm", "block_shuffle")), None)
        if disc_step and discovery_seed is None:
            discovery_seed = hash(disc_step) & 0x7FFFFFFF

        # Temporarily register as a pipeline and delegate
        pipeline_key = "_".join(core_steps)
        self.PIPELINES[pipeline_key] = core_steps
        try:
            cfn, dfn = self._build_pipeline(
                pipeline_key,
                discovery_seed=discovery_seed,
                rle_min_run=rle_min_run,
            )
        finally:
            # Remove temporary entry only if it wasn't a standard pipeline
            _STANDARD = {"rle_only", "bwt_rle", "bwt_mtf_rle", "delta_rle",
                         "bpe_rle", "bpe_only", "delta_bpe", "bwt_bpe"}
            if pipeline_key not in _STANDARD:
                self.PIPELINES.pop(pipeline_key, None)
        return cfn, dfn

    def build_method_from_recipe(self, recipe: "NovelMethodRecipe",
                                 method_name: Optional[str] = None
                                 ) -> CompressionMethod:
        """Build a full ``CompressionMethod`` from a ``NovelMethodRecipe``.

        The resulting method has ``is_novelty`` set according to the
        recipe's maturity (only *mature* recipes are genuinely novel).
        """
        cfn, dfn = RecipeEvolver.build_from_recipe(recipe, self)
        if not self._verify_invertibility(cfn, dfn):
            # Fallback: strip discovery transforms and retry
            cfn, dfn = self._build_pipeline("rle_only", rle_min_run=recipe.rle_min_run)
            gen_logger.debug(
                f"Recipe {recipe.recipe_id[:8]} failed invertibility, "
                f"fell back to rle_only"
            )

        if method_name is None:
            method_name = f"recipe_{recipe.recipe_id[:8]}"

        desc = f"Recipe: {recipe.get_pipeline_name()}, min_run={recipe.rle_min_run}"
        if recipe.discovery_seed is not None:
            desc += f", seed={recipe.discovery_seed}"
        if recipe.is_mature:
            desc += f" [MATURE, {len(recipe.improvement_log)} improvements]"

        return CompressionMethod(
            name=method_name,
            language=CompressionLanguage.HYBRID,
            compress_fn=cfn,
            decompress_fn=dfn,
            description=desc,
            author="RecipeEvolver",
            is_novelty=recipe.is_mature,   # Only mature recipes are "novel"
            metadata=recipe.to_legacy_metadata(),
        )

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
