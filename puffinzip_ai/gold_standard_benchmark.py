# PuffinZipAI - Gold Standard Head-to-Head Benchmark
"""
After each generation, pits the best agent head-to-head against every standard
compression method (gzip, bz2, lzma, zlib, zstd).

* If the agent **outcompetes all** baselines on every test item the generation
  is automatically saved as a **gold standard** checkpoint.
* If it **fails** to beat one or more baselines, a compact diagnostics folder
  is written with the compressed + decompressed artefacts alongside a
  human-readable summary so the developer can see *exactly* what went wrong.
"""

import gzip
import bz2
import lzma
import zlib
import json
import logging
import os
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    import zstandard  # type: ignore
    _ZSTD_AVAILABLE = True
except ImportError:
    _ZSTD_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _measure_compressed_size(text: str) -> int:
    """Measure the conceptual byte-count of AI-compressed text.

    Since the pipeline now uses Latin-1 (1 byte per char for bytes 0-255),
    chars in that range are counted as 1 byte each.  Higher codepoints (e.g.
    BPE Private-Use-Area substitution chars) are counted at their minimum
    UTF-8 byte width so the comparison with baselines remains fair.
    """
    try:
        return len(text.encode('latin-1'))
    except UnicodeEncodeError:
        # Fallback: count per char — 1 byte for Latin-1, multi for higher
        size = 0
        for ch in text:
            cp = ord(ch)
            if cp <= 0xFF:
                size += 1
            elif cp <= 0x7FF:
                size += 2
            else:
                size += 3
        return size


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ItemResult:
    """Outcome of compressing one test item with one method."""
    method: str
    original_size: int
    compressed_size: int
    ratio_pct: float          # (1 - compressed/original) * 100
    verified: bool             # decompress == original?
    error: Optional[str] = None


@dataclass
class HeadToHeadResult:
    """Per-item comparison: AI vs every baseline."""
    item_index: int
    item_preview: str          # first 80 chars of the original
    original_size: int
    ai_result: Optional[ItemResult] = None
    baseline_results: List[ItemResult] = field(default_factory=list)
    ai_beats_all: bool = False  # True iff AI compressed < every baseline


@dataclass
class GenerationBenchmarkReport:
    """Full report for one generation."""
    generation: int
    timestamp: str
    num_items: int
    agent_id: str
    agent_fitness: float
    items: List[HeadToHeadResult] = field(default_factory=list)
    gold_standard: bool = False   # True when AI beats ALL baselines on ALL items
    summary: str = ""


@dataclass
class RobustnessBenchmarkReport:
    """Robustness gold-standard report for one generation.

    The anti-corruption analogue of :class:`GenerationBenchmarkReport`.  The
    head-to-head is run on CORRUPTED data using the best anti-corruption agent:
    an item is a "win" when the agent both survives the noise (verified
    round-trip) AND beats every baseline compressor's size on that corrupted
    stream.  ``win_rate`` gates the corruption-track advancement.
    """
    generation: int
    timestamp: str
    num_items: int
    agent_id: str
    robustness_fitness: float
    wins: int = 0
    survived: int = 0            # items the agent round-tripped without error
    win_rate: float = 0.0        # wins / num_items
    survival_rate: float = 0.0   # survived / num_items
    gold_standard: bool = False  # True when the agent beat ALL baselines on ALL items
    summary: str = ""


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------

_BASELINE_METHODS: Dict[str, Tuple[Callable, Callable]] = {}


def _init_baselines():
    """Lazily populate baseline compress/decompress pairs."""
    global _BASELINE_METHODS
    if _BASELINE_METHODS:
        return

    _BASELINE_METHODS["gzip"] = (
        lambda d: gzip.compress(d, compresslevel=9),
        gzip.decompress,
    )
    _BASELINE_METHODS["bz2"] = (
        lambda d: bz2.compress(d, compresslevel=9),
        bz2.decompress,
    )
    _BASELINE_METHODS["lzma"] = (
        lambda d: lzma.compress(d, preset=9),
        lzma.decompress,
    )
    _BASELINE_METHODS["zlib"] = (
        lambda d: zlib.compress(d, level=9),
        zlib.decompress,
    )

    if _ZSTD_AVAILABLE:
        import zstandard as _zstd
        cctx = _zstd.ZstdCompressor(level=19)
        dctx = _zstd.ZstdDecompressor()
        _BASELINE_METHODS["zstd"] = (cctx.compress, dctx.decompress)


def _compress_baseline(method: str, data_bytes: bytes) -> ItemResult:
    """Compress *data_bytes* with a named baseline and verify round-trip."""
    _init_baselines()
    compress_fn, decompress_fn = _BASELINE_METHODS[method]
    original_size = len(data_bytes)
    try:
        compressed = compress_fn(data_bytes)
        decompressed = decompress_fn(compressed)
        verified = decompressed == data_bytes
        compressed_size = len(compressed)
        ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0.0
        return ItemResult(
            method=method,
            original_size=original_size,
            compressed_size=compressed_size,
            ratio_pct=ratio,
            verified=verified,
        )
    except Exception as exc:
        return ItemResult(
            method=method,
            original_size=original_size,
            compressed_size=original_size,
            ratio_pct=0.0,
            verified=False,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# AI compression wrapper
# ---------------------------------------------------------------------------

def _compress_with_agent(agent_ai, item_text: str, rle_compress_fn, rle_decompress_fn) -> Tuple[str, str, str]:
    """Use the agent's Q-table to pick an action and compress *item_text*.

    Returns
    -------
    (compressed_text, decompressed_text, action_name)
    """
    state_idx = agent_ai._get_state_representation(item_text)
    action_idx = agent_ai._choose_action(state_idx, use_exploration=False)
    action_name = agent_ai.action_names.get(action_idx, f"Unknown({action_idx})")

    compressed_text = item_text
    decompressed_text = item_text

    rle_min_run = getattr(agent_ai, 'rle_min_encodable_run_length', 2)

    if action_name == "RLE":
        compressed_text = rle_compress_fn(item_text, method="simple",
                                          min_run_len_override=rle_min_run)
        decompressed_text = rle_decompress_fn(compressed_text, method="simple",
                                              min_run_len_override=rle_min_run)
    elif action_name == "AdvancedRLE":
        compressed_text = rle_compress_fn(item_text, method="advanced")
        decompressed_text = rle_decompress_fn(compressed_text, method="advanced")
    elif action_name == "NovelMethod":
        novel_c = getattr(agent_ai, '_novel_compress_fn', None)
        novel_d = getattr(agent_ai, '_novel_decompress_fn', None)
        if novel_c and novel_d:
            try:
                compressed_text = novel_c(item_text)
                decompressed_text = novel_d(compressed_text)
            except Exception:
                compressed_text = item_text
                decompressed_text = item_text
    elif action_name == "ReferenceMethod":
        ref_c = getattr(agent_ai, '_reference_compress_fn', None)
        ref_d = getattr(agent_ai, '_reference_decompress_fn', None)
        if ref_c and ref_d:
            try:
                item_bytes = item_text.encode('utf-8') if isinstance(item_text, str) else item_text
                compressed_bytes = ref_c(item_bytes)
                decompressed_bytes = ref_d(compressed_bytes)
                # For size comparison keep byte length; store placeholder string
                compressed_text = "X" * len(compressed_bytes)
                decompressed_text = (decompressed_bytes.decode('utf-8')
                                     if isinstance(decompressed_bytes, bytes)
                                     else decompressed_bytes)
            except Exception:
                compressed_text = item_text
                decompressed_text = item_text
    # else NoCompression — compressed_text stays == item_text

    return compressed_text, decompressed_text, action_name


# ---------------------------------------------------------------------------
# Core benchmark class
# ---------------------------------------------------------------------------

class GoldStandardBenchmark:
    """Runs a head-to-head benchmark of the generation's best agent against
    all standard compression methods after each generation."""

    # Relative to the project root
    ARTIFACTS_DIR = "gold_standard_results"
    # Maximum total size of artifacts directory before old generations are pruned
    MAX_ARTIFACTS_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB
    # Keep the most recent N generations' eval dirs uncompressed on disk
    # so external viewers (Explorer, WebUI cache) can read them directly.
    KEEP_RECENT_GENS_UNCOMPRESSED = 100

    def __init__(self, project_root: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.project_root = project_root or os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        self.artifacts_dir = os.path.join(self.project_root, self.ARTIFACTS_DIR)
        os.makedirs(self.artifacts_dir, exist_ok=True)

        self.logger = logger or self._default_logger()

        # Lazy imports of AI-level compress / decompress
        self._rle_compress = None
        self._rle_decompress = None
        self._ensure_rle_fns()

        # One-time migration: compress any uncompressed eval dirs from prior runs
        self._migrate_uncompressed_evals()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_logger() -> logging.Logger:
        lg = logging.getLogger("GoldStandardBenchmark")
        if not lg.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
            lg.addHandler(h)
            lg.setLevel(logging.INFO)
        return lg

    def _ensure_rle_fns(self):
        """Lazy-import RLE compress/decompress so we don't create circular deps."""
        if self._rle_compress is not None:
            return
        try:
            from .rle_utils import rle_compress, rle_decompress
            self._rle_compress = rle_compress
            self._rle_decompress = rle_decompress
        except ImportError:
            self.logger.warning("Could not import rle_utils — AI compression unavailable.")

    # ------------------------------------------------------------------
    # Artifact compression & size management
    # ------------------------------------------------------------------

    def _compress_eval_to_zip(self, eval_dir: str) -> Optional[str]:
        """Compress an eval directory into a ``.zip`` archive and remove the
        loose directory tree.

        Parameters
        ----------
        eval_dir : str
            Absolute path to the ``eval_NNN`` directory to compress.

        Returns
        -------
        str or None
            Path to the resulting ``.zip`` file, or ``None`` on failure.
        """
        zip_path = eval_dir + ".zip"
        if not os.path.isdir(eval_dir):
            return zip_path if os.path.isfile(zip_path) else None
        # If a zip already exists (e.g. from an interrupted prior run),
        # just remove the loose directory — the zip is already complete.
        if os.path.isfile(zip_path):
            shutil.rmtree(eval_dir, ignore_errors=True)
            return zip_path
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED,
                                 compresslevel=6) as zf:
                for root, _dirs, files in os.walk(eval_dir):
                    for fname in files:
                        full = os.path.join(root, fname)
                        arcname = os.path.relpath(full, os.path.dirname(eval_dir))
                        zf.write(full, arcname)
            shutil.rmtree(eval_dir, ignore_errors=True)
            return zip_path
        except Exception as exc:
            self.logger.warning(f"Failed to zip {eval_dir}: {exc}")
            # Leave the uncompressed directory in place so data is not lost
            return None

    def _compress_old_generations(self, current_generation: int):
        """Compress eval dirs for generations older than the keep-recent window.

        Only generations whose number is <= ``current_generation - KEEP_RECENT_GENS_UNCOMPRESSED``
        have their loose ``eval_*`` dirs compressed into zips.  Recent
        generations are left uncompressed so external tools can read them
        directly.
        """
        cutoff = current_generation - self.KEEP_RECENT_GENS_UNCOMPRESSED
        if cutoff < 1:
            return  # Nothing old enough to compress yet
        if not os.path.isdir(self.artifacts_dir):
            return
        compressed = 0
        try:
            for gen_entry in sorted(os.listdir(self.artifacts_dir)):
                gen_path = os.path.join(self.artifacts_dir, gen_entry)
                if not os.path.isdir(gen_path) or not gen_entry.startswith("gen_"):
                    continue
                gen_num = self._extract_gen_number(gen_entry)
                if gen_num is None or gen_num > cutoff:
                    continue  # Recent — leave uncompressed
                for eval_entry in sorted(os.listdir(gen_path)):
                    eval_path = os.path.join(gen_path, eval_entry)
                    if (
                        os.path.isdir(eval_path)
                        and eval_entry.startswith("eval_")
                    ):
                        if self._compress_eval_to_zip(eval_path):
                            compressed += 1
        except Exception as exc:
            self.logger.warning(f"Compress-old-generations failed: {exc}")
        if compressed:
            self.logger.info(
                f"Compressed {compressed} eval dir(s) for gens <= {cutoff}"
            )

    def _migrate_uncompressed_evals(self):
        """One-time migration: walk every ``gen_*`` directory and compress any
        ``eval_*`` sub-directories that have not yet been zipped.

        Called from ``__init__`` so existing loose artifacts are compressed
        transparently.  Only runs once per process; subsequent instantiations
        skip if nothing remains.

        Respects :pyattr:`KEEP_RECENT_GENS_UNCOMPRESSED` — the most recent N
        generations are left uncompressed.
        """
        if not os.path.isdir(self.artifacts_dir):
            return

        # Determine the maximum generation number on disk so we can figure
        # out which gens are "recent" and should stay uncompressed.
        max_gen = 0
        try:
            for entry in os.listdir(self.artifacts_dir):
                gn = self._extract_gen_number(entry)
                if gn is not None and gn > max_gen:
                    max_gen = gn
        except Exception:
            pass
        cutoff = max_gen - self.KEEP_RECENT_GENS_UNCOMPRESSED

        migrated = 0
        try:
            for gen_entry in sorted(os.listdir(self.artifacts_dir)):
                gen_path = os.path.join(self.artifacts_dir, gen_entry)
                if not os.path.isdir(gen_path) or not gen_entry.startswith("gen_"):
                    continue
                gen_num = self._extract_gen_number(gen_entry)
                if gen_num is not None and gen_num > cutoff:
                    continue  # Recent — keep uncompressed
                for eval_entry in sorted(os.listdir(gen_path)):
                    eval_path = os.path.join(gen_path, eval_entry)
                    if (
                        os.path.isdir(eval_path)
                        and eval_entry.startswith("eval_")
                    ):
                        if self._compress_eval_to_zip(eval_path):
                            migrated += 1
                # If gen dir is now empty (shouldn't be, but be safe), leave it
        except Exception as exc:
            self.logger.warning(f"Migration of uncompressed evals failed: {exc}")
        if migrated:
            self.logger.info(
                f"Migrated {migrated} eval dir(s) to .zip in {self.artifacts_dir}"
            )

    def _get_artifacts_total_bytes(self) -> int:
        """Return total on-disk size (bytes) of the artifacts directory."""
        total = 0
        try:
            for root, _dirs, files in os.walk(self.artifacts_dir):
                for fname in files:
                    try:
                        total += os.path.getsize(os.path.join(root, fname))
                    except OSError:
                        pass
        except Exception:
            pass
        return total

    @staticmethod
    def _extract_gen_number(name: str) -> Optional[int]:
        """Parse the integer from a ``gen_<N>`` directory name."""
        if name.startswith("gen_"):
            suffix = name[4:]
            if suffix.isdigit():
                return int(suffix)
        return None

    def _enforce_size_limit(self):
        """If ``gold_standard_results/`` exceeds :pyattr:`MAX_ARTIFACTS_BYTES`,
        delete the **oldest** generation directories (lowest gen number first)
        until the total drops below the limit.

        Protects the most recent :pyattr:`KEEP_RECENT_GENS_UNCOMPRESSED`
        generations from deletion.
        """
        total = self._get_artifacts_total_bytes()
        if total <= self.MAX_ARTIFACTS_BYTES:
            return

        # Build sorted list of (gen_number, dir_path)
        gen_entries: List[tuple] = []
        max_gen = 0
        try:
            for entry in os.listdir(self.artifacts_dir):
                full = os.path.join(self.artifacts_dir, entry)
                gen_num = self._extract_gen_number(entry)
                if gen_num is not None and os.path.isdir(full):
                    gen_entries.append((gen_num, full))
                    if gen_num > max_gen:
                        max_gen = gen_num
        except Exception:
            return

        gen_entries.sort()  # oldest generation first
        protect_cutoff = max_gen - self.KEEP_RECENT_GENS_UNCOMPRESSED

        removed = 0
        for gen_num, gen_path in gen_entries:
            if total <= self.MAX_ARTIFACTS_BYTES:
                break
            if gen_num > protect_cutoff:
                break  # The rest are recent — do not delete
            # Measure this gen folder's size before deletion
            gen_size = 0
            for root, _dirs, files in os.walk(gen_path):
                for fname in files:
                    try:
                        gen_size += os.path.getsize(os.path.join(root, fname))
                    except OSError:
                        pass
            try:
                shutil.rmtree(gen_path, ignore_errors=True)
                total -= gen_size
                removed += 1
                self.logger.info(
                    f"Size limit: removed gen_{gen_num} "
                    f"({gen_size / (1024*1024):.1f} MB freed)"
                )
            except Exception as exc:
                self.logger.warning(f"Could not remove {gen_path}: {exc}")

        if removed:
            self.logger.info(
                f"Size cleanup: removed {removed} generation(s), "
                f"~{self._get_artifacts_total_bytes() / (1024*1024):.0f} MB remaining"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def benchmark_generation(
        self,
        generation: int,
        best_agent,                        # EvolvingAgent
        test_items: List[str],
        checkpoint_save_fn: Optional[Callable] = None,
        gui_msg_fn: Optional[Callable] = None,
    ) -> GenerationBenchmarkReport:
        """Run head-to-head and return a report.

        Parameters
        ----------
        generation : int
            Current generation number (1-based).
        best_agent : EvolvingAgent
            The top agent from this generation.
        test_items : list[str]
            The benchmark texts used this generation.
        checkpoint_save_fn : callable, optional
            ``fn(name: str) -> bool``  — called to save a gold-standard checkpoint.
        gui_msg_fn : callable, optional
            ``fn(message: str, level: str)``  — for sending status to GUI/console.

        Returns
        -------
        GenerationBenchmarkReport
        """
        self._ensure_rle_fns()
        _init_baselines()

        agent_ai = best_agent.get_puffin_ai() if hasattr(best_agent, 'get_puffin_ai') else best_agent
        agent_id = getattr(best_agent, 'agent_id', 'unknown')
        agent_fitness = (best_agent.get_fitness()
                         if hasattr(best_agent, 'get_fitness') else 0.0) or 0.0

        report = GenerationBenchmarkReport(
            generation=generation,
            timestamp=datetime.now().isoformat(),
            num_items=len(test_items),
            agent_id=agent_id,
            agent_fitness=agent_fitness,
        )

        if not test_items:
            report.summary = "No test items — skipped."
            return report

        all_items_beat = True
        wins = 0
        losses = 0
        details_lines: List[str] = []

        for idx, item_text in enumerate(test_items):
            if not item_text:
                continue

            original_bytes = item_text.encode('utf-8') if isinstance(item_text, str) else item_text
            original_size = len(original_bytes)

            h2h = HeadToHeadResult(
                item_index=idx,
                item_preview=item_text[:80],
                original_size=original_size,
            )

            # --- AI compression ---
            if self._rle_compress and self._rle_decompress:
                try:
                    comp_text, decomp_text, action = _compress_with_agent(
                        agent_ai, item_text, self._rle_compress, self._rle_decompress
                    )
                    ai_compressed_size = _measure_compressed_size(comp_text) if isinstance(comp_text, str) else len(comp_text)
                    ai_verified = (decomp_text == item_text)
                    ai_ratio = (1 - ai_compressed_size / original_size) * 100 if original_size > 0 else 0.0
                    h2h.ai_result = ItemResult(
                        method=f"PuffinZipAI ({action})",
                        original_size=original_size,
                        compressed_size=ai_compressed_size,
                        ratio_pct=ai_ratio,
                        verified=ai_verified,
                    )
                except Exception as exc:
                    h2h.ai_result = ItemResult(
                        method="PuffinZipAI (error)",
                        original_size=original_size,
                        compressed_size=original_size,
                        ratio_pct=0.0,
                        verified=False,
                        error=str(exc),
                    )

            # --- Baseline compression ---
            for method_name in _BASELINE_METHODS:
                bl_result = _compress_baseline(method_name, original_bytes)
                h2h.baseline_results.append(bl_result)

            # --- Head-to-head verdict ---
            if h2h.ai_result and h2h.ai_result.verified and not h2h.ai_result.error:
                ai_size = h2h.ai_result.compressed_size
                beaten_all = all(
                    ai_size < bl.compressed_size
                    for bl in h2h.baseline_results
                    if not bl.error
                )
                h2h.ai_beats_all = beaten_all
                if beaten_all:
                    wins += 1
                else:
                    losses += 1
                    all_items_beat = False
            else:
                all_items_beat = False
                losses += 1

            report.items.append(h2h)

            # Build per-item detail line
            ai_sz = h2h.ai_result.compressed_size if h2h.ai_result else original_size
            best_bl = min(
                (bl.compressed_size for bl in h2h.baseline_results if not bl.error),
                default=original_size,
            )
            best_bl_name = "N/A"
            for bl in h2h.baseline_results:
                if not bl.error and bl.compressed_size == best_bl:
                    best_bl_name = bl.method
                    break
            verdict = "WIN" if h2h.ai_beats_all else "LOSS"
            details_lines.append(
                f"  Item {idx:>3}: orig={original_size:>6}B  AI={ai_sz:>6}B  "
                f"best_baseline={best_bl:>6}B ({best_bl_name:<5})  => {verdict}"
            )

        # --- Overall verdict ---
        report.gold_standard = all_items_beat and wins > 0
        report.summary = (
            f"Generation {generation} | Agent {agent_id} (fitness {agent_fitness:.4f})\n"
            f"Items: {len(test_items)}  Wins: {wins}  Losses: {losses}\n"
            f"Verdict: {'*** GOLD STANDARD ***' if report.gold_standard else 'DID NOT BEAT ALL BASELINES'}\n"
            + "\n".join(details_lines)
        )

        # --- Log to main logger ---
        if report.gold_standard:
            self.logger.info(f"GOLD STANDARD achieved at generation {generation}!")
            self.logger.info(report.summary)
        else:
            self.logger.info(f"Gen {generation} head-to-head: {wins}/{len(test_items)} wins.")
            self.logger.debug(report.summary)

        # --- GUI notification ---
        if gui_msg_fn:
            if report.gold_standard:
                gui_msg_fn(
                    f"Gen {generation}: GOLD STANDARD — agent beat ALL baselines on ALL items!",
                    "info",
                )
            else:
                gui_msg_fn(
                    f"Gen {generation}: Head-to-head {wins}/{len(test_items)} wins vs baselines.",
                    "info",
                )

        # --- Gold standard checkpoint ---
        if report.gold_standard and checkpoint_save_fn:
            try:
                cp_name = f"gold_standard_gen{generation}"
                ok = checkpoint_save_fn(cp_name)
                if ok:
                    self.logger.info(f"Gold standard checkpoint saved: {cp_name}")
                    if gui_msg_fn:
                        gui_msg_fn(f"Gen {generation}: Gold standard checkpoint saved!", "info")
                else:
                    self.logger.warning(f"Gold standard checkpoint save returned False.")
            except Exception as exc:
                self.logger.error(f"Failed to save gold standard checkpoint: {exc}", exc_info=True)

        # --- Failure diagnostics: save artefacts ---
        if not report.gold_standard:
            self._save_failure_artifacts(generation, report, test_items, agent_ai)

        return report

    # ------------------------------------------------------------------
    # Robustness gold-standard (anti-corruption track)
    # ------------------------------------------------------------------

    def benchmark_robustness(
        self,
        generation: int,
        best_anti_agent,                   # EvolvingAgent (best anti_corruption)
        corrupted_items: List[str],
        gui_msg_fn: Optional[Callable] = None,
    ) -> RobustnessBenchmarkReport:
        """Robustness head-to-head on CORRUPTED data.

        The anti-corruption analogue of :meth:`benchmark_generation`.  For each
        corrupted item the best anti-corruption agent compresses + decompresses
        it; an item is a "win" when the agent survives the noise (verified
        round-trip, smaller than original) AND its compressed size beats every
        baseline compressor's size on that same corrupted stream.  Off-the-shelf
        compressors are brittle on corrupted input, so a high win rate proves
        the anti-corruption lineage is genuinely more resilient.

        Parameters
        ----------
        generation : int
            Current generation number.
        best_anti_agent : EvolvingAgent
            Top anti-corruption agent (ranked by robustness fitness).
        corrupted_items : list[str]
            The corrupted benchmark texts the anti-corruption agents were
            evaluated on this generation.
        gui_msg_fn : callable, optional
            ``fn(message, level)`` for status output.

        Returns
        -------
        RobustnessBenchmarkReport
            ``win_rate`` (0.0-1.0) gates the corruption-track advancement.
        """
        self._ensure_rle_fns()
        _init_baselines()

        agent_ai = (best_anti_agent.get_puffin_ai()
                    if hasattr(best_anti_agent, 'get_puffin_ai') else best_anti_agent)
        agent_id = getattr(best_anti_agent, 'agent_id', 'unknown')
        rfit = (getattr(best_anti_agent, 'robustness_fitness', 0.0) or 0.0)

        report = RobustnessBenchmarkReport(
            generation=generation,
            timestamp=datetime.now().isoformat(),
            num_items=len(corrupted_items),
            agent_id=agent_id,
            robustness_fitness=rfit,
        )

        if not corrupted_items or not (self._rle_compress and self._rle_decompress):
            report.summary = "No corrupted items or RLE fns unavailable — skipped."
            return report

        wins = 0
        survived = 0
        all_items_beat = True
        for item_text in corrupted_items:
            if not item_text:
                continue
            original_bytes = item_text.encode('utf-8') if isinstance(item_text, str) else item_text
            original_size = len(original_bytes)

            # --- AI on the corrupted stream ---
            ai_verified = False
            ai_size = original_size
            try:
                comp_text, decomp_text, _action = _compress_with_agent(
                    agent_ai, item_text, self._rle_compress, self._rle_decompress
                )
                ai_size = _measure_compressed_size(comp_text) if isinstance(comp_text, str) else len(comp_text)
                ai_verified = (decomp_text == item_text) and ai_size < original_size
            except Exception:
                ai_verified = False

            if ai_verified:
                survived += 1

            # --- Baselines on the same corrupted stream ---
            baseline_sizes = []
            for method_name in _BASELINE_METHODS:
                bl = _compress_baseline(method_name, original_bytes)
                if not bl.error and bl.verified:
                    baseline_sizes.append(bl.compressed_size)

            beats_all = ai_verified and bool(baseline_sizes) and all(
                ai_size < bs for bs in baseline_sizes
            )
            if beats_all:
                wins += 1
            else:
                all_items_beat = False

        n = max(1, report.num_items)
        report.wins = wins
        report.survived = survived
        report.win_rate = wins / n
        report.survival_rate = survived / n
        report.gold_standard = all_items_beat and wins > 0
        report.summary = (
            f"Gen {generation} robustness H2H | Agent {agent_id} (rfit {rfit:.4f}) | "
            f"items {report.num_items} | survived {survived} ({report.survival_rate:.0%}) | "
            f"wins {wins} ({report.win_rate:.0%}) | "
            f"{'ROBUSTNESS GOLD STANDARD' if report.gold_standard else 'not gold'}"
        )

        if report.gold_standard:
            self.logger.info(f"ROBUSTNESS GOLD STANDARD achieved at generation {generation}!")
        self.logger.info(report.summary)
        if gui_msg_fn:
            if report.gold_standard:
                gui_msg_fn(
                    f"Gen {generation}: ROBUSTNESS GOLD STANDARD — best anti-corruption agent "
                    f"beat ALL baselines on ALL corrupted items!", "info")
            else:
                gui_msg_fn(
                    f"Gen {generation}: Robustness H2H {wins}/{report.num_items} wins "
                    f"({report.survival_rate:.0%} survived) vs baselines on corrupted data.", "info")

        return report

    # ------------------------------------------------------------------
    # Failure artefact persistence
    # ------------------------------------------------------------------

    def _save_failure_artifacts(
        self,
        generation: int,
        report: GenerationBenchmarkReport,
        test_items: List[str],
        agent_ai,
    ):
        """Write compressed + decompressed files and a summary for each item
        where the AI lost, so the developer can inspect what went wrong.

        Layout::

            gold_standard_results/
              gen_<N>/
                eval_<K>/
                  summary.txt          – human-readable report
                  summary.json         – machine-readable report
                  item_000/
                    original.txt
                    ai_compressed.txt
                    ai_decompressed.txt
                    ai_action.txt
                    <method>_compressed.txt   (hex dump for each baseline)
        """
        gen_dir = os.path.join(self.artifacts_dir, f"gen_{generation}")
        os.makedirs(gen_dir, exist_ok=True)

        eval_numbers = []
        try:
            for entry in os.listdir(gen_dir):
                # Match both "eval_NNN" directories and "eval_NNN.zip" archives
                name = entry
                if name.endswith(".zip"):
                    name = name[:-4]
                if not name.startswith("eval_"):
                    continue
                suffix = name[5:]
                if suffix.isdigit():
                    eval_numbers.append(int(suffix))
        except Exception:
            eval_numbers = []

        next_eval_index = (max(eval_numbers) + 1) if eval_numbers else 1
        eval_dir = os.path.join(gen_dir, f"eval_{next_eval_index:03d}")
        os.makedirs(eval_dir, exist_ok=True)

        # ---- summary.txt ----
        try:
            with open(os.path.join(eval_dir, "summary.txt"), "w", encoding="utf-8") as f:
                f.write(report.summary)
        except Exception as exc:
            self.logger.warning(f"Could not write summary.txt: {exc}")

        # ---- summary.json ----
        try:
            json_data = {
                "generation": report.generation,
                "evaluation_index": next_eval_index,
                "timestamp": report.timestamp,
                "agent_id": report.agent_id,
                "agent_fitness": report.agent_fitness,
                "num_items": report.num_items,
                "gold_standard": report.gold_standard,
                "items": [],
            }
            for h2h in report.items:
                item_data = {
                    "item_index": h2h.item_index,
                    "item_preview": h2h.item_preview,
                    "original_size": h2h.original_size,
                    "ai_beats_all": h2h.ai_beats_all,
                }
                if h2h.ai_result:
                    item_data["ai"] = {
                        "method": h2h.ai_result.method,
                        "compressed_size": h2h.ai_result.compressed_size,
                        "ratio_pct": round(h2h.ai_result.ratio_pct, 2),
                        "verified": h2h.ai_result.verified,
                        "error": h2h.ai_result.error,
                    }
                item_data["baselines"] = [
                    {
                        "method": bl.method,
                        "compressed_size": bl.compressed_size,
                        "ratio_pct": round(bl.ratio_pct, 2),
                        "verified": bl.verified,
                        "error": bl.error,
                    }
                    for bl in h2h.baseline_results
                ]
                json_data["items"].append(item_data)

            with open(os.path.join(eval_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            self.logger.warning(f"Could not write summary.json: {exc}")

        # ---- Per-item artefacts (only items the AI lost) ----
        for h2h in report.items:
            if h2h.ai_beats_all:
                continue  # No need to save winning items

            item_dir = os.path.join(eval_dir, f"item_{h2h.item_index:03d}")
            os.makedirs(item_dir, exist_ok=True)

            # Original text
            item_text = test_items[h2h.item_index] if h2h.item_index < len(test_items) else ""
            try:
                with open(os.path.join(item_dir, "original.txt"), "w", encoding="utf-8") as f:
                    f.write(item_text)
            except Exception:
                pass

            # AI compressed + decompressed
            if self._rle_compress and self._rle_decompress:
                try:
                    comp_text, decomp_text, action = _compress_with_agent(
                        agent_ai, item_text, self._rle_compress, self._rle_decompress
                    )
                    # Save AI compressed output as viewable text
                    with open(os.path.join(item_dir, "ai_compressed.txt"), "w", encoding="utf-8") as f:
                        if isinstance(comp_text, str):
                            f.write(comp_text)
                        else:
                            # Binary output: write hex dump for readability
                            f.write(comp_text.hex())
                    with open(os.path.join(item_dir, "ai_decompressed.txt"), "w", encoding="utf-8") as f:
                        f.write(decomp_text if isinstance(decomp_text, str) else str(decomp_text))
                    with open(os.path.join(item_dir, "ai_action.txt"), "w") as f:
                        f.write(action)
                except Exception as exc:
                    self.logger.debug(f"Could not save AI artefact for item {h2h.item_index}: {exc}")

            # Baseline compressed files
            original_bytes = item_text.encode('utf-8') if isinstance(item_text, str) else item_text
            _init_baselines()
            for method_name, (compress_fn, _decompress_fn) in _BASELINE_METHODS.items():
                try:
                    compressed = compress_fn(original_bytes)
                    # Save baseline compressed output as viewable hex dump
                    with open(os.path.join(item_dir, f"{method_name}_compressed.txt"), "w", encoding="utf-8") as f:
                        f.write(f"# {method_name} compressed ({len(compressed)} bytes)\n")
                        f.write(compressed.hex())
                except Exception:
                    pass

        # Compress eval dirs for generations older than the keep-recent threshold
        self._compress_old_generations(generation)

        self.logger.info(
            f"Gen {generation}: Failure diagnostics saved to {eval_dir}"
        )

        # Enforce 10 GB size limit
        self._enforce_size_limit()
