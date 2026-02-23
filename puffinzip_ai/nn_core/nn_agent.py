# PuffinZipAI_Project/puffinzip_ai/nn_core/nn_agent.py
"""
**Hyper-Complex Adaptive DQN Compression Agent for PuffinZipAI.**

``PuffinZipAI_NN`` extends the base ``PuffinZipAI`` class, replacing the
tabular Q-table with a Dueling DQN featuring:

    * **20-dimensional continuous state features** — log-length, unique-char ratio,
      max/avg/median run lengths, byte entropy, digit/alpha/space/punct fractions,
      bigram entropy, repeated-block ratio, byte-range spread, compressibility
      estimates, and more.
    * **Dueling DQN + NoisyNet + Multi-Head Attention + Adaptive Feature Gate**
    * **Prioritized Experience Replay (PER)** with importance-sampling correction.
    * **Cosine-annealing learning rate** with warm restarts for training stability.
    * **Adaptive exploration** using NoisyNet (state-dependent) + ε-greedy fallback.
    * **Multi-step returns** (n-step = 3) for faster reward propagation.
    * **Gradient accumulation** for effective larger batch sizes on small buffers.
    * **Soft target updates** (Polyak averaging) for smoother training.
    * **Training metrics tracking** — loss, Q-value statistics, gradient norms,
      noise magnitudes, learning rate, all exported for GUI/logging.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import random
import time
import traceback
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- Base class import ---
from ..ai_core import PuffinZipAI as _PuffinZipAI_base

# --- Config imports ---
try:
    from ..config import (
        NN_STATE_FEATURE_DIM,
        NN_HIDDEN_SIZES,
        NN_REPLAY_BUFFER_CAPACITY,
        NN_REPLAY_MIN_SIZE,
        NN_TRAIN_BATCH_SIZE,
        NN_TARGET_NETWORK_UPDATE_FREQ,
        NN_LEARNING_RATE,
        NN_GRAD_CLIP_NORM,
        NN_WEIGHT_DECAY,
        NN_SOFTMAX_ACTION_TEMPERATURE,
        NN_PER_ALPHA,
        NN_PER_BETA_START,
        NN_PER_BETA_FRAMES,
        NN_COSINE_LR_T_MAX,
        NN_COSINE_LR_ETA_MIN,
        NN_SOFT_TARGET_TAU,
        NN_NSTEP_RETURNS,
        NN_DROPOUT,
        NN_ATTENTION_HEADS,
        NN_NOISY_SIGMA,
        DEFAULT_LEARNING_RATE,
        DEFAULT_DISCOUNT_FACTOR,
        DEFAULT_EXPLORATION_RATE,
        DEFAULT_EXPLORATION_DECAY_RATE,
        DEFAULT_MIN_EXPLORATION_RATE,
        ACCELERATION_TARGET_DEVICE as CONFIG_ACCELERATION_TARGET_DEVICE_DEFAULT,
    )
except ImportError:
    # Fallback defaults if config not available
    NN_STATE_FEATURE_DIM = 20
    NN_HIDDEN_SIZES = [256, 256]
    NN_REPLAY_BUFFER_CAPACITY = 50_000
    NN_REPLAY_MIN_SIZE = 128
    NN_TRAIN_BATCH_SIZE = 128
    NN_TARGET_NETWORK_UPDATE_FREQ = 100
    NN_LEARNING_RATE = 3e-4
    NN_GRAD_CLIP_NORM = 1.0
    NN_WEIGHT_DECAY = 1e-5
    NN_SOFTMAX_ACTION_TEMPERATURE = 0.5
    NN_PER_ALPHA = 0.6
    NN_PER_BETA_START = 0.4
    NN_PER_BETA_FRAMES = 100_000
    NN_COSINE_LR_T_MAX = 5000
    NN_COSINE_LR_ETA_MIN = 1e-6
    NN_SOFT_TARGET_TAU = 0.005
    NN_NSTEP_RETURNS = 3
    NN_DROPOUT = 0.1
    NN_ATTENTION_HEADS = 4
    NN_NOISY_SIGMA = 0.5
    DEFAULT_LEARNING_RATE = 0.1
    DEFAULT_DISCOUNT_FACTOR = 0.9
    DEFAULT_EXPLORATION_RATE = 1.0
    DEFAULT_EXPLORATION_DECAY_RATE = 0.9995
    DEFAULT_MIN_EXPLORATION_RATE = 0.01
    CONFIG_ACCELERATION_TARGET_DEVICE_DEFAULT = "CPU"

# --- NN-specific imports ---
import torch  # type: ignore[import-unresolved]
import torch.nn.functional as F  # type: ignore[import-unresolved]
import torch.optim as optim  # type: ignore[import-unresolved]

TORCH_AVAILABLE = True

from .dqn_model import DQNNetwork
from .replay_buffer import ReplayBuffer

logger = logging.getLogger("puffinzip_ai.nn_core.nn_agent")


# ---------------------------------------------------------------------------
# Advanced Feature Extraction (20 dimensions)
# ---------------------------------------------------------------------------

def _byte_entropy(data: str) -> float:
    """Shannon entropy (bits) of the byte distribution. Normalised to [0, 1]."""
    if not data:
        return 0.0
    n = len(data)
    freq: Dict[str, int] = {}
    for ch in data:
        freq[ch] = freq.get(ch, 0) + 1
    entropy = 0.0
    for count in freq.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)
    return min(entropy / 8.0, 1.0)


def _bigram_entropy(data: str) -> float:
    """Shannon entropy of character bigrams. Normalised to [0, 1]."""
    if len(data) < 2:
        return 0.0
    n = len(data) - 1
    freq: Dict[str, int] = {}
    for i in range(n):
        bg = data[i:i+2]
        freq[bg] = freq.get(bg, 0) + 1
    entropy = 0.0
    for count in freq.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)
    # Max possible bigram entropy = log2(65536) = 16
    return min(entropy / 16.0, 1.0)


def _run_lengths(data: str) -> List[int]:
    """Return list of consecutive-character run lengths."""
    if not data:
        return []
    runs = []
    current_run = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)
    return runs


def _max_run_length(data: str) -> int:
    """Length of the longest consecutive-character run in *data*."""
    runs = _run_lengths(data)
    return max(runs) if runs else 0


def _avg_run_length(data: str) -> float:
    """Average consecutive-character run length in *data*."""
    runs = _run_lengths(data)
    return sum(runs) / max(len(runs), 1)


def _median_run_length(data: str) -> float:
    """Median consecutive-character run length."""
    runs = _run_lengths(data)
    if not runs:
        return 0.0
    sorted_runs = sorted(runs)
    mid = len(sorted_runs) // 2
    if len(sorted_runs) % 2 == 0:
        return (sorted_runs[mid - 1] + sorted_runs[mid]) / 2.0
    return float(sorted_runs[mid])


def _run_length_variance(data: str) -> float:
    """Variance of run lengths. High variance = mixed compressibility."""
    runs = _run_lengths(data)
    if len(runs) < 2:
        return 0.0
    mean = sum(runs) / len(runs)
    var = sum((r - mean) ** 2 for r in runs) / len(runs)
    return var


def _repeated_block_ratio(data: str, block_size: int = 4) -> float:
    """Fraction of non-overlapping blocks that appear more than once."""
    if len(data) < block_size * 2:
        return 0.0
    blocks: Dict[str, int] = {}
    n_blocks = len(data) // block_size
    for i in range(n_blocks):
        block = data[i * block_size:(i + 1) * block_size]
        blocks[block] = blocks.get(block, 0) + 1
    repeated = sum(1 for c in blocks.values() if c > 1)
    return repeated / max(len(blocks), 1)


def _byte_range_spread(data: str) -> float:
    """Range of byte values normalised by 255. Narrow = more compressible."""
    if not data:
        return 0.0
    ords = [ord(c) for c in data]
    return (max(ords) - min(ords)) / 255.0


def _longest_repeated_substring_ratio(data: str) -> float:
    """Approximate longest repeated substring ratio using suffix heuristic."""
    if len(data) < 4:
        return 0.0
    n = len(data)
    # Use a fast heuristic: check fixed block sizes
    best = 0
    for blen in [2, 3, 4, 6, 8]:
        if blen > n // 2:
            break
        seen = set()
        for i in range(n - blen + 1):
            sub = data[i:i + blen]
            if sub in seen:
                best = max(best, blen)
                break
            seen.add(sub)
    return best / max(n, 1)


def _simple_rle_compressibility(data: str) -> float:
    """Estimate compressibility: fraction of chars in runs of 3+."""
    if not data:
        return 0.0
    runs = _run_lengths(data)
    compressible_chars = sum(r for r in runs if r >= 3)
    return compressible_chars / len(data)


def _is_mostly_ascii(data: str) -> float:
    """Fraction of characters that are printable ASCII (32-126)."""
    if not data:
        return 0.0
    ascii_count = sum(1 for c in data if 32 <= ord(c) <= 126)
    return ascii_count / len(data)


def extract_features(data: str) -> np.ndarray:
    """Extract a 20-dimensional continuous feature vector from input text.

    Returns
    -------
    np.ndarray  shape ``(NN_STATE_FEATURE_DIM,)``  dtype float32

    Features (all normalised roughly to [0, 1]):
         0. log-length            — log2(len+1) / 20 (caps at ~1 MB)
         1. unique-char ratio     — unique_chars / 256
         2. max-run ratio         — max_run / len
         3. byte entropy (norm)   — Shannon entropy / 8
         4. avg-run ratio         — avg_run / len
         5. digit fraction        — fraction of chars that are digits
         6. alpha fraction        — fraction of chars that are alphabetic
         7. space fraction        — fraction of whitespace chars
         8. punctuation fraction  — fraction of punctuation chars
         9. bigram entropy (norm) — bigram Shannon entropy / 16
        10. median-run ratio      — median_run / len
        11. run-length variance   — var(runs) / len² (normalised)
        12. repeated block ratio  — fraction of 4-byte blocks that repeat
        13. byte range spread     — (max_byte - min_byte) / 255
        14. longest repeat ratio  — longest repeated substring / len
        15. RLE compressibility   — fraction of chars in runs of 3+
        16. uppercase ratio       — fraction of uppercase letters
        17. is-mostly-ASCII       — fraction of printable ASCII chars
        18. char frequency skew   — max_char_freq / avg_char_freq (normalised)
        19. length category        — bucketed: tiny(0.1)/small(0.3)/medium(0.5)/large(0.7)/huge(0.9)
    """
    n = len(data) if data else 0
    if n == 0:
        return np.zeros(NN_STATE_FEATURE_DIM, dtype=np.float32)

    unique_chars = len(set(data))
    runs = _run_lengths(data)
    max_run = max(runs) if runs else 0
    avg_run = sum(runs) / max(len(runs), 1)
    median_run = _median_run_length(data)
    run_var = _run_length_variance(data)
    entropy = _byte_entropy(data)
    bigram_ent = _bigram_entropy(data)

    digit_count = sum(1 for c in data if c.isdigit())
    alpha_count = sum(1 for c in data if c.isalpha())
    space_count = sum(1 for c in data if c.isspace())
    upper_count = sum(1 for c in data if c.isupper())
    # Punctuation: not alphanumeric, not whitespace, printable
    punct_count = sum(1 for c in data if not c.isalnum() and not c.isspace() and 32 <= ord(c) <= 126)

    # Char frequency skew: how dominant is the most common char?
    freq: Dict[str, int] = {}
    for ch in data:
        freq[ch] = freq.get(ch, 0) + 1
    max_freq = max(freq.values()) / n
    avg_freq = (1.0 / max(unique_chars, 1))
    freq_skew = min(max_freq / max(avg_freq, 1e-6), 10.0) / 10.0  # normalise to [0,1]

    # Length category bucket
    if n < 50:
        len_cat = 0.1
    elif n < 200:
        len_cat = 0.3
    elif n < 1000:
        len_cat = 0.5
    elif n < 10000:
        len_cat = 0.7
    else:
        len_cat = 0.9

    features = np.array([
        min(math.log2(n + 1) / 20.0, 1.0),           #  0: log-length
        unique_chars / 256.0,                           #  1: unique-char ratio
        max_run / n,                                    #  2: max-run ratio
        entropy,                                        #  3: byte entropy
        min(avg_run / n, 1.0),                          #  4: avg-run ratio
        digit_count / n,                                #  5: digit fraction
        alpha_count / n,                                #  6: alpha fraction
        space_count / n,                                #  7: space fraction
        punct_count / n,                                #  8: punctuation fraction
        bigram_ent,                                     #  9: bigram entropy
        min(median_run / n, 1.0),                       # 10: median-run ratio
        min(run_var / max(n * n, 1), 1.0),              # 11: run-length variance (norm)
        _repeated_block_ratio(data),                    # 12: repeated block ratio
        _byte_range_spread(data),                       # 13: byte range spread
        _longest_repeated_substring_ratio(data),        # 14: longest repeat ratio
        _simple_rle_compressibility(data),              # 15: RLE compressibility
        upper_count / max(alpha_count, 1),              # 16: uppercase ratio
        _is_mostly_ascii(data),                         # 17: printable ASCII fraction
        freq_skew,                                      # 18: char frequency skew
        len_cat,                                        # 19: length category
    ], dtype=np.float32)

    return features


# ---------------------------------------------------------------------------
# Torch device resolution
# ---------------------------------------------------------------------------

def _resolve_torch_device(target_device_str: str) -> torch.device:
    """Map a PuffinZipAI target_device string to a ``torch.device``."""
    if not TORCH_AVAILABLE:
        return None  # type: ignore[return-value]
    td = target_device_str.upper().strip() if target_device_str else "CPU"
    if td.startswith("GPU") and torch.cuda.is_available():
        if td.startswith("GPU_ID:"):
            try:
                gpu_id = int(td.split(":")[1])
                if 0 <= gpu_id < torch.cuda.device_count():
                    return torch.device(f"cuda:{gpu_id}")
            except (ValueError, IndexError):
                pass
        # GPU_AUTO or fallback
        return torch.device("cuda:0")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# N-Step Return Buffer
# ---------------------------------------------------------------------------

class _NStepBuffer:
    """Accumulates n-step returns before pushing to the main replay buffer.

    Instead of pushing (s, a, r, s') with 1-step reward, accumulates n
    transitions and pushes (s_0, a_0, R_n, s_n) where R_n = Σ γ^i r_i.
    This propagates rewards faster through the network.
    """

    def __init__(self, n_steps: int = 3, gamma: float = 0.99) -> None:
        self.n_steps = n_steps
        self.gamma = gamma
        self._buffer: deque = deque(maxlen=n_steps)

    def push(self, state, action, reward, next_state, done) -> Optional[tuple]:
        """Push a transition. Returns an n-step transition if buffer is full."""
        self._buffer.append((state, action, reward, next_state, done))

        if len(self._buffer) < self.n_steps:
            return None

        # Compute n-step return
        R = 0.0
        for i in reversed(range(self.n_steps)):
            s, a, r, ns, d = self._buffer[i]
            R = r + self.gamma * R * (1.0 - float(d))

        first = self._buffer[0]
        last = self._buffer[-1]
        return (first[0], first[1], R, last[3], last[4])

    def flush(self) -> List[tuple]:
        """Flush remaining transitions (at episode boundaries)."""
        result = []
        while self._buffer:
            R = 0.0
            for i in reversed(range(len(self._buffer))):
                s, a, r, ns, d = self._buffer[i]
                R = r + self.gamma * R * (1.0 - float(d))
            first = self._buffer[0]
            last = self._buffer[-1]
            result.append((first[0], first[1], R, last[3], last[4]))
            self._buffer.popleft()
        return result

    def clear(self) -> None:
        self._buffer.clear()


# ---------------------------------------------------------------------------
# PuffinZipAI_NN class
# ---------------------------------------------------------------------------

class PuffinZipAI_NN(_PuffinZipAI_base):
    """Hyper-Complex Adaptive DQN Compression Agent.

    Drop-in replacement for ``PuffinZipAI`` / ``PuffinZipAI_GPU`` inside the
    evolutionary optimiser.  Features:

    * Dueling DQN with NoisyNet + Multi-Head Attention + Adaptive Feature Gate
    * Prioritized Experience Replay (PER) with importance-sampling
    * 20-dimensional continuous state features
    * Cosine-annealing LR scheduler with warm restarts
    * N-step returns for faster reward propagation
    * Soft target network updates (Polyak averaging)
    * Comprehensive training metrics for monitoring

    Parameters
    ----------
    nn_hidden_sizes : list[int], optional
        Override ``NN_HIDDEN_SIZES`` from config.
    nn_lr : float, optional
        Override ``NN_LEARNING_RATE`` from config.
    _defer_gpu_transfer : bool
        When True, defer moving the network to GPU — call
        :meth:`finalize_gpu_init` later.
    """

    # Class-level marker so external code can detect NN agents quickly
    MODEL_TYPE = "dqn"

    def __init__(
        self,
        len_thresholds=None,
        learning_rate=None,
        discount_factor=None,
        exploration_rate=None,
        exploration_decay_rate=None,
        min_exploration_rate=None,
        rle_min_encodable_run: Optional[int] = None,
        rle_min_encodable_run_length: Optional[int] = None,
        target_device: Optional[str] = None,
        nn_hidden_sizes=None,
        nn_lr: Optional[float] = None,
        _defer_gpu_transfer: bool = False,
        **kwargs,
    ) -> None:
        # Merge RLE param aliases
        if rle_min_encodable_run is None and rle_min_encodable_run_length is not None:
            rle_min_encodable_run = rle_min_encodable_run_length

        # Initialise base class (creates Q-table, logger, thresholds, etc.)
        _base_kwargs: Dict[str, Any] = dict(
            len_thresholds=len_thresholds,
            learning_rate=learning_rate or DEFAULT_LEARNING_RATE,
            discount_factor=discount_factor or DEFAULT_DISCOUNT_FACTOR,
            exploration_rate=exploration_rate or DEFAULT_EXPLORATION_RATE,
            exploration_decay_rate=exploration_decay_rate or DEFAULT_EXPLORATION_DECAY_RATE,
            min_exploration_rate=min_exploration_rate or DEFAULT_MIN_EXPLORATION_RATE,
            target_device=target_device or CONFIG_ACCELERATION_TARGET_DEVICE_DEFAULT,
        )
        if rle_min_encodable_run is not None:
            _base_kwargs["rle_min_encodable_run"] = rle_min_encodable_run
        super().__init__(**_base_kwargs)

        # NN-specific config
        self._nn_hidden_sizes = list(nn_hidden_sizes or NN_HIDDEN_SIZES)
        self._nn_lr = nn_lr if nn_lr is not None else NN_LEARNING_RATE
        self._nn_grad_clip = NN_GRAD_CLIP_NORM
        self._nn_weight_decay = NN_WEIGHT_DECAY
        self._nn_target_update_freq = NN_TARGET_NETWORK_UPDATE_FREQ
        self._nn_train_batch_size = NN_TRAIN_BATCH_SIZE
        self._nn_replay_min = NN_REPLAY_MIN_SIZE
        self._nn_softmax_temp = NN_SOFTMAX_ACTION_TEMPERATURE
        self._nn_step_count: int = 0  # global optimisation step counter
        self._nn_soft_tau = NN_SOFT_TARGET_TAU
        self._nn_nstep = NN_NSTEP_RETURNS
        self._nn_dropout = NN_DROPOUT
        self._nn_attention_heads = NN_ATTENTION_HEADS
        self._nn_noisy_sigma = NN_NOISY_SIGMA

        # Training metrics
        self._training_metrics: Dict[str, deque] = {
            "loss": deque(maxlen=500),
            "avg_q_value": deque(maxlen=500),
            "avg_td_error": deque(maxlen=500),
            "grad_norm": deque(maxlen=500),
            "noise_magnitude": deque(maxlen=500),
            "learning_rate": deque(maxlen=500),
            "replay_beta": deque(maxlen=500),
        }

        # Resolve torch device
        self._deferred_gpu = _defer_gpu_transfer
        self._torch_device: torch.device = torch.device("cpu")  # tentative
        if not _defer_gpu_transfer:
            self._torch_device = _resolve_torch_device(self.target_device)

        # Build networks
        self._policy_net = DQNNetwork(
            state_dim=NN_STATE_FEATURE_DIM,
            action_dim=self.action_space_size,
            hidden_sizes=self._nn_hidden_sizes,
            dropout=self._nn_dropout,
            attention_heads=self._nn_attention_heads,
            noisy_sigma=self._nn_noisy_sigma,
        )
        self._target_net = self._policy_net.clone()
        self._target_net.eval()  # target net is never trained directly

        # Optimiser — AdamW for better weight decay handling
        self._optimizer = optim.AdamW(
            self._policy_net.parameters(),
            lr=self._nn_lr,
            weight_decay=self._nn_weight_decay,
            amsgrad=True,
        )

        # Learning rate scheduler — Cosine annealing with warm restarts
        self._lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self._optimizer,
            T_0=NN_COSINE_LR_T_MAX,
            T_mult=2,
            eta_min=NN_COSINE_LR_ETA_MIN,
        )

        # Prioritized Experience Replay buffer
        self._replay_buffer = ReplayBuffer(
            capacity=NN_REPLAY_BUFFER_CAPACITY,
            alpha=NN_PER_ALPHA,
            beta_start=NN_PER_BETA_START,
            beta_frames=NN_PER_BETA_FRAMES,
        )

        # N-step return buffer
        self._nstep_buffer = _NStepBuffer(
            n_steps=self._nn_nstep,
            gamma=self.discount_factor,
        )

        # Move to device
        if not _defer_gpu_transfer:
            self._move_nets_to_device()

        self.logger.info(
            f"PuffinZipAI_NN created (Dueling+Noisy+Attention).  "
            f"Network: {self._policy_net}  Device: {self._torch_device}  "
            f"Features: {NN_STATE_FEATURE_DIM}  Deferred: {_defer_gpu_transfer}"
        )

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------
    def _move_nets_to_device(self) -> None:
        """Transfer both networks to ``self._torch_device``."""
        self._policy_net.to(self._torch_device)
        self._target_net.to(self._torch_device)
        # Recreate optimiser so its state is on the right device
        self._optimizer = optim.AdamW(
            self._policy_net.parameters(),
            lr=self._nn_lr,
            weight_decay=self._nn_weight_decay,
            amsgrad=True,
        )
        self._lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self._optimizer,
            T_0=NN_COSINE_LR_T_MAX,
            T_mult=2,
            eta_min=NN_COSINE_LR_ETA_MIN,
        )

    def finalize_gpu_init(self) -> None:
        """Finalise GPU transfer for agents created with ``_defer_gpu_transfer=True``."""
        if not self._deferred_gpu:
            return
        self._deferred_gpu = False
        self._torch_device = _resolve_torch_device(self.target_device)
        self._move_nets_to_device()
        self.logger.debug(f"PuffinZipAI_NN GPU finalised → {self._torch_device}")

    # ------------------------------------------------------------------
    # State representation  (OVERRIDE)
    # ------------------------------------------------------------------
    def _get_state_representation(self, item_text: str) -> int:
        """Return integer state index **and** cache the continuous feature vector.

        The integer index is still computed (via the base class) so that the
        shadow Q-table stays usable.  The cached ``_last_features`` is used
        by :meth:`_choose_action` and :meth:`_update_q_table` for NN ops.
        """
        # Compute 20-dimensional continuous features
        self._last_features: np.ndarray = extract_features(item_text)
        # Base class discrete index (for shadow Q-table compatibility)
        return super()._get_state_representation(item_text)

    # ------------------------------------------------------------------
    # Action selection  (OVERRIDE) — NoisyNet + ε-greedy hybrid
    # ------------------------------------------------------------------
    def _choose_action(self, state_idx: int, use_exploration: bool = True) -> int:
        """Choose an action using the DQN policy network with NoisyNet exploration.

        During training (use_exploration=True):
            - Primary: NoisyNet provides state-dependent exploration via learned noise
            - Fallback: ε-greedy with decaying ε ensures minimum exploration
        During evaluation (use_exploration=False):
            - Greedy action selection (NoisyNet noise disabled by eval mode)
        """
        features = getattr(self, "_last_features", None)
        if features is None:
            return super()._choose_action(state_idx, use_exploration=use_exploration)

        action_space = self.action_space_size
        effective_actions = 3  # base: RLE, NoCompression, AdvancedRLE
        if self._novel_compress_fn:
            effective_actions = 4  # include NovelMethod
        # Include ReferenceMethod only when novel method is also available
        # and scaffolding allows it
        if (self._scaffolding_enabled and self._reference_compress_fn
                and effective_actions >= 4):
            ref_allowed = True
            try:
                from puffinzip_ai.compression_scaffolding import get_scaffolding_manager
                agent_id = str(self._scaffold_agent_id or id(self))
                ref_allowed = get_scaffolding_manager().is_reference_allowed(
                    agent_id, getattr(self, '_current_generation', 0)
                )
            except ImportError:
                pass
            if ref_allowed:
                effective_actions = 5
        effective_actions = min(effective_actions, action_space)

        # ε-greedy fallback exploration (on top of NoisyNet)
        if use_exploration and random.random() < self.exploration_rate:
            action_idx = random.randint(0, effective_actions - 1)
        else:
            # Reset noise for this forward pass (NoisyNet exploration)
            if use_exploration:
                self._policy_net.reset_noise()
                self._policy_net.train()
            else:
                self._policy_net.eval()

            with torch.no_grad():
                state_tensor = torch.from_numpy(features).unsqueeze(0).to(self._torch_device)
                q_values = self._policy_net(state_tensor).squeeze(0)
                # Mask unavailable actions
                if effective_actions < action_space:
                    q_values[effective_actions:] = float("-inf")
                action_idx = int(q_values.argmax().item())

        # Track action statistics
        if use_exploration:
            action_name = self.action_names.get(action_idx)
            if action_name == "RLE":
                self.training_stats['rle_chosen_count'] += 1
            elif action_name == "NoCompression":
                self.training_stats['nocomp_chosen_count'] += 1
            elif action_name == "AdvancedRLE":
                self.training_stats['advanced_rle_chosen_count'] += 1
            elif action_name == "NovelMethod":
                self.training_stats['novel_method_chosen_count'] += 1
            elif action_name == "ReferenceMethod":
                self.training_stats['reference_method_chosen_count'] = self.training_stats.get('reference_method_chosen_count', 0) + 1

        return action_idx

    # ------------------------------------------------------------------
    # Batch inference  (GPU-optimised for pipeline)
    # ------------------------------------------------------------------
    def batch_choose_actions(
        self, texts: list, use_exploration: bool = False
    ) -> list:
        """Select actions for *all* texts in ONE batched GPU forward pass.

        Stacks every item's feature vector into a single ``(N, feature_dim)``
        tensor and calls the policy network once.  On a CUDA device this is
        **10-100x** faster than per-item inference.
        """
        if not texts:
            return []

        # --- CPU: feature extraction ---
        features_list = [extract_features(t) for t in texts]
        features_batch = np.stack(features_list)

        effective_actions = 3  # base: RLE, NoCompression, AdvancedRLE
        if self._novel_compress_fn:
            effective_actions = 4
        if (self._scaffolding_enabled and self._reference_compress_fn
                and effective_actions >= 4):
            ref_allowed = True
            try:
                from puffinzip_ai.compression_scaffolding import get_scaffolding_manager
                agent_id = str(self._scaffold_agent_id or id(self))
                ref_allowed = get_scaffolding_manager().is_reference_allowed(
                    agent_id, getattr(self, '_current_generation', 0)
                )
            except ImportError:
                pass
            if ref_allowed:
                effective_actions = 5
        effective_actions = min(effective_actions, self.action_space_size)

        # Reset noise if exploring
        if use_exploration:
            self._policy_net.reset_noise()
            self._policy_net.train()
        else:
            self._policy_net.eval()

        # --- GPU: single batched forward pass ---
        with torch.no_grad():
            batch_tensor = torch.from_numpy(features_batch).to(self._torch_device)
            q_values = self._policy_net(batch_tensor)
            if effective_actions < self.action_space_size:
                q_values[:, effective_actions:] = float("-inf")
            best_actions = q_values.argmax(dim=1).cpu().numpy()

        # --- Apply ε-greedy exploration ---
        results = []
        for i in range(len(texts)):
            if use_exploration and random.random() < self.exploration_rate:
                act = random.randint(0, effective_actions - 1)
            else:
                act = int(best_actions[i])
            results.append((act, features_list[i]))

        return results

    def batch_push_experiences(self, experiences: list) -> None:
        """Push multiple ``(features, action, reward)`` tuples and run batch DQN training."""
        for features, action_idx, reward in experiences:
            next_feats = np.zeros_like(features)
            # Push through n-step buffer
            nstep_transition = self._nstep_buffer.push(
                features, action_idx, reward, next_feats, True
            )
            if nstep_transition is not None:
                s, a, r, ns, d = nstep_transition
                self._replay_buffer.push(s, a, r, ns, d)

        # Flush remaining n-step transitions
        for transition in self._nstep_buffer.flush():
            s, a, r, ns, d = transition
            self._replay_buffer.push(s, a, r, ns, d)

        # Batch DQN training — run up to 10 gradient steps
        steps = max(min(len(experiences) // max(self._nn_train_batch_size, 1), 10), 1)
        for _ in range(steps):
            if self._replay_buffer.is_ready(self._nn_replay_min):
                self._dqn_train_step()

    # ------------------------------------------------------------------
    # Q-table update  (OVERRIDE → DQN gradient step)
    # ------------------------------------------------------------------
    def _update_q_table(
        self,
        state_idx: int,
        action_idx: int,
        reward_val: float,
        next_state_idx: Optional[int] = None,
    ) -> None:
        """Store transition via n-step buffer → PER, then do one DQN update."""
        state_feats = getattr(self, "_last_features", None)
        if state_feats is None:
            state_feats = np.zeros(NN_STATE_FEATURE_DIM, dtype=np.float32)

        next_feats = np.zeros(NN_STATE_FEATURE_DIM, dtype=np.float32)
        done = True

        # Push through n-step buffer
        nstep_transition = self._nstep_buffer.push(
            state_feats, action_idx, reward_val, next_feats, done
        )
        if nstep_transition is not None:
            s, a, r, ns, d = nstep_transition
            self._replay_buffer.push(s, a, r, ns, d)

        # DQN training step
        if self._replay_buffer.is_ready(self._nn_replay_min):
            self._dqn_train_step()

        # Shadow Q-table update
        super()._update_q_table(state_idx, action_idx, reward_val, next_state_idx=next_state_idx)

    def _dqn_train_step(self) -> None:
        """One mini-batch gradient descent step on the DQN loss with PER."""
        self._policy_net.train()
        self._policy_net.reset_noise()
        self._target_net.reset_noise()

        # Sample from PER buffer
        sample_result = self._replay_buffer.sample(self._nn_train_batch_size)
        states, actions, rewards, next_states, dones, indices, is_weights = sample_result

        # Move to device
        states = states.to(self._torch_device)
        actions = actions.to(self._torch_device)
        rewards = rewards.to(self._torch_device)
        next_states = next_states.to(self._torch_device)
        dones = dones.to(self._torch_device)
        is_weights = is_weights.to(self._torch_device)

        # Current Q-values for taken actions
        q_values = self._policy_net(states).gather(1, actions)

        # Double DQN: use policy net to select actions, target net to evaluate
        with torch.no_grad():
            # Policy net selects best action
            next_actions = self._policy_net(next_states).argmax(dim=1, keepdim=True)
            # Target net evaluates Q-value of that action
            next_q_values = self._target_net(next_states).gather(1, next_actions)
            # N-step discount: γ^n
            gamma_n = self.discount_factor ** self._nn_nstep
            target = rewards + gamma_n * next_q_values * (1.0 - dones)

        # TD-errors for priority updates
        td_errors = (q_values - target).detach().cpu().numpy()

        # Weighted Huber loss (PER importance-sampling correction)
        element_wise_loss = F.smooth_l1_loss(q_values, target, reduction="none")
        loss = (element_wise_loss * is_weights).mean()

        # Backprop
        self._optimizer.zero_grad()
        loss.backward()

        # Track gradient norm before clipping
        total_norm = 0.0
        for p in self._policy_net.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        if self._nn_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self._policy_net.parameters(), self._nn_grad_clip
            )
        self._optimizer.step()
        self._lr_scheduler.step()

        # Update PER priorities
        self._replay_buffer.update_priorities(indices, np.abs(td_errors))

        # Soft target network update (Polyak averaging)
        self._nn_step_count += 1
        if self._nn_step_count % self._nn_target_update_freq == 0:
            self._target_net.soft_sync_from(self._policy_net, tau=self._nn_soft_tau)

        # Record training metrics
        current_lr = self._optimizer.param_groups[0]["lr"]
        self._training_metrics["loss"].append(loss.item())
        self._training_metrics["avg_q_value"].append(q_values.mean().item())
        self._training_metrics["avg_td_error"].append(np.abs(td_errors).mean())
        self._training_metrics["grad_norm"].append(total_norm)
        self._training_metrics["noise_magnitude"].append(self._policy_net.get_noise_magnitude())
        self._training_metrics["learning_rate"].append(current_lr)
        self._training_metrics["replay_beta"].append(self._replay_buffer.beta)

    def get_training_metrics_summary(self) -> Dict[str, float]:
        """Return latest training metrics for logging/GUI display."""
        summary = {}
        for key, values in self._training_metrics.items():
            if values:
                arr = list(values)
                summary[f"{key}_latest"] = arr[-1]
                summary[f"{key}_avg_100"] = sum(arr[-100:]) / max(len(arr[-100:]), 1)
            else:
                summary[f"{key}_latest"] = 0.0
                summary[f"{key}_avg_100"] = 0.0
        summary["training_steps"] = self._nn_step_count
        summary["replay_size"] = len(self._replay_buffer)
        summary["replay_capacity"] = self._replay_buffer.capacity
        return summary

    # ------------------------------------------------------------------
    # Reinitialise (OVERRIDE)
    # ------------------------------------------------------------------
    def _reinitialize_state_dependent_vars(self) -> None:
        """Called when thresholds change — rebuild shadow Q-table."""
        super()._reinitialize_state_dependent_vars()

    # ------------------------------------------------------------------
    # Cloning
    # ------------------------------------------------------------------
    def clone_core_model(self) -> "PuffinZipAI_NN":
        """Deep-clone this agent, including NN weights and replay buffer."""
        config = self.get_config_dict()
        config["nn_hidden_sizes"] = list(self._nn_hidden_sizes)
        config["nn_lr"] = self._nn_lr
        config["_defer_gpu_transfer"] = True
        cloned = PuffinZipAI_NN(**config)

        # Copy NN weights
        cloned._policy_net.load_state_dict(self._policy_net.state_dict())
        cloned._target_net.load_state_dict(self._target_net.state_dict())

        # Copy shadow Q-table
        if self.q_table is not None:
            if cloned.q_table is not None and cloned.q_table.shape == self.q_table.shape:
                cloned.q_table = np.copy(self.q_table)

        # Copy exploration rate and step count
        cloned.exploration_rate = self.exploration_rate
        cloned._nn_step_count = self._nn_step_count

        # Propagate novel compression method
        cloned.novel_method = self.novel_method
        cloned._novel_compress_fn = self._novel_compress_fn
        cloned._novel_decompress_fn = self._novel_decompress_fn

        # Propagate scaffolding settings
        cloned._scaffolding_enabled = self._scaffolding_enabled
        cloned._preferred_reference = self._preferred_reference
        cloned._reference_compress_fn = self._reference_compress_fn
        cloned._reference_decompress_fn = self._reference_decompress_fn

        # Note: replay buffer is NOT copied (child starts fresh replay)
        # N-step buffer is also fresh

        # Finalise GPU transfer
        cloned._deferred_gpu = False
        cloned._torch_device = self._torch_device
        cloned._move_nets_to_device()

        self.logger.debug(
            f"Cloned PuffinZipAI_NN → device={cloned._torch_device}, "
            f"params={cloned._policy_net.parameter_count():,}"
        )
        return cloned

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save_model(self, fp: Optional[str] = None) -> bool:
        """Save the NN agent state (base .npy + PyTorch .pt)."""
        success = super().save_model(fp)
        if not success:
            return False

        base_path = os.path.abspath(fp) if fp else os.path.abspath("puffin_ai_model_default.dat")
        nn_path = base_path.rsplit(".", 1)[0] + "_nn.pt"
        try:
            nn_state = {
                "policy_net": self._policy_net.state_dict(),
                "target_net": self._target_net.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "scheduler": self._lr_scheduler.state_dict(),
                "nn_hidden_sizes": self._nn_hidden_sizes,
                "nn_lr": self._nn_lr,
                "nn_step_count": self._nn_step_count,
                "state_dim": NN_STATE_FEATURE_DIM,
                "action_dim": self.action_space_size,
                "architecture": "dueling_noisy_attention_v2",
            }
            torch.save(nn_state, nn_path)
            self.logger.info(f"NN weights saved to: {nn_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save NN weights to {nn_path}: {e}", exc_info=True)
            return False

    def load_model(self, fp: Optional[str] = None) -> bool:
        """Load the NN agent state."""
        success = super().load_model(fp)

        base_path = os.path.abspath(fp) if fp else os.path.abspath("puffin_ai_model_default.dat")
        nn_path = base_path.rsplit(".", 1)[0] + "_nn.pt"

        if os.path.exists(nn_path):
            try:
                nn_state = torch.load(nn_path, map_location=self._torch_device, weights_only=False)
                loaded_hidden = nn_state.get("nn_hidden_sizes", self._nn_hidden_sizes)
                loaded_state_dim = nn_state.get("state_dim", NN_STATE_FEATURE_DIM)
                loaded_action_dim = nn_state.get("action_dim", self.action_space_size)

                if (
                    loaded_hidden != self._nn_hidden_sizes
                    or loaded_state_dim != NN_STATE_FEATURE_DIM
                    or loaded_action_dim != self.action_space_size
                ):
                    self.logger.warning(
                        f"NN architecture mismatch on load.  "
                        f"Saved: state={loaded_state_dim}, hidden={loaded_hidden}, action={loaded_action_dim}.  "
                        f"Current: state={NN_STATE_FEATURE_DIM}, hidden={self._nn_hidden_sizes}, action={self.action_space_size}.  "
                        f"Rebuilding from scratch."
                    )
                    self._nn_hidden_sizes = loaded_hidden
                    self._policy_net = DQNNetwork(
                        loaded_state_dim, loaded_action_dim, loaded_hidden,
                        dropout=self._nn_dropout,
                        attention_heads=self._nn_attention_heads,
                        noisy_sigma=self._nn_noisy_sigma,
                    )
                    self._target_net = self._policy_net.clone()

                self._policy_net.load_state_dict(nn_state["policy_net"])
                self._target_net.load_state_dict(nn_state["target_net"])
                self._optimizer = optim.AdamW(
                    self._policy_net.parameters(),
                    lr=nn_state.get("nn_lr", self._nn_lr),
                    weight_decay=self._nn_weight_decay,
                    amsgrad=True,
                )
                if "optimizer" in nn_state:
                    try:
                        self._optimizer.load_state_dict(nn_state["optimizer"])
                    except Exception:
                        pass
                if "scheduler" in nn_state:
                    try:
                        self._lr_scheduler.load_state_dict(nn_state["scheduler"])
                    except Exception:
                        pass
                self._nn_step_count = nn_state.get("nn_step_count", 0)
                self._move_nets_to_device()
                self.logger.info(f"NN weights loaded from: {nn_path}")
            except Exception as e:
                self.logger.error(f"Failed to load NN weights from {nn_path}: {e}", exc_info=True)
        else:
            self.logger.info(f"No NN weights file at {nn_path} — using fresh network.")

        return success

    # ------------------------------------------------------------------
    # Config dict (extends base)
    # ------------------------------------------------------------------
    def get_config_dict(self) -> dict:
        d = super().get_config_dict()
        d["nn_hidden_sizes"] = list(self._nn_hidden_sizes)
        d["nn_lr"] = self._nn_lr
        return d

    # ------------------------------------------------------------------
    # Pickle support — serialize NN weights as state_dicts
    # ------------------------------------------------------------------
    def __getstate__(self):
        state = super().__getstate__()
        # Replace live PyTorch objects with serializable state_dicts
        state['_policy_net_state_dict'] = self._policy_net.state_dict()
        state['_target_net_state_dict'] = self._target_net.state_dict()
        state['_optimizer_state_dict'] = self._optimizer.state_dict()
        state['_scheduler_state_dict'] = self._lr_scheduler.state_dict()
        state['_torch_device_str'] = str(self._torch_device)
        # Remove live torch objects (not reliably picklable across devices)
        state.pop('_policy_net', None)
        state.pop('_target_net', None)
        state.pop('_optimizer', None)
        state.pop('_lr_scheduler', None)
        state.pop('_torch_device', None)
        state.pop('_last_features', None)
        state.pop('_nstep_buffer', None)
        return state

    def __setstate__(self, state):
        # Pop NN state before base class restore
        policy_sd = state.pop('_policy_net_state_dict', None)
        target_sd = state.pop('_target_net_state_dict', None)
        optim_sd = state.pop('_optimizer_state_dict', None)
        sched_sd = state.pop('_scheduler_state_dict', None)
        device_str = state.pop('_torch_device_str', 'cpu')
        # Restore base class
        super().__setstate__(state)
        # Fill in missing NN config defaults for backward compatibility
        if not hasattr(self, '_nn_dropout'):
            self._nn_dropout = 0.1
        if not hasattr(self, '_nn_attention_heads'):
            self._nn_attention_heads = 4
        if not hasattr(self, '_nn_noisy_sigma'):
            self._nn_noisy_sigma = 0.5
        if not hasattr(self, '_nn_soft_tau'):
            self._nn_soft_tau = 0.005
        if not hasattr(self, '_nn_nstep'):
            self._nn_nstep = 3
        if not hasattr(self, '_training_metrics'):
            self._training_metrics = {
                "loss": deque(maxlen=500), "avg_q_value": deque(maxlen=500),
                "avg_td_error": deque(maxlen=500), "grad_norm": deque(maxlen=500),
                "noise_magnitude": deque(maxlen=500), "learning_rate": deque(maxlen=500),
                "replay_beta": deque(maxlen=500),
            }
        # Rebuild networks
        self._torch_device = torch.device(device_str)
        self._policy_net = DQNNetwork(
            state_dim=NN_STATE_FEATURE_DIM,
            action_dim=self.action_space_size,
            hidden_sizes=self._nn_hidden_sizes,
            dropout=self._nn_dropout,
            attention_heads=self._nn_attention_heads,
            noisy_sigma=self._nn_noisy_sigma,
        )
        self._target_net = self._policy_net.clone()
        self._target_net.eval()
        self._optimizer = optim.AdamW(
            self._policy_net.parameters(),
            lr=self._nn_lr,
            weight_decay=self._nn_weight_decay,
            amsgrad=True,
        )
        self._lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self._optimizer,
            T_0=NN_COSINE_LR_T_MAX,
            T_mult=2,
            eta_min=NN_COSINE_LR_ETA_MIN,
        )
        self._nstep_buffer = _NStepBuffer(
            n_steps=self._nn_nstep,
            gamma=self.discount_factor,
        )
        # Load saved weights
        if policy_sd:
            try:
                self._policy_net.load_state_dict(policy_sd)
            except Exception:
                self.logger.warning("Policy net state_dict mismatch — using fresh weights")
        if target_sd:
            try:
                self._target_net.load_state_dict(target_sd)
            except Exception:
                self._target_net = self._policy_net.clone()
        if optim_sd:
            try:
                self._optimizer.load_state_dict(optim_sd)
            except Exception:
                pass
        if sched_sd:
            try:
                self._lr_scheduler.load_state_dict(sched_sd)
            except Exception:
                pass
        # Move to device
        try:
            self._move_nets_to_device()
        except Exception:
            self._torch_device = torch.device('cpu')
            self._move_nets_to_device()

    # ------------------------------------------------------------------
    # Display / debug
    # ------------------------------------------------------------------
    def display_q_table_summary(self) -> None:
        """Show NN summary in addition to the shadow Q-table."""
        super().display_q_table_summary()
        arch = self._policy_net.get_architecture_summary()
        metrics = self.get_training_metrics_summary()
        nn_info = [
            f"\n{'='*50}",
            f"  ADAPTIVE DQN NEURAL NETWORK SUMMARY",
            f"{'='*50}",
            f"Architecture:     {arch['type']}",
            f"State features:   {arch['state_dim']}",
            f"Hidden layers:    {arch['hidden_sizes']}",
            f"Total parameters: {arch['total_params']:,}",
            f"Memory (params):  {arch['memory_kb']:.1f} KB",
            f"Device:           {self._torch_device}",
            f"",
            f"--- Training Status ---",
            f"Total steps:      {metrics['training_steps']}",
            f"Latest loss:      {metrics.get('loss_latest', 0):.6f}",
            f"Avg Q-value:      {metrics.get('avg_q_value_avg_100', 0):.4f}",
            f"Avg TD-error:     {metrics.get('avg_td_error_avg_100', 0):.4f}",
            f"Grad norm:        {metrics.get('grad_norm_latest', 0):.4f}",
            f"Noise magnitude:  {metrics.get('noise_magnitude_latest', 0):.6f}",
            f"Learning rate:    {metrics.get('learning_rate_latest', self._nn_lr):.2e}",
            f"",
            f"--- Replay Buffer (PER) ---",
            f"Size:             {metrics['replay_size']}/{metrics['replay_capacity']}",
            f"PER beta:         {metrics.get('replay_beta_latest', 0):.4f}",
            f"",
            f"--- Components ---",
            f"Attention:        {arch['has_attention']}",
            f"Feature gate:     {arch['has_feature_gate']}",
            f"NoisyLinear:      {arch['has_noisy_linear']}",
            f"Residual blocks:  {arch['has_residual']}",
            f"{'='*50}",
        ]
        self._send_to_gui("\n".join(nn_info))

    def __repr__(self) -> str:
        return (
            f"PuffinZipAI_NN(Dueling+Noisy+Attention, device={self._torch_device}, "
            f"net={self._policy_net}, steps={self._nn_step_count}, "
            f"replay={len(self._replay_buffer)}/{self._replay_buffer.capacity})"
        )
