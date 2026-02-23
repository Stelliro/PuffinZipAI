# PuffinZipAI_Project/puffinzip_ai/nn_core/replay_buffer.py
"""
**Prioritized Experience Replay (PER) buffer for DQN training.**

Implements the proportional-priority variant from Schaul et al., 2016
("Prioritized Experience Replay") using a Sum-Tree data structure for
O(log N) sampling and priority updates.

Key features:
    * **Priority-based sampling** — transitions with higher TD-error are
      sampled more frequently, accelerating learning from surprising events.
    * **Importance-sampling (IS) weights** — corrects the bias introduced by
      non-uniform sampling, ensuring convergence guarantees.
    * **Annealing β** — IS exponent starts low and anneals to 1.0 over
      training, gradually shifting from prioritised to uniform sampling.
    * **Sum-Tree** — efficient O(log N) proportional sampling.
    * **Backward compatible** — ``sample()`` returns the same tensor tuple
      format as the old uniform buffer, plus indices and IS weights.

Falls back to uniform sampling when α = 0 (equivalent to standard replay).
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, NamedTuple, Optional, Sequence, Tuple

try:
    import torch  # type: ignore[import-unresolved]
    import numpy as np
except ImportError as _e:
    raise ImportError("PyTorch and NumPy are required for replay_buffer.") from _e


class Transition(NamedTuple):
    """A single environment transition."""
    state: np.ndarray       # shape (state_dim,)
    action: int
    reward: float
    next_state: np.ndarray  # shape (state_dim,)
    done: bool              # True if terminal (no next-state bootstrap)


# ---------------------------------------------------------------------------
# Sum-Tree for O(log N) proportional sampling
# ---------------------------------------------------------------------------

class SumTree:
    """Binary sum-tree for efficient proportional priority sampling.

    Stores priorities in a complete binary tree where each parent is the
    sum of its children. Enables O(log N) sampling proportional to priority
    and O(log N) priority updates.

    Parameters
    ----------
    capacity : int
        Maximum number of leaf nodes (transitions).
    """

    __slots__ = ("_capacity", "_tree", "_data", "_write_pos", "_size")

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._data: List[Optional[Transition]] = [None] * capacity
        self._write_pos = 0
        self._size = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self._tree[parent] += change
        if parent > 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, value: float) -> int:
        """Retrieve leaf index for a given cumulative value."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self._tree):
            return idx

        if value <= self._tree[left] or right >= len(self._tree):
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self._tree[left])

    @property
    def total_priority(self) -> float:
        """Sum of all priorities (root node)."""
        return float(self._tree[0])

    @property
    def max_priority(self) -> float:
        """Maximum priority among all stored transitions."""
        end = self._capacity - 1 + self._size
        start = self._capacity - 1
        if self._size == 0:
            return 1.0
        return float(self._tree[start:end].max())

    @property
    def min_priority(self) -> float:
        """Minimum non-zero priority among all stored transitions."""
        end = self._capacity - 1 + self._size
        start = self._capacity - 1
        if self._size == 0:
            return 1.0
        priorities = self._tree[start:end]
        non_zero = priorities[priorities > 0]
        return float(non_zero.min()) if len(non_zero) > 0 else 1.0

    def add(self, priority: float, data: Transition) -> None:
        """Add a transition with the given priority."""
        tree_idx = self._write_pos + self._capacity - 1
        self._data[self._write_pos] = data

        self.update(tree_idx, priority)

        self._write_pos = (self._write_pos + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def update(self, tree_idx: int, priority: float) -> None:
        """Update the priority of a transition at the given tree index."""
        change = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, value: float) -> Tuple[int, float, Transition]:
        """Sample a transition proportional to priority.

        Parameters
        ----------
        value : float
            Random value in [0, total_priority) for proportional selection.

        Returns
        -------
        tree_idx : int
            Index in the tree (for later priority update).
        priority : float
            Priority of the sampled transition.
        data : Transition
            The sampled transition.
        """
        idx = self._retrieve(0, value)
        data_idx = idx - self._capacity + 1
        return idx, float(self._tree[idx]), self._data[data_idx]

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Prioritized Experience Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Prioritized Experience Replay (PER) buffer with Sum-Tree.

    Backward-compatible with the old uniform ReplayBuffer API:
    ``push()``, ``sample()``, ``is_ready()``, ``clear()``, ``capacity``, ``__len__``.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored.
    alpha : float
        Priority exponent. 0 = uniform sampling, 1 = full prioritisation.
        Default 0.6 (recommended by Schaul et al.).
    beta_start : float
        Initial importance-sampling exponent. Anneals to 1.0.
        Default 0.4.
    beta_frames : int
        Number of frames over which β anneals from beta_start to 1.0.
        Default 100_000.
    epsilon : float
        Small constant added to TD-errors to ensure all transitions
        have non-zero sampling probability. Default 1e-6.
    """

    __slots__ = (
        "_capacity", "_alpha", "_beta_start", "_beta_frames", "_epsilon",
        "_tree", "_frame_count", "_max_priority",
    )

    def __init__(
        self,
        capacity: int = 10_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        epsilon: float = 1e-6,
    ) -> None:
        self._capacity = max(1, capacity)
        self._alpha = alpha
        self._beta_start = beta_start
        self._beta_frames = beta_frames
        self._epsilon = epsilon
        self._tree = SumTree(self._capacity)
        self._frame_count = 0
        self._max_priority = 1.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def beta(self) -> float:
        """Current importance-sampling exponent (anneals from beta_start to 1.0)."""
        fraction = min(self._frame_count / max(self._beta_frames, 1), 1.0)
        return self._beta_start + fraction * (1.0 - self._beta_start)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        """Add a transition with maximum priority (ensures it gets sampled at least once)."""
        transition = Transition(state, action, reward, next_state, done)
        # New transitions get max priority so they're sampled at least once
        priority = self._max_priority ** self._alpha
        self._tree.add(priority, transition)

    def clear(self) -> None:
        """Remove all stored transitions."""
        self._tree = SumTree(self._capacity)
        self._frame_count = 0
        self._max_priority = 1.0

    # ------------------------------------------------------------------
    # Sampling (Prioritized)
    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a prioritized mini-batch and return tensors + IS weights.

        Returns
        -------
        states      : Tensor  ``(batch, state_dim)``  float32
        actions     : Tensor  ``(batch, 1)``           int64
        rewards     : Tensor  ``(batch, 1)``           float32
        next_states : Tensor  ``(batch, state_dim)``  float32
        dones       : Tensor  ``(batch, 1)``           float32  (0.0 or 1.0)
        indices     : list[int]   tree indices for priority updates
        is_weights  : Tensor  ``(batch, 1)``           float32  importance-sampling weights
        """
        actual_size = min(batch_size, len(self._tree))
        if actual_size == 0:
            raise ValueError("Cannot sample from empty buffer")

        indices: List[int] = []
        priorities: List[float] = []
        batch: List[Transition] = []

        total = self._tree.total_priority
        segment = total / actual_size

        current_beta = self.beta
        self._frame_count += 1

        for i in range(actual_size):
            low = segment * i
            high = segment * (i + 1)
            value = random.uniform(low, high)
            tree_idx, priority, transition = self._tree.get(value)

            # Safety: skip None transitions (shouldn't happen normally)
            if transition is None:
                value = random.uniform(0, total)
                tree_idx, priority, transition = self._tree.get(value)
                if transition is None:
                    continue

            indices.append(tree_idx)
            priorities.append(priority)
            batch.append(transition)

        if not batch:
            raise ValueError("Failed to sample any valid transitions")

        # Compute importance-sampling weights
        total_p = max(self._tree.total_priority, 1e-10)
        n = max(len(self._tree), 1)
        min_prob = max(self._tree.min_priority, 1e-10) / total_p
        max_weight = (n * min_prob) ** (-current_beta)

        is_weights = []
        for p in priorities:
            prob = max(p, 1e-10) / total_p
            weight = (n * prob) ** (-current_beta)
            is_weights.append(weight / max(max_weight, 1e-10))

        # Build tensors
        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        is_weights_arr = np.array(is_weights, dtype=np.float32)

        return (
            torch.from_numpy(states),
            torch.from_numpy(actions).unsqueeze(1),
            torch.from_numpy(rewards).unsqueeze(1),
            torch.from_numpy(next_states),
            torch.from_numpy(dones).unsqueeze(1),
            indices,
            torch.from_numpy(is_weights_arr).unsqueeze(1),
        )

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """Update priorities for sampled transitions based on TD-errors.

        Parameters
        ----------
        indices : list[int]
            Tree indices returned by ``sample()``.
        td_errors : ndarray
            Absolute TD-errors for the corresponding transitions.
        """
        for idx, td_error in zip(indices, td_errors.flatten()):
            priority = (abs(td_error) + self._epsilon) ** self._alpha
            self._tree.update(idx, priority)
            self._max_priority = max(self._max_priority, abs(td_error) + self._epsilon)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._tree)

    def is_ready(self, min_size: int) -> bool:
        """Return True when the buffer has at least *min_size* transitions."""
        return len(self._tree) >= min_size

    def memory_estimate_bytes(self) -> int:
        """Rough estimate of memory usage in bytes."""
        if len(self._tree) == 0:
            return 0
        # Tree array + data array + per-entry overhead
        tree_bytes = self._tree._tree.nbytes
        per_entry = 200  # estimate per transition
        return tree_bytes + per_entry * len(self._tree)

    def get_stats(self) -> dict:
        """Return buffer statistics for monitoring."""
        return {
            "size": len(self._tree),
            "capacity": self._capacity,
            "fill_ratio": len(self._tree) / max(self._capacity, 1),
            "alpha": self._alpha,
            "beta": self.beta,
            "frame_count": self._frame_count,
            "max_priority": self._max_priority,
            "total_priority": self._tree.total_priority,
        }
