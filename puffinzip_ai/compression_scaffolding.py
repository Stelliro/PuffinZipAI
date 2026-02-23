# PuffinZipAI_Project/puffinzip_ai/compression_scaffolding.py
"""
Compression Scaffolding System — "Training Wheels" for AI Agents
-----------------------------------------------------------------
Provides established compression methods (gzip, zlib, bz2, lzma) as
*reference examples* that agents can study and use — but at a cost.

Agents that lean on reference methods too heavily:
  1. Receive progressively scaled reward penalties
  2. Get temporarily BANNED from the reference method
  3. After the ban cools down, the penalty ramp is even steeper

This forces agents to treat reference methods as *teaching tools*,
not crutches, and incentivises genuine novelty discovery.
"""

import gzip
import zlib
import bz2
import lzma
import logging
import math
from typing import Any, Dict, Optional, Tuple, Callable, List
from dataclasses import dataclass, field

scaffold_logger = logging.getLogger("puffinzip_ai.scaffolding")
if not scaffold_logger.handlers:
    scaffold_logger.setLevel(logging.INFO)
    scaffold_logger.addHandler(logging.NullHandler())


# ==========================================================================
# CONFIGURATION CONSTANTS
# ==========================================================================

# --- Penalty scaling ---
# Reward multiplier starts at this value when using a reference method
REFERENCE_INITIAL_REWARD_MULTIPLIER = 0.60
# Minimum reward multiplier (floor) before a ban kicks in
REFERENCE_MIN_REWARD_MULTIPLIER = 0.10

# --- Reliance tracking ---
# Rolling window size for tracking reference method usage
RELIANCE_WINDOW_SIZE = 50
# Usage ratio threshold that triggers a temporary ban
RELIANCE_BAN_THRESHOLD = 0.50
# Post-ban re-offence: if agent exceeds this ratio within the cooldown
# tracking window, the ban duration doubles
RELIANCE_REPEAT_OFFENCE_THRESHOLD = 0.35

# --- Ban mechanics ---
# How many evaluation rounds a ban lasts
BAN_DURATION_ITEMS = 30
# Maximum ban duration (caps exponential growth from repeat offences)
BAN_MAX_DURATION_ITEMS = 200
# Penalty multiplier for the first evaluation after a ban expires
POST_BAN_PENALTY_MULTIPLIER = 0.30

# --- Reward bonuses ---
# Bonus when agent beats a reference method's ratio with its OWN method
BEAT_REFERENCE_BONUS = 4.0
# Bonus when agent matches reference within 5% using own method
NEAR_REFERENCE_BONUS = 1.5
# Bonus when agent stops using reference after previous reliance
WEANING_BONUS = 2.0
WEANING_WINDOW = 20  # last N items to check for weaning

# --- Progressive generation scaling ---
SCAFFOLDING_GRACE_GENERATIONS = 10   # Full reference access for first N gens
SCAFFOLDING_RAMP_GENERATIONS = 80    # Penalty fully ramped by this generation
SCAFFOLDING_MATURE_GENERATIONS = 150 # Past this, reference methods auto-banned


# ==========================================================================
# REFERENCE COMPRESSION METHODS
# ==========================================================================

def _to_bytes(data) -> bytes:
    """Normalize input to bytes."""
    if isinstance(data, str):
        return data.encode('utf-8')
    return data


def _gzip_compress(data) -> bytes:
    return gzip.compress(_to_bytes(data))

def _gzip_decompress(data: bytes) -> str:
    return gzip.decompress(data).decode('utf-8')

def _zlib_compress(data) -> bytes:
    return zlib.compress(_to_bytes(data))

def _zlib_decompress(data: bytes) -> str:
    return zlib.decompress(data).decode('utf-8')

def _bz2_compress(data) -> bytes:
    return bz2.compress(_to_bytes(data))

def _bz2_decompress(data: bytes) -> str:
    return bz2.decompress(data).decode('utf-8')

def _lzma_compress(data) -> bytes:
    return lzma.compress(_to_bytes(data))

def _lzma_decompress(data: bytes) -> str:
    return lzma.decompress(data).decode('utf-8')


@dataclass
class ReferenceMethod:
    """A known, established compression method used for teaching."""
    name: str
    compress_fn: Callable
    decompress_fn: Callable
    description: str = ""
    # Whether this method works on bytes (True) or strings (False)
    is_binary: bool = True


# The library of available reference methods
REFERENCE_METHODS: Dict[str, ReferenceMethod] = {
    "gzip": ReferenceMethod(
        name="gzip",
        compress_fn=_gzip_compress,
        decompress_fn=_gzip_decompress,
        description="GNU zip — DEFLATE algorithm (LZ77 + Huffman)",
        is_binary=True,
    ),
    "zlib": ReferenceMethod(
        name="zlib",
        compress_fn=_zlib_compress,
        decompress_fn=_zlib_decompress,
        description="zlib — DEFLATE without gzip headers",
        is_binary=True,
    ),
    "bz2": ReferenceMethod(
        name="bz2",
        compress_fn=_bz2_compress,
        decompress_fn=_bz2_decompress,
        description="bzip2 — Burrows-Wheeler + Huffman",
        is_binary=True,
    ),
    "lzma": ReferenceMethod(
        name="lzma",
        compress_fn=_lzma_compress,
        decompress_fn=_lzma_decompress,
        description="LZMA — Lempel-Ziv-Markov chain Algorithm",
        is_binary=True,
    ),
}


def get_reference_method(name: str) -> Optional[ReferenceMethod]:
    """Get a reference method by name."""
    return REFERENCE_METHODS.get(name)


def list_reference_methods() -> List[str]:
    """List available reference method names."""
    return list(REFERENCE_METHODS.keys())


def compress_with_reference(text: str, method_name: str = "gzip") -> Tuple[bytes, int, int]:
    """Compress text with a reference method.

    Returns:
        (compressed_bytes, original_size, compressed_size)
    """
    ref = REFERENCE_METHODS.get(method_name)
    if not ref:
        raise ValueError(f"Unknown reference method: {method_name}")
    original_size = len(text.encode('utf-8'))
    compressed = ref.compress_fn(text)
    return compressed, original_size, len(compressed)


def get_best_reference_ratio(text) -> Tuple[str, float]:
    """Try all reference methods on text and return the best ratio and method name.

    Args:
        text: Input data as str or bytes.

    Returns:
        (best_method_name, best_ratio) where ratio = compressed/original
    """
    if not text:
        return "none", 1.0
    if isinstance(text, str):
        data_bytes = text.encode('utf-8')
    else:
        data_bytes = text
    original_bytes = len(data_bytes)
    if original_bytes == 0:
        return "none", 1.0

    best_name = "none"
    best_ratio = 1.0
    for name, ref in REFERENCE_METHODS.items():
        try:
            compressed = ref.compress_fn(data_bytes)
            ratio = len(compressed) / original_bytes
            if ratio < best_ratio:
                best_ratio = ratio
                best_name = name
        except Exception:
            continue
    return best_name, best_ratio


# ==========================================================================
# PER-AGENT SCAFFOLDING TRACKER
# ==========================================================================

@dataclass
class AgentScaffoldState:
    """Tracks a single agent's reference method usage and ban status."""

    # Rolling history of recent actions: True = used reference, False = own method
    usage_history: list = field(default_factory=list)

    # Cumulative counts
    total_references_used: int = 0
    total_own_methods_used: int = 0

    # Ban state
    is_banned: bool = False
    ban_remaining: int = 0
    ban_count: int = 0  # how many times this agent has been banned

    # Which reference method was last used (for targeted bans)
    last_reference_used: str = ""

    # Post-ban tracking
    post_ban_cooldown: int = 0  # items remaining in post-ban heightened penalty

    def record_action(self, used_reference: bool, reference_name: str = ""):
        """Record whether the agent used a reference method on this item."""
        self.usage_history.append(used_reference)
        if len(self.usage_history) > RELIANCE_WINDOW_SIZE:
            self.usage_history = self.usage_history[-RELIANCE_WINDOW_SIZE:]

        if used_reference:
            self.total_references_used += 1
            self.last_reference_used = reference_name
        else:
            self.total_own_methods_used += 1

    @property
    def reliance_ratio(self) -> float:
        """Current reference method reliance ratio over the rolling window."""
        if not self.usage_history:
            return 0.0
        return sum(1 for u in self.usage_history if u) / len(self.usage_history)

    @property
    def recent_reliance_ratio(self) -> float:
        """Reliance ratio over the most recent WEANING_WINDOW items."""
        recent = self.usage_history[-WEANING_WINDOW:] if self.usage_history else []
        if not recent:
            return 0.0
        return sum(1 for u in recent if u) / len(recent)

    @property
    def lifetime_reliance_ratio(self) -> float:
        """Lifetime reference method reliance ratio."""
        total = self.total_references_used + self.total_own_methods_used
        if total == 0:
            return 0.0
        return self.total_references_used / total

    def tick_ban(self):
        """Decrement ban counter. Call once per evaluation item."""
        if self.is_banned and self.ban_remaining > 0:
            self.ban_remaining -= 1
            if self.ban_remaining <= 0:
                self.is_banned = False
                self.post_ban_cooldown = WEANING_WINDOW
                scaffold_logger.info(
                    f"Agent ban expired (ban #{self.ban_count}). "
                    f"Post-ban penalty active for {self.post_ban_cooldown} items."
                )

        if self.post_ban_cooldown > 0:
            self.post_ban_cooldown -= 1


class ScaffoldingManager:
    """Manages scaffolding state across all agents in a population."""

    def __init__(self):
        self._agent_states: Dict[str, AgentScaffoldState] = {}

    def get_state(self, agent_id: str) -> AgentScaffoldState:
        """Get or create scaffold state for an agent."""
        if agent_id not in self._agent_states:
            self._agent_states[agent_id] = AgentScaffoldState()
        return self._agent_states[agent_id]

    def is_reference_allowed(self, agent_id: str, generation: int = 0) -> bool:
        """Check if an agent is currently allowed to use reference methods.

        Returns False if:
        - Agent is currently banned
        - Agent is past the mature generation threshold
        """
        if generation >= SCAFFOLDING_MATURE_GENERATIONS:
            return False

        state = self.get_state(agent_id)
        return not state.is_banned

    def record_and_check(self, agent_id: str, used_reference: bool,
                         reference_name: str = "", generation: int = 0) -> None:
        """Record an action and check if a ban should be triggered.

        Call this after every evaluation item.
        """
        state = self.get_state(agent_id)
        state.record_action(used_reference, reference_name)
        state.tick_ban()

        # Check if reliance threshold exceeded → trigger ban
        if (not state.is_banned
                and len(state.usage_history) >= 10
                and state.reliance_ratio >= RELIANCE_BAN_THRESHOLD):
            # Trigger ban
            state.ban_count += 1
            # Exponential ban duration (capped)
            duration = min(
                BAN_DURATION_ITEMS * (2 ** (state.ban_count - 1)),
                BAN_MAX_DURATION_ITEMS,
            )
            state.is_banned = True
            state.ban_remaining = int(duration)
            scaffold_logger.info(
                f"Agent {agent_id} BANNED from reference methods for {state.ban_remaining} items "
                f"(ban #{state.ban_count}, reliance={state.reliance_ratio:.0%}). "
                f"Last used: {state.last_reference_used}"
            )

    def calculate_reward_multiplier(self, agent_id: str, generation: int = 0) -> float:
        """Calculate the reward multiplier for a reference method usage.

        Returns a value in [REFERENCE_MIN_REWARD_MULTIPLIER, REFERENCE_INITIAL_REWARD_MULTIPLIER]
        that decreases as:
        - The agent's reliance ratio increases
        - The agent's generation increases
        - The agent is in post-ban cooldown
        """
        state = self.get_state(agent_id)

        # Generation-based scaling
        if generation <= SCAFFOLDING_GRACE_GENERATIONS:
            gen_factor = 1.0  # Full credit during grace period
        elif generation >= SCAFFOLDING_RAMP_GENERATIONS:
            gen_factor = 0.0  # Zero multiplier after ramp
        else:
            progress = (generation - SCAFFOLDING_GRACE_GENERATIONS) / (
                SCAFFOLDING_RAMP_GENERATIONS - SCAFFOLDING_GRACE_GENERATIONS
            )
            gen_factor = 1.0 - progress

        # Reliance-based scaling (more usage → less reward)
        reliance = state.reliance_ratio
        reliance_factor = max(0.0, 1.0 - reliance * 2.0)  # 0% reliance → 1.0, 50% → 0.0

        # Post-ban penalty
        post_ban_factor = POST_BAN_PENALTY_MULTIPLIER if state.post_ban_cooldown > 0 else 1.0

        # Combine factors
        raw_multiplier = REFERENCE_INITIAL_REWARD_MULTIPLIER * gen_factor * reliance_factor * post_ban_factor

        return max(REFERENCE_MIN_REWARD_MULTIPLIER, min(raw_multiplier, REFERENCE_INITIAL_REWARD_MULTIPLIER))

    def calculate_own_method_bonus(self, agent_id: str, own_ratio: float,
                                    text: str, generation: int = 0) -> float:
        """Calculate bonus for an agent using its OWN method vs reference methods.

        Args:
            agent_id: The agent identifier
            own_ratio: The compression ratio the agent achieved (compressed/original)
            text: The original text (used to compute reference ratio)
            generation: Current generation

        Returns:
            Bonus reward to add (always >= 0)
        """
        state = self.get_state(agent_id)
        bonus = 0.0

        # Compare against best reference method
        _, best_ref_ratio = get_best_reference_ratio(text)

        if own_ratio < best_ref_ratio:
            # BEAT the reference! Scale by margin
            margin = best_ref_ratio - own_ratio
            bonus += BEAT_REFERENCE_BONUS * min(margin / 0.1, 3.0)
        elif own_ratio < best_ref_ratio * 1.05:
            # Within 5% — close enough for a smaller bonus
            bonus += NEAR_REFERENCE_BONUS

        # Weaning bonus: agent was reliant but is now using own methods
        if (state.lifetime_reliance_ratio > 0.2
                and state.recent_reliance_ratio < 0.1
                and len(state.usage_history) >= WEANING_WINDOW):
            bonus += WEANING_BONUS

        return bonus

    def get_population_scaffold_stats(self) -> Dict[str, Any]:
        """Get aggregate scaffolding statistics for the population."""
        if not self._agent_states:
            return {"total_agents": 0}

        total = len(self._agent_states)
        banned = sum(1 for s in self._agent_states.values() if s.is_banned)
        avg_reliance = sum(s.reliance_ratio for s in self._agent_states.values()) / total
        avg_lifetime = sum(s.lifetime_reliance_ratio for s in self._agent_states.values()) / total
        total_bans = sum(s.ban_count for s in self._agent_states.values())

        return {
            "total_agents": total,
            "currently_banned": banned,
            "avg_reliance_ratio": round(avg_reliance, 4),
            "avg_lifetime_reliance": round(avg_lifetime, 4),
            "total_bans_issued": total_bans,
        }

    def reset_agent(self, agent_id: str):
        """Reset scaffolding state for an agent (e.g., on population replacement)."""
        if agent_id in self._agent_states:
            del self._agent_states[agent_id]

    def clear_all(self):
        """Clear all scaffolding state."""
        self._agent_states.clear()


# ==========================================================================
# MODULE-LEVEL SINGLETON
# ==========================================================================

_GLOBAL_SCAFFOLDING_MANAGER = ScaffoldingManager()


def get_scaffolding_manager() -> ScaffoldingManager:
    """Get the global scaffolding manager singleton."""
    return _GLOBAL_SCAFFOLDING_MANAGER


# ==========================================================================
# SELF-TEST
# ==========================================================================

if __name__ == "__main__":
    print("--- Compression Scaffolding System Test ---\n")

    # Test reference methods
    test_text = "AAABBBCCCDDDEEEFFFGGGHHHIIIJJJ" * 10
    print(f"Test text: {len(test_text)} chars")

    for name in list_reference_methods():
        _, orig, comp = compress_with_reference(test_text, name)
        ratio = comp / orig
        print(f"  {name}: {orig} → {comp} bytes (ratio={ratio:.4f})")

    best_name, best_ratio = get_best_reference_ratio(test_text)
    print(f"\n  Best reference: {best_name} (ratio={best_ratio:.4f})")

    # Test scaffolding manager
    print("\n--- Scaffolding Manager Test ---")
    mgr = ScaffoldingManager()

    agent_id = "test_agent_001"

    # Simulate heavy reference usage → ban
    for i in range(30):
        allowed = mgr.is_reference_allowed(agent_id)
        used_ref = allowed and (i % 3 != 0)  # Use reference 2/3 of the time
        mgr.record_and_check(agent_id, used_ref, "gzip")

        state = mgr.get_state(agent_id)
        if state.is_banned:
            print(f"  Item {i+1}: BANNED (remaining={state.ban_remaining}, "
                  f"ban #{state.ban_count})")
        elif used_ref:
            mult = mgr.calculate_reward_multiplier(agent_id)
            print(f"  Item {i+1}: Used reference (mult={mult:.2f}, "
                  f"reliance={state.reliance_ratio:.0%})")
        else:
            print(f"  Item {i+1}: Used own method")

    # Test reward multiplier across generations
    print("\n--- Generation Scaling ---")
    mgr2 = ScaffoldingManager()
    for gen in [0, 5, 20, 50, 80, 120, 150]:
        mult = mgr2.calculate_reward_multiplier("gen_test", gen)
        allowed = mgr2.is_reference_allowed("gen_test", gen)
        print(f"  Gen {gen:3d}: multiplier={mult:.3f}, allowed={allowed}")

    print("\n--- Population Stats ---")
    print(mgr.get_population_scaffold_stats())
    print("\n--- Test Complete ---")
