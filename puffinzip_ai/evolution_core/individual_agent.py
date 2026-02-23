# PuffinZipAI_Project/puffinzip_ai/evolution_core/individual_agent.py
"""
EvolvingAgent — wrapper around a PuffinZipAI core that participates in
evolutionary optimization.

Key concepts added in v0.9.7:
  * **Heritage / lineage memory ("grandpapi" lineage)**
    Each agent carries a ``heritage`` dict recording the best tricks learned
    from its ancestors.  During breeding the child inherits heritage entries
    from *both* parents (and transitively from grandparents, great-
    grandparents, etc.) because the parent's heritage already contains their
    ancestors' entries.  This gives every child "grandpapi memory" without
    needing an external lineage database.

  * **Agent type** (``agent_type``)
    ``"compression"`` — evaluated on clean data only.
    ``"anti_corruption"`` — evaluated on corrupted/noisy data.
    Cross-type breeding produces hybrids that inherit traits from both
    specializations.

  * **Dual fitness**
    ``compression_fitness``  — how well the agent compresses clean data.
    ``robustness_fitness``   — how well the agent handles corrupted data.
    The combined ``fitness`` property is the primary sort key and equals
    whichever score matches the agent's type (or a weighted blend for
    hybrids born from cross-type breeding).
"""
from __future__ import annotations

import copy
import traceback
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ai_core import PuffinZipAI
else:
    from ..ai_core import PuffinZipAI

# ---------------------------------------------------------------------------
#  Heritage entry — one "trick" remembered from an ancestor
# ---------------------------------------------------------------------------
# Each entry is a plain dict so it serializes trivially with pickle/JSON:
#   {
#       "trick":        str   — human-readable label (e.g. "bwt_mtf_rle pipeline"),
#       "pipeline":     str   — novel_compression_generator pipeline name,
#       "discovery_seed": int | None,
#       "rle_min_run":  int,
#       "fitness_when_learned": float,
#       "ancestor_id":  str   — agent that first discovered this trick,
#       "generation":   int   — generation when the trick was first learned,
#   }

# Maximum heritage entries per agent to bound memory usage
MAX_HERITAGE_ENTRIES = 32


class EvolvingAgent:
    """Wraps a PuffinZipAI core with evolutionary metadata.

    Attributes (new in v0.9.7):
        heritage:  list[dict]  — "grandpapi" lineage memory (see module docstring).
        agent_type: str        — ``"compression"`` or ``"anti_corruption"``.
        compression_fitness:   float — fitness on clean data.
        robustness_fitness:    float — fitness on corrupted data.
    """

    def __init__(self, puffin_ai_instance: PuffinZipAI,  # type: ignore[valid-type]
                 agent_id: str | None = None,
                 generation_born: int = 0,
                 parent_ids: list | None = None,
                 agent_type: str = "compression",
                 heritage: list | None = None):

        if not isinstance(puffin_ai_instance, PuffinZipAI):
            raise TypeError("puffin_ai_instance must be an instance of PuffinZipAI.")

        self.puffin_ai: PuffinZipAI = puffin_ai_instance  # type: ignore[valid-type]
        self.agent_id: str = agent_id if agent_id is not None else str(uuid.uuid4())
        self.fitness: float = 0.0
        self.generation_born: int = generation_born
        self.parent_ids: list = parent_ids if parent_ids is not None else []
        self.evaluation_stats: dict = {}

        # --- v0.9.7: Agent type (compression vs anti-corruption) ---
        # Valid values: "compression", "anti_corruption"
        self.agent_type: str = agent_type if agent_type in ("compression", "anti_corruption") else "compression"

        # --- v0.9.7: Dual fitness scores ---
        self.compression_fitness: float = 0.0
        self.robustness_fitness: float = 0.0

        # --- v0.9.7: Heritage / "grandpapi" lineage memory ---
        # Inherited from parents during breeding; each entry records a
        # successful trick from an ancestor.  Parents' heritage already
        # contains *their* ancestors' entries, so the child transitively
        # inherits the full family tree of tricks without storing the entire
        # agent graph — this is the "grandpapi" pattern.
        self.heritage: list = list(heritage) if heritage else []

        # Link scaffold tracking ID so it persists across clones
        if hasattr(self.puffin_ai, '_scaffold_agent_id'):
            self.puffin_ai._scaffold_agent_id = self.agent_id

    def get_puffin_ai(self) -> PuffinZipAI:  # type: ignore[valid-type]
        return self.puffin_ai

    def set_fitness(self, fitness_score: float):
        """Set the primary (clean / compression) fitness.

        Does NOT auto-update type-specific sub-scores.  Use
        ``set_compression_fitness`` / ``set_robustness_fitness`` for those.
        ``get_fitness()`` always returns this value, regardless of agent type.
        """
        self.fitness = fitness_score

    def set_compression_fitness(self, score: float):
        """Set the compression sub-score only."""
        self.compression_fitness = score

    def set_robustness_fitness(self, score: float):
        """Set the robustness sub-score only.

        Does NOT touch ``self.fitness`` so that ``get_fitness()`` always
        returns the clean / compression fitness.
        """
        self.robustness_fitness = score

    def get_fitness(self) -> float:
        return self.fitness

    # -----------------------------------------------------------------
    #  Heritage helpers — "grandpapi" lineage memory
    # -----------------------------------------------------------------
    def record_trick(self, trick_label: str, pipeline: str,
                     discovery_seed, rle_min_run: int, fitness: float,
                     generation: int):
        """Record a newly-discovered trick into this agent's heritage.

        This entry will be inherited by all future descendants.
        """
        entry = {
            "trick": trick_label,
            "pipeline": pipeline,
            "discovery_seed": discovery_seed,
            "rle_min_run": rle_min_run,
            "fitness_when_learned": fitness,
            "ancestor_id": self.agent_id,
            "generation": generation,
        }
        self.heritage.append(entry)
        # Keep bounded
        if len(self.heritage) > MAX_HERITAGE_ENTRIES:
            # Drop the weakest (lowest fitness_when_learned) entries
            self.heritage.sort(key=lambda e: e.get("fitness_when_learned", 0.0))
            self.heritage = self.heritage[-MAX_HERITAGE_ENTRIES:]

    @staticmethod
    def merge_heritage(parent1_heritage: list, parent2_heritage: list) -> list:
        """Merge heritage from two parents, deduplicating by ancestor_id + trick.

        This is the core of "grandpapi" inheritance: each parent already
        carries their own ancestors' entries, so the merged set contains
        tricks from grandparents, great-grandparents, etc. automatically.
        """
        seen = set()
        merged = []
        for entry in (parent1_heritage or []) + (parent2_heritage or []):
            key = (entry.get("ancestor_id", ""), entry.get("trick", ""))
            if key not in seen:
                seen.add(key)
                merged.append(copy.copy(entry))
        # Keep only the top MAX_HERITAGE_ENTRIES by fitness
        if len(merged) > MAX_HERITAGE_ENTRIES:
            merged.sort(key=lambda e: e.get("fitness_when_learned", 0.0))
            merged = merged[-MAX_HERITAGE_ENTRIES:]
        return merged

    def get_best_heritage_pipeline(self):
        """Return the metadata dict of the highest-fitness trick in heritage,
        or None if heritage is empty.  Used during breeding to prefer
        inheriting successful novel methods from ancestors.
        """
        if not self.heritage:
            return None
        best = max(self.heritage, key=lambda e: e.get("fitness_when_learned", 0.0))
        return best

    def clone(self, new_agent_id: str | None = None, new_generation_born: int | None = None):
        if not hasattr(self.puffin_ai, 'clone_core_model') or not callable(self.puffin_ai.clone_core_model):
            raise NotImplementedError("PuffinZipAI instance must have a 'clone_core_model()' method.")

        cloned_puffin_ai = self.puffin_ai.clone_core_model()
        clone_id = new_agent_id if new_agent_id is not None else str(uuid.uuid4())
        clone_generation = new_generation_born if new_generation_born is not None else self.generation_born

        cloned_agent = EvolvingAgent(
            puffin_ai_instance=cloned_puffin_ai,
            agent_id=clone_id,
            generation_born=clone_generation,
            parent_ids=[self.agent_id],
            agent_type=self.agent_type,
            # Heritage is copied so the clone inherits the full family history
            heritage=list(self.heritage),
        )
        # Carry over dual fitness scores
        cloned_agent.compression_fitness = self.compression_fitness
        cloned_agent.robustness_fitness = self.robustness_fitness
        # EvolvingAgent.__init__ already sets _scaffold_agent_id = clone_id
        return cloned_agent

    def __repr__(self):
        thresholds_str = str(self.puffin_ai.len_thresholds) if self.puffin_ai else 'N/A'
        heritage_count = len(self.heritage) if self.heritage else 0
        return (f"EvolvingAgent(ID:{self.agent_id},Gen:{self.generation_born},"
                f"Fit:{self.fitness:.4f},Type:{self.agent_type},"
                f"CFit:{self.compression_fitness:.4f},RFit:{self.robustness_fitness:.4f},"
                f"Heritage:{heritage_count},Parents:{self.parent_ids},Thresh:{thresholds_str})")

    # -----------------------------------------------------------------
    #  Pickle support — preserve v0.9.7 fields across checkpoint/restart
    # -----------------------------------------------------------------
    def __getstate__(self):
        """Serialize ALL v0.9.7 fields so they survive checkpoint/restart.

        The underlying PuffinZipAI core has its own __getstate__ that strips
        unpicklable closures; we don't need to touch it here — pickle will
        call it automatically when serializing self.puffin_ai.
        """
        state = self.__dict__.copy()
        # Ensure v0.9.7 fields are present even if something cleared them
        state.setdefault('heritage', [])
        state.setdefault('agent_type', 'compression')
        state.setdefault('compression_fitness', 0.0)
        state.setdefault('robustness_fitness', 0.0)
        return state

    def __setstate__(self, state):
        """Restore from pickle, providing defaults for any missing v0.9.7
        fields.  This handles loading pre-v0.9.7 checkpoints that were
        saved without heritage/agent_type/dual-fitness.
        """
        self.__dict__.update(state)
        # --- Backward-compatible migration from pre-v0.9.7 checkpoints ---
        if not hasattr(self, 'heritage'):
            self.heritage = []
        if not hasattr(self, 'agent_type'):
            self.agent_type = 'compression'
        if not hasattr(self, 'compression_fitness'):
            self.compression_fitness = getattr(self, 'fitness', 0.0) or 0.0
        if not hasattr(self, 'robustness_fitness'):
            self.robustness_fitness = 0.0

    def __lt__(self, other):
        if not isinstance(other, EvolvingAgent):
            return NotImplemented
        return self.fitness < other.fitness


if __name__ == "__main__":
    print("--- Testing EvolvingAgent ---")

    class MockPuffinZipAI(PuffinZipAI):
        def __init__(self, len_thresholds=None, **kwargs):
            super().__init__(len_thresholds=(len_thresholds if len_thresholds is not None else [10, 20]))
            if hasattr(self, 'logger') and hasattr(self.logger, 'disabled'):
                self.logger.disabled = True  # type: ignore[union-attr]

        def clone_core_model(self):
            import numpy
            cloned = MockPuffinZipAI(len_thresholds=list(self.len_thresholds))
            if hasattr(self, 'q_table') and self.q_table is not None:
                cloned.q_table = numpy.copy(self.q_table)
            if hasattr(self, 'exploration_rate'):
                cloned.exploration_rate = self.exploration_rate
            return cloned

    original_PuffinZipAI_ref = PuffinZipAI
    PuffinZipAI = MockPuffinZipAI

    try:
        base_ai1 = PuffinZipAI(len_thresholds=[10, 50, 100])
        base_ai2 = PuffinZipAI(len_thresholds=[15, 60, 120])

        agent1 = EvolvingAgent(puffin_ai_instance=base_ai1, agent_id="agent_001", generation_born=1)
        agent1.set_fitness(10.5789)
        print(agent1)

        agent2 = EvolvingAgent(puffin_ai_instance=base_ai2, generation_born=1, parent_ids=["ancestor_X"])
        agent2.set_fitness(12.3123)
        print(agent2)

        cloned_agent1 = agent1.clone(new_agent_id="cloned_A", new_generation_born=2)
        cloned_agent1.set_fitness(11.0)
        print(f"\nCloned Agent: {cloned_agent1}")

        agent_list = [agent2, agent1, cloned_agent1]
        agent_list.sort(key=lambda ag: ag.get_fitness(), reverse=True)
        print("\nAgents sorted by fitness (desc):")
        for ag in agent_list:
            print(f"  ID: {ag.agent_id}, Fitness: {ag.get_fitness():.4f}")

    except Exception as e:
        print(f"Error during EvolvingAgent test: {e}")
        traceback.print_exc()
    finally:
        PuffinZipAI = original_PuffinZipAI_ref