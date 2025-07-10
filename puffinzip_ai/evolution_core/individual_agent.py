# PuffinZipAI_Project/puffinzip_ai/evolution_core/individual_agent.py
import traceback
import uuid
from ..ai_core import PuffinZipAI

class EvolvingAgent:
    def __init__(self, puffin_ai_instance: PuffinZipAI,
                 agent_id: str = None,
                 generation_born: int = 0,
                 parent_ids: list = None):

        if not isinstance(puffin_ai_instance, PuffinZipAI):
            raise TypeError("puffin_ai_instance must be an instance of PuffinZipAI.")

        self.puffin_ai: PuffinZipAI = puffin_ai_instance
        self.agent_id: str = agent_id if agent_id is not None else str(uuid.uuid4())
        self.fitness: float = 0.0
        self.generation_born: int = generation_born
        self.parent_ids: list = parent_ids if parent_ids is not None else []
        self.evaluation_stats: dict = {}

    def get_puffin_ai(self) -> PuffinZipAI:
        return self.puffin_ai

    def set_fitness(self, fitness_score: float):
        self.fitness = fitness_score

    def get_fitness(self) -> float:
        return self.fitness

    def clone(self, new_agent_id: str = None, new_generation_born: int = None):
        if not hasattr(self.puffin_ai, 'clone_core_model') or not callable(self.puffin_ai.clone_core_model):
            raise NotImplementedError("PuffinZipAI instance must have a 'clone_core_model()' method.")

        cloned_puffin_ai = self.puffin_ai.clone_core_model()
        clone_id = new_agent_id if new_agent_id is not None else str(uuid.uuid4())
        clone_generation = new_generation_born if new_generation_born is not None else self.generation_born

        cloned_agent = EvolvingAgent(
            puffin_ai_instance=cloned_puffin_ai,
            agent_id=clone_id,
            generation_born=clone_generation,
            parent_ids=[self.agent_id]
        )
        return cloned_agent

    def __repr__(self):
        thresholds_str = str(self.puffin_ai.len_thresholds) if self.puffin_ai else 'N/A'
        return (f"EvolvingAgent(ID:{self.agent_id},Gen:{self.generation_born},"
                f"Fit:{self.fitness:.4f},Parents:{self.parent_ids},Thresh:{thresholds_str})")

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
                self.logger.disabled = True

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