# PuffinZipAI_Project/puffinzip_ai/evolution_core/selection_methods.py
import random
import numpy as np

def tournament_selection(population: list, num_to_select: int, tournament_size: int = 3) -> list:
    if not population or num_to_select <= 0:
        return []
    selected_parents = []
    pop_size = len(population)
    if pop_size == 0: return []
    actual_tournament_size = min(tournament_size, pop_size)
    for _ in range(num_to_select):
        participants = random.sample(population, actual_tournament_size) if pop_size > actual_tournament_size else population
        winner = max(participants, key=lambda agent: agent.fitness)
        selected_parents.append(winner)
    return selected_parents

def roulette_wheel_selection(population: list, num_to_select: int) -> list:
    if not population or num_to_select <= 0:
        return []
    fitness_values = np.array([agent.fitness for agent in population])
    min_fitness = np.min(fitness_values)
    if min_fitness < 0:
        fitness_values -= (min_fitness - 1e-6)
    total_fitness = np.sum(fitness_values)
    if total_fitness == 0:
        return random.choices(population, k=num_to_select) if population else []
    probabilities = fitness_values / total_fitness
    selected_indices = np.random.choice(len(population), size=num_to_select, p=probabilities, replace=True)
    selected_parents = [population[i] for i in selected_indices]
    return selected_parents

def rank_selection(population: list, num_to_select: int) -> list:
    if not population or num_to_select <= 0:
        return []
    sorted_population = sorted(population, key=lambda agent: agent.fitness, reverse=True)
    ranks = np.arange(len(sorted_population), 0, -1)
    total_rank_sum = np.sum(ranks)
    if total_rank_sum == 0:
        return random.choices(population, k=num_to_select) if population else []
    probabilities = ranks / total_rank_sum
    selected_indices_in_sorted = np.random.choice(len(sorted_population),size=num_to_select,p=probabilities,replace=True)
    selected_parents = [sorted_population[i] for i in selected_indices_in_sorted]
    return selected_parents

def truncation_selection_for_breeding(population: list, num_to_select_for_breeding_pool: int) -> list:
    if not population or num_to_select_for_breeding_pool <= 0:
        return []
    sorted_population = sorted(population, key=lambda agent: agent.fitness, reverse=True)
    return sorted_population[:min(num_to_select_for_breeding_pool, len(sorted_population))]

if __name__ == "__main__":
    print("--- Testing Selection Methods ---")
    class MockEvolvingAgent:
        def __init__(self,id_num,fit_score):self.agent_id=f"agent_{id_num}";self.fitness=fit_score
        def __repr__(self):return f"MockAgent(id='{self.agent_id}',fit={self.fitness:.2f})"
    pop=[MockEvolvingAgent(1,10.0),MockEvolvingAgent(2,50.0),MockEvolvingAgent(3,5.0),MockEvolvingAgent(4,25.0),MockEvolvingAgent(5,25.0),MockEvolvingAgent(6,1.0)]
    num_parents=4;print(f"\nOrig Pop({len(pop)}):");[print(f"  {ag}")for ag in pop]
    print(f"\n--- Tourn Sel(k=3,n={num_parents}) ---");ts_parents=tournament_selection(pop,num_parents,3);print(f"Sel {len(ts_parents)}:");[print(f"  {p}")for p in ts_parents]
    print(f"\n--- Roulette Sel(n={num_parents}) ---");rs_parents=roulette_wheel_selection(pop,num_parents);print(f"Sel {len(rs_parents)}:");[print(f"  {p}")for p in rs_parents]
    print(f"\n--- Rank Sel(n={num_parents}) ---");rank_parents=rank_selection(pop,num_parents);print(f"Sel {len(rank_parents)}:");[print(f"  {p}")for p in rank_parents]
    num_pool=3;print(f"\n--- Trunc Sel(n={num_pool}) ---");pool=truncation_selection_for_breeding(pop,num_pool);print(f"Sel {len(pool)}:");[print(f"  {i}")for i in pool]