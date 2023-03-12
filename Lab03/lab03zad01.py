import pygad
import math


def endurance(solution, solution_idx):
    x, y, z, u, v, w = solution
    return math.exp(-2 * (y - math.sin(x)) ** 2) + math.sin(z * u) + math.cos(v * w)


gene_space = [{"low": 0, "high": 1} for i in range(6)]

fitness_function = endurance

sol_per_pop = 50
num_genes = 6
num_generations = 100
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 15
num_parents_mating = 4
keep_parents = 2
crossover_probability = 0.8
mutation_probability = 0.1


ga_instance = pygad.GA(
    gene_space=gene_space,
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_function,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    parent_selection_type=parent_selection_type,
    keep_parents=keep_parents,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    mutation_percent_genes=mutation_percent_genes,
    crossover_probability=crossover_probability,
    mutation_probability=mutation_probability,
)


ga_instance.run()
best_solution, best_solution_fitness, best_match_idx = ga_instance.best_solution()

print("Best solution : {solution}".format(solution=best_solution))
print(
    "Best solution fitness : {solution_fitness}".format(
        solution_fitness=best_solution_fitness
    )
)
ga_instance.plot_fitness()
