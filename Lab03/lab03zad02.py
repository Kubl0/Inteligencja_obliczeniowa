import time
import numpy as np
import pygad

labirynth = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

gene_space = [0, 1, 2, 3, 4]

# 1-prawo
# 2-lewo
# 3-gora
# 4-dol


def fitness_func(solution, solution_idx):
    x = 1
    y = 1
    steps = 0
    wallHits = 0
    endPoint = [10, 10]

    filter(lambda x: x != 0, solution)

    for gene in solution:
        if gene == 0:
            continue
        if gene == 1:
            if labirynth[y][x + 1] == 0:
                wallHits += 1
            else:
                x += 1
                steps += 1
        elif gene == 2:
            if labirynth[y][x - 1] == 0:
                wallHits += 1
            else:
                x -= 1
                steps += 1
        elif gene == 3:
            if labirynth[y - 1][x] == 0:
                wallHits += 1
            else:
                y -= 1
                steps += 1
        elif gene == 4:
            if labirynth[y + 1][x] == 0:
                wallHits += 1
            else:
                y += 1
                steps += 1
        if steps >= 30:
            break
        if x == endPoint[0] and y == endPoint[1]:
            break

    end_distance = abs(x - endPoint[0]) + abs(y - endPoint[1])
    fitness = 100 / (steps + wallHits + end_distance * 100)

    return fitness


fitness_function = fitness_func

num_genes = 30
parent_selection_number = 10
number_generations = 10000
keep_parents = 4
parent_selection_type = "sss"
mutation_percent_genes = 8
sol_per_pop = 100
stop_criteria = "reach_5.0"


start = time.time()
ga_instance = pygad.GA(
    gene_space=gene_space,
    num_generations=number_generations,
    num_parents_mating=parent_selection_number,
    fitness_func=fitness_function,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    parent_selection_type=parent_selection_type,
    keep_parents=keep_parents,
    mutation_percent_genes=mutation_percent_genes,
    stop_criteria=stop_criteria,
)

ga_instance.run()
end = time.time()

time = end - start

with open("./times.txt", "a") as file:
    file.write(str(time) + "\n")


best_solution, best_solution_fitness, best_match_idx = ga_instance.best_solution()

print("Best solution: ", best_solution)
print("Best solution fitness: ", best_solution_fitness)
with open("times.txt", "r") as file:
    lines = file.readlines()
    lines = lines[-10:]
    lines = list(map(lambda x: float(x), lines))
    average = sum(lines) / len(lines)
    print("Average time: ", average, "s")
ga_instance.plot_fitness()
