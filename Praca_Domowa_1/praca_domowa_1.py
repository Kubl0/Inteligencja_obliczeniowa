import numpy as np
import pygad
import math
import time

board_to_solve = np.array(
    [[0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0],
     [3, 0, 0, 0, 0, 2],
     [2, 0, 0, 0, 0, 0],
     [0, 0, 8, 0, 0, 0]]
)
sudoku_width = 6
sudoku_size = 36


def fitness_func(solution, solution_idx):
    # Reshape solution into a 9x9 sudoku grid
    sudoku = np.reshape(solution, (sudoku_width, sudoku_width))

    for i in range(sudoku_width):
        for j in range(sudoku_width):
            if board_to_solve[i][j] != 0:
                sudoku[i][j] = board_to_solve[i][j]

    # Calculate the fitness of the solution
    fitness = 1

    for i in range(sudoku_width):
        # Check the row
        fitness += sudoku_width - len(np.unique(sudoku[i, :]))

        # Check the column
        fitness += sudoku_width - len(np.unique(sudoku[:, i]))

    # Check the 3x3 subgrids
    for i in range(sudoku_width // 3):
        for j in range(sudoku_width // 3):
            fitness += 9 - len(np.unique(sudoku[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3]))

    return 1 / fitness


num_generations = 5000
parent_selection_number = 20
keep_parents = 8
parent_selection_type = "sss"
mutation_percent_genes = 1 / sudoku_size * 100
sol_per_pop = 200

start = time.time()
initial_population = []
for i in range(sol_per_pop):
    initial_population.append(np.random.randint(1, 10, size=sudoku_size))

ga_instance = pygad.GA(initial_population=initial_population,
                       num_generations=num_generations,
                       num_parents_mating=parent_selection_number,
                       keep_parents=keep_parents,
                       parent_selection_type=parent_selection_type,
                       mutation_percent_genes=mutation_percent_genes,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=sudoku_size,
                       gene_type=int,
                       crossover_type="two_points",
                       mutation_type="random",
                       gene_space=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                       stop_criteria="reach_1.0"
                       )

ga_instance.run()

end = time.time()

best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()

solution_array = np.reshape(best_solution, (sudoku_width, sudoku_width))

print("Solution : \n", solution_array, best_solution_fitness)
print("Time : ", end - start)
