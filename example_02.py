# na podstawie przykładu: https://pypi.org/project/pygad/1.0.18/
import logging
import pygad
import numpy
import benchmark_functions as bf
import matplotlib.pyplot as plt
import os
import datetime

num_genes = 2
func = bf.Hypersphere(n_dimensions=num_genes)


def fitness_func(ga_instance, solution, solution_idx):
    fitness = func(solution)
    return 1. / fitness


fitness_function = fitness_func
num_generations = 100
sol_per_pop = 80
num_parents_mating = 50
boundary = func.suggested_bounds()
init_range_low = 0
init_range_high = 2
gene_type = float
mutation_num_genes = 1



level = logging.DEBUG
name = 'logfile.txt'
logger = logging.getLogger(name)
logger.setLevel(level)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

objective_best = []
objective_mean = []
objective_std = []
gen_results = dict()


def on_generation(ga_instance):
    ga_instance.logger.info("Generation = {generation}".format(generation=ga_instance.generations_completed))
    solution, solution_fitness, solution_idx = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness)
    ga_instance.logger.info("Best    = {fitness}".format(fitness=1. / solution_fitness))
    ga_instance.logger.info("Individual    = {solution}".format(solution=repr(solution)))

    tmp = [1. / x for x in ga_instance.last_generation_fitness]

    ga_instance.logger.info("Min    = {min}".format(min=numpy.min(tmp)))
    ga_instance.logger.info("Max    = {max}".format(max=numpy.max(tmp)))
    ga_instance.logger.info("Average    = {average}".format(average=numpy.average(tmp)))
    ga_instance.logger.info("Std    = {std}".format(std=numpy.std(tmp)))
    ga_instance.logger.info("\r\n")

    objective_best.append(numpy.min(tmp))
    objective_mean.append(numpy.mean(tmp))
    objective_std.append(numpy.std(tmp))
    gen_results[ga_instance.generations_completed] = {
        "best": numpy.min(tmp),
        "mean": numpy.mean(tmp),
        "std": numpy.std(tmp),
        "max": numpy.max(tmp),
        "min": numpy.min(tmp),
        "individual": repr(solution),
        "value": 1. / solution_fitness
    }


# Genetic Algorithm parameters - Adjust these parameters
parent_selection_type = "tournament" # "random", "rws", "tournament"
crossover_type = "single_point" # "single_point", "two_points", "uniform"
mutation_type = "random" # "random", "swap"


ga_instance = pygad.GA(num_generations=num_generations,
                       sol_per_pop=sol_per_pop,
                       num_parents_mating=num_parents_mating,
                       num_genes=num_genes,
                       fitness_func=fitness_func,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       mutation_num_genes=mutation_num_genes,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       keep_elitism=1,
                       K_tournament=3,
                       random_mutation_max_val=10,
                       random_mutation_min_val=-10,
                       logger=logger,
                       on_generation=on_generation,
                       parallel_processing=['thread', 4],
                       gene_type=gene_type
                       )



if __name__ == "__main__":
    ga_instance.run()

    best = ga_instance.best_solution()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=1. / solution_fitness))
    print("Minimum value: ", min(func.minimum()))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(os.path.dirname(__file__), "results", timestamp+f"_{parent_selection_type}_{crossover_type}_{mutation_type}")

    os.makedirs(results_dir, exist_ok=True)

    ga_instance.best_solutions_fitness = [1. / x for x in ga_instance.best_solutions_fitness]
    plt.figure()
    plt.plot(ga_instance.best_solutions_fitness)
    plt.xlabel('Generation')
    plt.ylabel('Objective function value')
    plt.title('Objective Function Value per Generation')
    plt.savefig(os.path.join(results_dir, "fitness.png"))
    plt.show()

    generations = range(1, len(objective_best) + 1)

    plt.figure()
    plt.plot(generations, objective_mean, label='Mean', color='green')
    plt.xlabel('Generation')
    plt.ylabel('Mean objective function value')
    plt.title('Mean Objective Function Value per Generation')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "mean.png"))
    plt.close()

    plt.figure()
    plt.plot(generations, objective_std, label='Std', color='red')
    plt.xlabel('Generation')
    plt.ylabel('Standard deviation of objective function value')
    plt.title('Standard Deviation per Generation')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "std.png"))
    plt.close()

    results_txt_path = os.path.join(results_dir, "results.csv")
    with open(results_txt_path, "w") as f:
        f.write("Generation;Best;Mean;Std;Max;Min;Individual;Value\n")
        for gen, vals in gen_results.items():
            f.write(
                f"{gen};{vals['best']};{vals['mean']};{vals['std']};{vals['max']};{vals['min']};{vals['individual']};{vals['value']}\n")



