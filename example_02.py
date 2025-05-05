import logging
import pygad
import numpy
import benchmark_functions as bf
import matplotlib.pyplot as plt
import os
import datetime

num_genes = 2
bits_per_gene = 20
total_bits = num_genes * bits_per_gene
gene_type = int  # Can be float or int

func = bf.Hypersphere(n_dimensions=num_genes)

def decode_individual(individual, low=-5, high=5):
    decoded = []
    for i in range(num_genes):
        start = i * bits_per_gene
        end = start + bits_per_gene
        gene_bits = individual[start:end]
        gene_int = 0
        for bit in gene_bits:
            gene_int = (gene_int << 1) | int(bit)

        max_int = (2 ** bits_per_gene) - 1
        gene_float = (gene_int / max_int) * (high - low) + low
        decoded.append(gene_float)

    return decoded


def fitness_func(ga_instance, solution, solution_idx):
    if gene_type == int:
        ind = decode_individual(solution)
    else:
        ind = solution
    fitness = func(ind)
    return -fitness


logger_name = 'logfile.txt'
logger = logging.getLogger(logger_name)
if not logger.hasHandlers():
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

fitness_function = fitness_func
num_generations = 100
sol_per_pop = 80
num_parents_mating = 50
mutation_num_genes = 1

if gene_type == int:
    pygad_num_genes = total_bits
    init_range_low = 0
    init_range_high = 2
    random_mutation_min_val = 0
    random_mutation_max_val = 1
else:
    pygad_num_genes = num_genes
    init_range_low = -5
    init_range_high = 5
    random_mutation_min_val = -0.5
    random_mutation_max_val = 0.5

objective_best = []
objective_mean = []
objective_std = []
gen_results = dict()


def on_generation(ga_instance):
    generation = ga_instance.generations_completed
    logger.info(f"Generation = {generation}")

    solution, _, solution_idx = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness
    )

    if gene_type == int:
        decoded_solution = decode_individual(solution)
    else:
        decoded_solution = solution

    best_objective = -fitness_func(ga_instance, solution, solution_idx)
    logger.info(f"Best = {best_objective}")
    logger.info(f"Individual = {decoded_solution}")

    objectives = []
    for individual in ga_instance.population:
        obj_val = -fitness_func(ga_instance, individual, 0)
        objectives.append(obj_val)

    min_val = numpy.min(objectives)
    max_val = numpy.max(objectives)
    mean_val = numpy.mean(objectives)
    std_val = numpy.std(objectives)

    logger.info(f"Min = {min_val}")
    logger.info(f"Max = {max_val}")
    logger.info(f"Average = {mean_val}")
    logger.info(f"Std = {std_val}\n")

    objective_best.append(min_val)
    objective_mean.append(mean_val)
    objective_std.append(std_val)
    gen_results[generation] = {
        "best": min_val,
        "mean": mean_val,
        "std": std_val,
        "max": max_val,
        "min": min_val,
        "individual": decoded_solution,
        "value": best_objective
    }


selection_type = "tournament" # "random" or "rws", or "tournament"
crossover_type = "single_point" # "single_point" or "two_points" or "uniform"
mutation_type = "random" # "random" or "swap"

ga_instance = pygad.GA(
    num_generations=num_generations,
    sol_per_pop=sol_per_pop,
    num_parents_mating=num_parents_mating,
    num_genes=pygad_num_genes,
    fitness_func=fitness_func,
    init_range_low=init_range_low,
    init_range_high=init_range_high,
    mutation_num_genes=mutation_num_genes,
    parent_selection_type=selection_type,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    keep_elitism=1,
    K_tournament=3,
    random_mutation_min_val=random_mutation_min_val,
    random_mutation_max_val=random_mutation_max_val,
    logger=logger,
    on_generation=on_generation,
    parallel_processing=['thread', 4],
    gene_type=gene_type
)


if __name__ == "__main__":
    ga_instance.run()

    solution, solution_fitness, _ = ga_instance.best_solution()
    if gene_type == int:
        decoded_solution = decode_individual(solution)
    else:
        decoded_solution = solution

    print("Parameters of the best solution:", decoded_solution)
    print("Fitness value of the best solution =", -solution_fitness)
    print("Minimum value: ", min(func.minimum()))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(os.path.dirname(__file__), "results",
                               f"{timestamp}_tournament_single_point_random_{repr(gene_type)}")
    os.makedirs(results_dir, exist_ok=True)

    ga_instance.best_solutions_fitness = [-x for x in ga_instance.best_solutions_fitness]

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
    plt.ylabel('Standard deviation')
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
