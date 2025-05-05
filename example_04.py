#źródło przykładu: https://pypi.org/project/pygad/#description

import pygad

def fitness_func(ga_instance, solution, solution_idx):
    return 1

fitness_function = fitness_func

def on_start(ga_instance):
    print("on_start()")

def on_fitness(ga_instance, population_fitness):
    print("on_fitness()")

def on_parents(ga_instance, selected_parents):
    print("on_parents()")

def on_crossover(ga_instance, offspring_crossover):
    print("on_crossover()")

def on_mutation(ga_instance, offspring_mutation):
    print("on_mutation()")

def on_generation(ga_instance):
    print("on_generation()")

def on_stop(ga_instance, last_population_fitness):
    print("on_stop()")

ga_instance = pygad.GA(num_generations=3,
                       num_parents_mating=5,
                       fitness_func=fitness_function,
                       sol_per_pop=10,
                       num_genes=3,
                       on_start=on_start,
                       on_fitness=on_fitness,
                       on_parents=on_parents,
                       on_crossover=on_crossover,
                       on_mutation=on_mutation,
                       on_generation=on_generation,
                       on_stop=on_stop,
                       gene_space = [{'low': 1, 'high': 2}, {'low': 3, 'high': 4}, {'low': 5, 'high': 6}])

ga_instance.run()

# w przypadku dyskretnego przedziału wykorzystajmy:
# gene_space = [[0.4, 12, -5, 21.2],
#               [-2, 0.3],
#               [1.2, 63.2, 7.4]]