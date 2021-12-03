import random
from operator import itemgetter
from multiprocessing import Pool

class GeneticAlgo(object):
    def __init__(self, gens, pop_size, mutation_rate, sim_fun):
        """Arguments: number of generations to run in a call to run(),
        how many individuals in each generation,
        rate of mutation as a decimal (e.g., 0.07),
        simulation function.
        The simulation function takes as arguments the individual genes
        and should return a fitness score.
        Simulation function is passed in as an argument instead of overridden
        by a subclass method to simplify life with multiprocessing and reduce
        overhead of copying entire self with each task passed to worker pool."""
        self.generations = gens
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.pop = []
        self.fitness = []
        self.sim_fun = sim_fun
        self.pool = Pool()

    def gen_individual(self):
        """Generates a random list of genes.
        Needs to be implemented by subclass.
        Number of genes specified by this function will be number of
        arguments to the sim_fun specified in __init__"""
        raise NotImplementedError()

    def crossover(self, parent1, parent2):
        """Crosses over two given parents to form a new child."""
        child = []
        for pool in zip(parent1, parent2):
            gene = random.choice(pool)
            child.append(gene)
        return child

    def mutate(self, chromosome, chance):
        """Mutates the genes of an individual with pre-set chance."""
        source = self.gen_individual()
        mutated = []
        for pool in zip(chromosome, source):
            if random.random() < chance:
                mutated.append(pool[1])
            else:
                mutated.append(pool[0])
        return mutated


    def init_pop(self):
        """Initializes the population with random individuals."""
        for i in range(self.pop_size):
            self.pop.append(self.gen_individual())

    def run(self):
        """Convenience method, initializes the populations and runs through
        pre-set number of generations."""
        self.init_pop()
        for i in range(self.generations):
            self.run_single_gen()

    def run_single_gen(self):
        """Runs a single generation. Calculates the fitness of the current
        population, selects the top 2 individuals, crosses over their genes to
        form new individuals, applies mutations to those individuals, 
        and builds a new population.
        Assumes that there is a population already initialized.
        Both parents are included in the new population to prevent regression of
         fitness.
        Multiprocessing version of the map statement shows much greater benefits
        for longer-running simulations than for shorter ones due to overhead of
        adding a new task to the pool/returning the result from the pool."""

        res = self.pool.map(self.sim_fun, self.pop)
        self.fitness = list(zip(res, self.pop))

        self.fitness.sort(key=itemgetter(0))
        parent1 = self.fitness[0][1]
        parent2 = self.fitness[1][1]

        new_pop = [parent1, parent2]
        for i in range(self.pop_size - 2):
            child = self.mutate(self.crossover(parent1, parent2), self.mutation_rate)
            new_pop.append(child)
        self.pop = new_pop
        new_pop = None

