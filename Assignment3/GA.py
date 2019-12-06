import random
import numpy as np

class Population:
    chromosomes = []

    def __init__(self, mutation_rate):
        self.mutation_rate = mutation_rate
        self.parent1 = 0
        self.parent2 = 0
    
    def relativeFitness(self):
        
        sum1 = sum([x.distance for x in self.chromosomes])
        for i in range(len(self.chromosomes)):
            self.chromosomes[i].fitness = self.chromosomes[i].distance / sum1
        
        print("sum of fitness: ", sum([x.fitness for x in self.chromosomes]))

    def cross_select(self):
        # get new parents and make population
        temp_chromosomes = []
        # Mutate chromosomes

        # BELOW IS TEMPORARY GARBAGE WITHOUT CROSS OVER
        self.sort_chromosomes()

        for i, chromosome in enumerate(self.chromosomes):
            noi = int(round(chromosome.fitness * len(self.chromosomes)))

            for j in range(noi):
                temp_chromosomes.append(chromosome)

                if(len(temp_chromosomes) >= len(self.chromosomes)):
                    break
            
            if(len(temp_chromosomes) >= len(self.chromosomes)):
                break
        
        for chromosome in temp_chromosomes:
            if random.uniform(0, 1) < self.mutation_rate:
                chromosome.mutate()
        
        self.chromosomes = temp_chromosomes

    def create_parents(self):

        pass

    def sort_chromosomes(self):
        self.chromosomes.sort(key = lambda chromosome: chromosome.fitness)
        
    def get_best_chromosome(self):
        #print([x.fitness for x in self.chromosomes])
        fitness_list = [x.fitness for x in self.chromosomes]
        best_index = fitness_list.index(min(fitness_list))
        return self.chromosomes[best_index]

class Chromosome:
    def __init__(self, genes):
        # Sample randomly from the provided gene list
        self.sequence = random.sample(genes, len(genes))
        # Set the last element to the first element.
        self.sequence[-1] = self.sequence[0]

        self.distance = -1
        self.fitness = -1
        
    def mutate(self):
        i, j, = 0, 0
        while(i == j):
            i, j = random.randrange(1, len(self.sequence)-1), random.randrange(1, len(self.sequence)-1)
        self.sequence[i], self.sequence[j] = self.sequence[j], self.sequence[i]
        #self.sequence[j] = self.sequence[i]

    def calculate_distance(self):
        self.distance = 0
        for i in range(len(self.sequence) - 1):
            a = self.sequence[i]
            b = self.sequence[i+1]
            # Euclidian distance of [x,y] in gene
            self.distance += np.sqrt(np.power(b[0] - a[0], 2) + np.power(b[1] - a[1], 2))
