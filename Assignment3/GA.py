import random
import numpy as np

class Population:
    chromosomes = []
    r = 0.5

    def __init__(self, mutation_rate):
        self.mutation_rate = mutation_rate
    
    def relativeFitness(self):
        for i in range(len(self.chromosomes)):
            self.chromosomes[i].fitness = 1 / ((self.chromosomes[i].distance) + 1)
        sum1 = sum([x.distance for x in self.chromosomes])
        for i in range(len(self.chromosomes)):
            self.chromosomes[i].fitness = self.chromosomes[i].distance / sum1
        
        #print("sum of fitness: ", sum([x.fitness for x in self.chromosomes]))


    def cross_select(self):
        new_pop = []
        #print(self.chromosomes)

        for i in range(int(len(self.chromosomes) * self.r)):
            parent1, parent2, length = self.create_parents()
            child = self.crossover(parent1, parent2)
            #print("CHILD    ", child)
            new_pop.append(child)
        #print(new_pop)
        for j in range(len(new_pop)):
            #print("child = ", new_pop[j].sequence)
            a = random.uniform(0,1)
            print(a)
            if a < self.mutation_rate:
                new_pop[j].mutate()
        print("NEW_POP  ", len(new_pop))
        self.chromosomes[len(self.chromosomes) - length - 1:-1] = new_pop

    def crossover(self, parent1, parent2):
        child = Chromosome([[-1, -1] for x in range(self.chromosomes[0].size)])
        #print("SEQUENCE:    ", child.sequence)
        #print("PARENT SIZE  ", parent1.size)
        #print(parent1.sequence)
        startPos = random.randrange(1, parent1.size - 1, 1)
        endPos = random.randrange(1, parent1.size - 1, 1)

        for i in range(parent1.size - 1):
            if startPos < endPos and i > startPos and i < endPos:
                child.sequence[i] = parent1.sequence[i]
            elif startPos > endPos:
                if not (i < startPos and i > endPos):
                    child.sequence[i] = parent1.sequence[i]

        for j in range(parent2.size - 1):
            if not (parent2.sequence[j] in child.sequence):
                for k in range(parent2.size - 1):
                    if child.sequence[k] == [-1, -1]:
                        child.sequence[k] = parent2.sequence[j]
                        break

        child.sequence[-1] = child.sequence[0]
        return child

    def create_parents(self):
        temp_chromosomes = []
        self.sort_chromosomes()
        length = int(len(self.chromosomes) * self.r)
        for i, chromosome in enumerate(self.chromosomes[0:length]):
            noi = int(round(chromosome.fitness * len(self.chromosomes)))

            for j in range(noi):
                temp_chromosomes.append(chromosome)

                if(len(temp_chromosomes) >= length):
                    break
            
            if(len(temp_chromosomes) >= length):
                break

        i = random.randrange(0, len(temp_chromosomes), 1)
        j = random.randrange(0, len(temp_chromosomes), 1)

        return temp_chromosomes[i], temp_chromosomes[j], length

    def sort_chromosomes(self):
        self.chromosomes.sort(key = lambda chromosome: chromosome.fitness, reverse=True)
        
    def get_best_chromosome(self):
        #print([x.fitness for x in self.chromosomes])
        fitness_list = [x.fitness for x in self.chromosomes]
        best_index = fitness_list.index(min(fitness_list))
        return self.chromosomes[best_index]

class Chromosome:
    def __init__(self, genes):
        # Sample randomly from the provided gene list
        self.sequence = genes
        # Set the last element to the first element.
        #self.sequence[-1] = self.sequence[0]
        self.size = len(genes)
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
