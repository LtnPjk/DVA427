import numpy as np
import sys
import os
import random
import time
import matplotlib.pyplot as plt
import statistics as stat
import math
from GA import Population, Chromosome
import GA

plt.show()
plt.ion()

def parse_file():

    # Parse training set
    # Split lines into locations
    with open(os.path.join(sys.path[0], "berlin52.tsp"), "r") as f:
        locations = f.read().splitlines()

    #remove the first unusable lines
    for i in range(0, 6):
        locations.pop(0)
        
    location_att = []
    locations.pop(-1)
    locations.pop(-1)

    #locations = locations[0:20]
    # Every location has 3 attributes, split them by " "
    for i, location in enumerate(locations):
        location_att.append(location.split(' '))

    locations = [location[1:3] for location in location_att]
    locations = [[float(j) for j in i] for i in locations]

    #print(locations)
    return locations

def draw_chromosome(Chromosome):
    plt.clf()

    #recordDistance = distance
    #bestEver = pop[i]
    #print(recordDistance, "  ", sum(fitness))
    #print(fitness)

    listx = [x[0] for x in Chromosome.sequence]
    listy = [y[1] for y in Chromosome.sequence]
    
    '''for j in range(0, len(Chromosome.sequence):
        listx.append(locations[bestEver[j]][0])
        listy.append(locations[bestEver[j]][1])
    '''
    plt.plot(listx, listy, 'r-')
    plt.plot(listx, listy, 'b.')
    #plt.plot(locations[bestEver[0]][0], locations[bestEver[0]][1], 'go')
    #plt.plot(locations[bestEver[len(bestEver) - 1]][0], locations[bestEver[len(bestEver - 1)]][1], 'go')
    plt.title(str(Chromosome.distance) + "   " + str(len(Chromosome.sequence)))
    plt.draw()
    plt.pause(0.001)

# How many chromosomes in each generation
population_size = 1000
# Probability of mutation
mutation_rate = 1.0
# Number of iterations
NOI = 1000
# Number of locations
chrom_size = 15

# Parse file.
locations = parse_file()

# Initialize Population
pop = Population(mutation_rate)
for i in range(population_size):
    chrom = Chromosome(random.sample(locations[0:chrom_size], len(locations[0:chrom_size])))
    chrom.sequence.append(chrom.sequence[0])
    pop.chromosomes.append(chrom)

# Main loop for the iterations
for i in range(NOI):
    # Calculate fitness of population
    for Chromosome in pop.chromosomes:
        Chromosome.calculate_distance()

    print([x.distance for x in pop.chromosomes])
    pop.relativeFitness()
    #print(i, '  ', len(pop.chromosomes[0].sequence))
    #print([x.fitness for x in pop.chromosomes])
    #print(pop.get_best_chromosome().fitness)
    print(i, "  LENGTH   ", len(pop.chromosomes))
    draw_chromosome(pop.get_best_chromosome())
    #print(i, '  ', len(pop.get_best_chromosome().sequence))

    # Create new population
    pop.cross_select()
