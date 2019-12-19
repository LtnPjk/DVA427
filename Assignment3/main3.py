import numpy as np
import sys
import os
import random
import time
import matplotlib.pyplot as plt
import statistics as stat
import math
from GA import Population, Chromosome

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
    plt.title(str(Chromosome.fitness))
    plt.draw()
    plt.pause(0.001)

# How many chromosomes in each generation
population_size = 10
# Probability of mutation
mutation_rate = 0.02
# Number of iterations
NOI = 1000

# Parse file.
locations = parse_file()

bestEver = Chromosome(locations)

# Initialize Population
pop = Population(mutation_rate)
for i in range(population_size):
    pop.chromosomes.append(Chromosome(locations[0:15]))

# Main loop for the iterations
for i in range(NOI):
    # Calculate fitness of population
    for Chromosome in pop.chromosomes:
        Chromosome.calculate_distance()

    #print([x.distance for x in pop.chromosomes])
    pop.relativeFitness()
    #print(i, '  ', len(pop.chromosomes[0].sequence))
    print([x.fitness for x in pop.chromosomes])
    print(pop.get_best_chromosome().fitness)
    best_chromosome = pop.get_best_chromosome()
    if(best_chromosome.fitness > bestEver.fitness):
        draw_chromosome(best_chromosome)
        bestEver = best_chromosome
    print(i, '  ', len(pop.get_best_chromosome().sequence))

    # Create new population
    pop.cross_select()
