import numpy as np
import sys
import os
import random
import time

start_location = []

def parse_file():

    global start_location
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

    # Every location has 3 attributes, split them by " "
    for i, location in enumerate(locations):
        location_att.append(location.split(' '))

    locations = [location[1:3] for location in location_att]
    locations = [[float(j) for j in i] for i in locations]
   
    start_location = locations[0]
    locations.pop(0)

    #print(locations)
    return locations

# a, b = indices of cities
def calc_distance(a, b):
    return np.sqrt(np.power(b[0] - a[0], 2) + np.power(b[1] - a[1], 2))

def eval_pop(pop):
    pop_fitness = []

    for x in pop:
        fitness = 0
        for i in range(0, len(x)-1):
            #print(x[i])
            fitness += calc_distance(x[i], x[i+1])
        pop_fitness.append(fitness)

    sum_pop_fitness = sum(pop_fitness)
    for i in range(0, len(pop_fitness)):
        pop_fitness[i] = pop_fitness[i] / sum_pop_fitness

    #print("sum: ", sum(pop_fitness))
    return pop_fitness

def swap(candidate):
    idxA, idxB = 0, 0
    while(idxA == idxB):
        idxA, idxB = random.randrange(1, len(candidate)-1), random.randrange(1, len(candidate)-1)
    #print("BEFORE:	", candidate[idxA],"    ", candidate[idxB])
    candidate[idxA], candidate[idxB] = candidate[idxB], candidate[idxA]
    #print("AFTER:	", candidate[idxA],"    ", candidate[idxB])
    
    return candidate

def mutate(pop):
    for candidate in pop:
        nos = random.randrange(10, 30)
        for i in range(0, nos):

            
            fitness = 0
            for j in range(0, len(candidate)-1):
                fitness += calc_distance(candidate[i], candidate[i+1])

            #print(fitness)
            fitness = 0
            for j in range(0, len(candidate)-1):
                fitness += calc_distance(candidate[i], candidate[i+1])
            #print(fitness)
            
            candidate = swap(candidate)

    return pop

def create_pop(pop, pop_fitness):
    noi = 0
    new_pop = []    
    ma = max(pop_fitness)
    mi = min(pop_fitness)
    ''' 
    for i, candidate in enumerate(pop):
        can_fit = pop_fitness[i]

        #print(ma,"   ", mi, "   ", can_fit)
        if mi - ma != 0:
            noi = int(round(((can_fit - ma) / (mi - ma)) * population_size))
            #noi += (pop_fitness[i] / sum(pop_fitness)) * population_size) 
            print(noi)
        else:
            noi = 1 
            print("else")
            
        for j in range(0, noi): 
            new_pop.append(candidate)

            if(len(new_pop) >= population_size):
                return mutate(new_pop)
    '''
    
    for i in range(len(pop)):
        index = 0
        r = random.uniform(0,1)
        #print("r:   ", r)
        while(r > 0):
            r = r - pop_fitness[index]
            index = index + 1

        index = index -1
        new_pop.append(pop[index])
        #print(index)
    #return new_pop
    return mutate(new_pop)

locations = parse_file()
#print(calc_distance(1,1))
#print(len(locations))
#print(locations)
population_size = 10
population = []

for i in range(0, population_size):
    candidate = [x for x in locations]
    random.shuffle(candidate)

    candidate.insert(0, start_location)
    candidate.insert(len(candidate), start_location)
    population.append(candidate)

#for i, candidate in enumerate(population):
    #print(i, "  ", candidate)
#print(len(population[0]))
#print(population)
for i in range(0, 1000):
    #print(population[0])
    #print(len(population))
    pop_fitness = eval_pop(population)
    #print(pop_fitness)
    #print(max(pop_fitness))
    population = create_pop(population, pop_fitness)
    #print("")
    #print(population[0])
    print("Generation:  ",i,"   ", min(pop_fitness), "  ", max(pop_fitness))

#print(population)
pop_fitness = eval_pop(population)
#print(pop_fitness)
#print(len(locations))
#print(start_location)
#print(population[0][0], population[0][-1])
