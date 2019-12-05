import numpy as np
import sys
import os
import random
import time
import matplotlib.pyplot as plt
import statistics as stat
import math

mutation_rate = 0.3
population_size = 200        # Number of orders to work with
population = []             # List of orders
order = []                  # Order of cities
fitness = []                # List of Fitness values to corresponding order
recordDistance = 100000     # Lowest current order distance found
bestEver = []               # Best current order             

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

    #locations = locations[0:20]
    # Every location has 3 attributes, split them by " "
    for i, location in enumerate(locations):
        location_att.append(location.split(' '))

    locations = [location[1:3] for location in location_att]
    locations = [[float(j) for j in i] for i in locations]

    #print(locations)
    return locations

# Calculate the distance between two cities.
# a, b = indices of cities
def calc_distance(a, b):
    return np.sqrt(np.power(b[0] - a[0], 2) + np.power(b[1] - a[1], 2))

# Calculate the total distance of a path.
def calc_sum_distance(locations, order1):
    distance = 0
    #print("order: ", order)
    for i in range(0, len(order1)-1):
        #print(locations)
        #print(order1[i])
        a = int(order1[i])
        b = int(order1[i+1])
        distance += calc_distance(locations[a], locations[b])
    return distance

# Normalize the fitness values so that they represent the fraction of the sum of all fitnesses.
def norm_fitness(temp_fitness):
    
    sum1 = sum(temp_fitness)
    for i in range(len(temp_fitness)):
        temp_fitness[i] = (temp_fitness[i] / sum1)
    print(temp_fitness)
    return temp_fitness
    '''
    mi = min(temp_fitness)
    ma = max(temp_fitness)

    for i in range(len(temp_fitness)):
        temp_fitness[i] = (temp_fitness[i] - ma)/(mi - ma)

    print(sum(temp_fitness))
    return temp_fitness
    '''
    
# Calculate the fitness of each order based on the distance of that orders path.
def calc_fitness(pop, k):
    global recordDistance, bestEver
    temp_fitness = []
    #print("calc_fitness")
    for i in range(population_size):
        #print(population[i])
        #time.sleep(1)
        distance = calc_sum_distance(locations, pop[i])
        print(distance)
        if distance < recordDistance:
            plt.clf()
            recordDistance = distance
            bestEver = pop[i]
            #print(recordDistance, "  ", sum(fitness))
            #print(fitness)
    
            listx = []
            listy = []
            for j in range(0, len(bestEver)):
                listx.append(locations[bestEver[j]][0])
                listy.append(locations[bestEver[j]][1])
            plt.plot(listx, listy, 'r-')
            plt.plot(listx, listy, 'b.')
            plt.plot(locations[bestEver[0]][0], locations[bestEver[0]][1], 'go')
            #plt.plot(locations[bestEver[len(bestEver) - 1]][0], locations[bestEver[len(bestEver - 1)]][1], 'go')
            plt.title(str(recordDistance) + "   " + str(k))
            plt.draw()
            plt.pause(0.001)
        temp_fitness.append(distance)
    return norm_fitness(temp_fitness)

def swap(candidate):
    idxA, idxB = 0, 0
    while(idxA == idxB):
        idxA, idxB = random.randrange(1, len(candidate)-1), random.randrange(1, len(candidate)-1)
    #print("BEFORE:	", candidate[idxA],"    ", candidate[idxB])
    candidate[idxA], candidate[idxB] = candidate[idxB], candidate[idxA]
    #print("AFTER:	", candidate[idxA],"    ", candidate[idxB])
    
    return candidate

# Swap elements of the orders in the population.
def mutate(pop):
    global mutation_rate
    for candidate in pop:
        #nos = random.randrange(1, 2)
        #print(nos)
        if random.uniform(0,1) < mutation_rate:
            candidate = swap(candidate)
    return pop

def get_fittest_id(pop):
    fittest_val = 100000 
    fittest_id = 0
    fitness = calc_fitness(pop, 0)
    for i in range(len(fitness)):
        if fitness[i] < fittest_val:
            fittest_val = fitness[i]
            fittest_id = i
    return fittest_id

def select(pop):
    selection = []
    #create new pop biassed towards higher fitness candidates
    #for i in range(population_size):
    #    selection.append(pop[int(random.uniform(0,1) * population_size)])
    for i, candidate in enumerate(pop):
            #print(ma,"   ", mi, "   ", can_fit)
            noi = int(round(fitness[i] * population_size))
            #noi += (pop_fitness[i] / sum(pop_fitness)) * population_size) 
            #print(noi)
            for j in range(0, noi): 
                selection.append(candidate)

                if(len(selection) >= population_size):
                    return mutate(selection)
    return mutate(selection)
    #remove this garbage below
    #fittest = selection[get_fittest_id(selection)]

    #return selection[int(random.uniform(0,1) * population_size)]


def crossover(parent1, parent2):
    child = [-1 for k in range(len(parent1))]

    startPos = int(random.uniform(0,1) * len(parent1))
    endPos = int(random.uniform(0,1) * len(parent1))

    for i in range(len(parent1)):
        if startPos < endPos and i > startPos and i < endPos:
            child[i] = parent1[i]
        elif startPos > endPos:
            if not (i < startPos and i > endPos):
                #print(i)
                child[i] = parent1[i]

    for j in range(len(parent2)):
        if not (parent2[j] in child):
            for k in range(len(parent2)):
                if child[k] == -1:
                    child[k] = parent2[j]
                    break

    child[-1] = child[0]

    return child

# Create a new population based of the fitness values of the last one.
def create_pop(pop):
    new_pop = []    
    
    for i in range(population_size):
        parent1 = []
        parent2 = []
        child = []

        parent1 = select(pop)
        #print(parent1)
        #exit()
        parent2 = select(pop)

        #if parent1 == parent2:
        #    print("GORA")

        child = crossover(parent1, parent2)

        new_pop.append(child)

        '''
        index = 0
        r = random.uniform(0,1)
        #print("r:   ", r)
        while(r > 0):
            r = r - fitness[index]
            index = index + 1

        index = index -1
        if index >= population_size - 1:
            index = random.randrange(1, population_size)
        #print("index:   ", index)
        new_pop.append(population[index])
        #print(population[i])
        #print(index)
        '''
    #return new_pop
    new_pop = mutate(new_pop)
    return new_pop

# Parse file.
locations = parse_file()

# Generate an order array [0,1,2,...,n] which defines the path of the salesman.
for i in range(len(locations)):
    order.append(i)
#print(order)
#exit()

# Generate a list of paths/orders in a random order by taking random samples of the order array above, at last append the first element to the lists.
for i in range(population_size):
    population.append(random.sample(order, len(order)))
    population[i].append(population[i][0])
#print(population)
#print("")

plt.show()
plt.ion()

# Calculate fitness and update the population with a better one.
for i in range(0, 1000):
    #fitness.clear()
    fitness = calc_fitness(population, i)
    #print(fitness)
    #fitness = norm_fitness()
    population = create_pop(population)
    print(len(population), "    mean:   ", stat.mean(fitness))
    #print(fitness)
    #print(population)
    #print("bestEver: ", bestEver)
    #print("recordDistance: ", recordDistance,"  ", i)

print(population)
