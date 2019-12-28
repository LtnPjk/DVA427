import os, time, sys, numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt

# A location with x and y values
class Chromosome:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # Calculate distance between two chromosomes
    def distance(self, chromosome):
        xDis = abs(self.x - chromosome.x)
        yDis = abs(self.y - chromosome.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

# Fitness value of a route
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    # Calculate distance of a route 
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromChromosome = self.route[i]
                toChromosome = None
                if i + 1 < len(self.route):
                    toChromosome = self.route[i + 1]
                else:
                    toChromosome = self.route[0]
                pathDistance += fromChromosome.distance(toChromosome)
            self.distance = pathDistance
        return self.distance
    
    # Translate the distance to a fitness
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

def createRoute(chromosomeList):
    route = random.sample(chromosomeList, len(chromosomeList))
    return route

def initialPopulation(popSize, chromosomeList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(chromosomeList))
    return population

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children


def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            chromosome1 = individual[swapped]
            chromosome2 = individual[swapWith]
            
            individual[swapped] = chromosome2
            individual[swapWith] = chromosome1
    return individual

def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

def drawIndividual(individual):

    listx = [x.x for x in individual]
    listy = [y.y for y in individual]
    listx.append(listx[0])
    listy.append(listy[0])
    
    plt.subplot(1, 2, 1)
    plt.plot(listx, listy, 'r-')
    plt.plot(listx, listy, 'b.')
    plt.pause(0.01)
    plt.draw()

def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    
    for individual in pop:
        if Fitness(individual).routeDistance() == progress[-1]:
            drawIndividual(individual)
            break
    print("Last Fitness: ", progress[-1])
    elapsed_time = time.time() - start_time
    print("Time of execution: ", elapsed_time)

    #print(Fitness(pop[0]).routeDistance())
    #print(progress[-1])
    #print(pop[0])
    
    plt.subplot(1, 2, 2)
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.pause(0.01)
    plt.draw()
    plt.show()
    
    
def parse_file():

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


locations = parse_file()
#print(locations)

chromosomeList = [Chromosome(x[0], x[1]) for x in locations]

#print(chromosomeList)
'''
for i in range(0,25):
    chromosomeList.append(Chromosome(x=int(random.random() * 200), y=int(random.random() * 200)))
'''

start_time = time.time()
geneticAlgorithmPlot(population=chromosomeList, popSize=100, eliteSize=20, mutationRate=0.0010, generations=5)


