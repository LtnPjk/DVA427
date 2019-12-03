import numpy as np
import sys
import os

def parse_file():

    # Parse training set
    # Split lines into patients
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
    locations = np.array([[float(j) for j in i] for i in locations])
   
    print(locations)
    return locations

# a, b = indexes of cities
def calc_distance(a, b):
    return np.sqrt(np.power(locations[b][0] - locations[a][0], 2) + np.power(locations[b][1] - locations[a][1], 2))

locations = parse_file()
print(locations[1])
#print(calc_distance(1,1))

