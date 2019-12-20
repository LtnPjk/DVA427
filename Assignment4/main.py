import numpy as np
import sys
import os
import random
import time

graph = {}
vertecies_no = 0


def parse_file():

    # Parse training set
    # Split lines into locations
    with open(os.path.join(sys.path[0], "city.txt"), "r") as f:
        edges = f.read().splitlines()

    #remove the first unusable lines
    for i in range(0, 3):
        edges.pop(0)

    edge_att = []

    #locations = locations[0:20]
    # Every location has 3 attributes, split them by " "
    for i, edge in enumerate(edges):
        edge_att.append(edge.split(' '))

    for vertex in edge_att:
        if not vertex[0] in graph.keys():
            graph[vertex[0]] = []
        if not vertex[1] in graph.keys():
            graph[vertex[1]] = []

    for edge in edge_att:
        add_edge(edge[0], edge[1], edge[2])
        add_edge(edge[1], edge[0], edge[2])


def add_edge(v1, v2 , d):
    global graph
    graph[v1].append([v2, int(d)])

def dijkstra(s):
    global graph

    # list of searched nodes
    searched = []

    # create dictionary for every city = [minimum distance, predecessor]
    paths = {}
    for vertex in graph.keys():
        paths[vertex] = [float('Inf'), None]

    # set startnode distance to 0 to start there
    paths[s][0] = 0
    # add startnode to searched
    searched.append(s)

    # unsearched
    unsearched = paths.copy()

    # as long as we haven't searched all cities
    while len(searched) < len(graph.keys()):
        # get key of city with minimum distance
        key_min = min(unsearched.keys(), key=(lambda k: unsearched[k][0]))
        #print(paths[key_min], "key_min: ", key_min)
        # for every neighbour of the above selected node
        for edge in graph[key_min]:
            # if neighbours best distance is larger than distance from current city + current citys best distance
            if paths[edge[0]][0] > edge[1] + paths[key_min][0]:
                 paths[edge[0]][0] = edge[1] + paths[key_min][0]
                 paths[edge[0]][1] = key_min
        
        searched.append(key_min)
        del unsearched[key_min]

    #print(paths['6'][0])
    return paths

def printPath(x):
    finalPath = []
    a =str(x)
    b = x
    distance = paths[x][0]
    while True:
        finalPath.append(a)
        b = a
        a = paths[a][1]
        if a == None:
            break
    finalPath.reverse()
    print("Length:\t", distance,"\tPath from\t", b, " to ", endVertex , '\t', finalPath)    
    
#parse_file()

endVertex = '5'

graph = {
    "1":[["2", 7], ["3", 9], ["6", 14]],
    "2":[["1", 7], ["3", 10], ["4", 15]],
    "3":[["2", 10], ["1", 9], ["6", 2], ["4", 11]],
    "4":[["2", 15], ["3", 11], ["5", 6]],
    "5":[["4", 6], ["6", 9]],
    "6":[["1", 14], ["3", 2], ["5", 9]],
    }

for city in graph.keys():
    if city == endVertex:
        continue
    paths = dijkstra(city)
    printPath(endVertex)

print(graph)
