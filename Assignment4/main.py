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

def dijkstra():
    global graph

    paths = {}
    for vertex in graph.keys():
        paths[vertex] = [[], int('Inf')]
    
    for vertex in graph.keys():
        paths[vertex] = 
        for edge in vertex:
            if edge[1] < paths[edge[0]][1]:



parse_file()
print(graph)
