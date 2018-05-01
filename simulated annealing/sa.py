import copy
import math
from random import *
import random
import networkx as nx


def generate_random_complete(nodes):
    '''
    :return: (g, None), g = Random complete weighted graph
    '''
    g = nx.fast_gnp_random_graph(n=nodes, p=1.0, seed=None, directed=False)
    for (u,v,d) in g.edges(data=True):
        d['weight'] = random.randint(0,100)
    return g, None


def generate_random_complete_with_solution(nodes):
    '''
    :param nodes: Number of nodes in random graph
    :return: a random weighted graph, and optimal solution expressed as list(g.edges)
    '''
    threshold = 10
    g = nx.fast_gnp_random_graph(n=nodes, p=0.0, seed=None, directed=False)

    # add minimum weight to each path
    for i in range(nodes):
        #print("adding {},{}".format(i,(i+1)%nodes))
        g.add_edge(i, (i+1)%nodes, weight = random.randint(0, threshold))

    # solution = list(g.edges())

    # add the rest of the edges with weights higher than threshold
    for i in range(nodes):
        for j in range(nodes):
            if i != j and (i,j) not in g.edges():
                #print("adding {},{}".format(i,j))
                g.add_edge(i, j, weight = threshold + random.randint(1, 100))

    return g, [i for i in range(nodes)]


def generator():
    """
    :return: randomly ordered list of integers from 0 - nodes
    """
    sample = [i for i in range(nodes)]
    random.shuffle(sample)
    return sample


def energy(dna):
    total = 0
    for i in range(len(dna)):
        u = dna[i]
        v = dna[(i+1) % nodes]
        total += graph.edges()[u,v]['weight']
    return total

nodes = 10
VISUALIZE = False
graph, solution = generate_random_complete_with_solution(nodes)


def annealing(generator, energy, T):
    '''
    :param generator: function that produces random items in the statespace
    :param energy: function that scores a given item in the statespace
    :param T: starting energy
    :return: best value
    '''
    state = generator()
    iteration = 0
    accepts = 0
    improves = 0

    # Attempt moves to new states
    while T > 0 and iteration < 1000:
        currentE = energy(state)
        print("State {}: {} | Energy: {} | T: ".format(iteration, state, currentE, T))

        new_state = generator()
        newE = energy(new_state)
        dE = newE - currentE

        if dE == 0:
            accepts += 1
            state = new_state
        elif dE < 0:
            # new state has smaller energy
            state = new_state
            accepts += 1
            improves += 1
        else:
            acceptance = math.exp((-dE)/T)
            if acceptance > uniform(0,1):
                accepts += 1
                state = new_state

        T /= math.log(1 + iteration)
        iteration += 1

    return state, energy(state)

annealing(generator,energy,1)