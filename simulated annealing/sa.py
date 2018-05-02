import copy
import math
from random import *
import random
import networkx as nx
import matplotlib.pyplot as plt


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


def get_neighbor(soln):
    '''
    :param soln: item to get the neighbor of
    :return: neighbor of this item
    '''
    # nbor = copy.deepcopy(soln)
    #
    # i = random.randint(0, len(soln) - 1)
    # j = i + random.randint(0, len(soln) - i - 1)
    # temp = nbor[i]
    # nbor[i] = nbor[j]
    # nbor[j] = temp
    # return nbor

    # reverses from start (inclusive) to end (non-inclusive)
    start = random.randint(0, len(soln) - 1)
    end = start + random.randint(0, len(soln) - start - 1)

    nbor = copy.deepcopy(soln)
    while start < end:
        temp = nbor[start]
        nbor[start] = nbor[end]
        nbor[end] = temp
        start += 1
        end -= 1
    return nbor


def annealing(state, generator, energy, init):
    '''
    :param generator: function that produces random items in the statespace
    :param energy: function that scores a given item in the statespace
    :param init: starting energy
    :return: best value
    '''
    T = init
    iteration = 1
    accepts = 0
    improves = 0

    # Attempt moves to new states
    while T > 0 and iteration < T_ITERS:
        currentE = energy(state)
        solutions.append(currentE)

        if not VERBOSE:
            if iteration % FREQ == 0:
                print("State {}: {} | Energy: {} | T: {}".format(iteration, state, currentE, T))
        else:
            print("State {}: {} | Energy: {} | T: {}".format(iteration, state, currentE, T))

        new_state = generator(state)
        newE = energy(new_state)
        dE = newE - currentE

        if dE == 0:
            state = new_state
        elif dE < 0:
            # new state has smaller energy
            state = new_state
        else:
            acceptance = math.exp((-dE)/T)
            if acceptance > uniform(0,1):
                state = new_state

        T = init/math.log(1 + iteration)
        iteration += 1

    return state, energy(state)

nodes = 20
graph, solution = generate_random_complete_with_solution(nodes)
solutionE = energy(solution)
VERBOSE = False
FREQ = 1000
T_ITERS = 7500
solutions = []
approximation, approxE = annealing(state=generator(), generator=get_neighbor, energy=energy, init=45)
print("Sol energy: {} | Approx. energy: {} | % diff: {}".format(solutionE, approxE, float(approxE-solutionE)/solutionE*100))
plt.plot(solutions)
plt.show()

