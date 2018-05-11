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
    sample = [i for i in range(NODES)]
    random.shuffle(sample)
    return sample


def energy(graph):

    def graph_energy(dna):
        total = 0
        for i in range(len(dna)):
            u = dna[i]
            v = dna[(i+1) % NODES]
            total += graph.edges()[u,v]['weight']
        return total

    return graph_energy


def get_neighbor(soln):
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
    :return: list of energies of solutions
    '''
    T = init
    iteration = 1
    accepts = 0
    improves = 0
    solutions = []

    # Attempt moves to new states
    while T > 0 and iteration < T_ITERS:
        currentE = energy(state)

        if iteration % SOLUTION_LOGGING_FREQ == 0:
            solutions.append(currentE)

        # if not VERBOSE:
        #     if iteration % FREQ == 0:
        #         print("State {}: {} | Energy: {} | T: {}".format(iteration, state, currentE, T))
        # else:
        #     print("State {}: {} | Energy: {} | T: {}".format(iteration, state, currentE, T))

        new_state = get_neighbor(state)
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

    return solutions

NODES = 50
T_ITERS = 15000
SOLUTION_LOGGING_FREQ = 1
REPEATS = 5000

graph, sol = generate_random_complete_with_solution(NODES)
fitness = energy(graph)
solutionE = fitness(sol)
solutions = annealing(generator(), get_neighbor, fitness, 45)
# print("Sol energy: {} | Approx. energy: {} | % diff: {}".format(solutionE, approxE, float(approxE-solutionE)/solutionE*100))
print("Solution energy: {}".format(solutionE))
plt.plot(solutions)
plt.show()

def test_init_temperature():
    vals = [5*i for i in range(4,15)]
    # print("Format: solution energy, energy at step i, energy at step 2i...")
    for v in vals:
        print("computing values for {}".format(v))
        f = open("./results/{}/{}raw.txt".format(NODES, v), "w+")
        # f.write("BASE TEMPERATURE:{}\nNODES:{}\nT_ITERS:{}\nLOGGING FREQUENCY:{}\nREPEATS:{}\n".format(
        #     v, NODES, T_ITERS, SOLUTION_LOGGING_FREQ, REPEATS))
        for _ in range(REPEATS):
            graph, sol = generate_random_complete_with_solution(NODES)
            fitness = energy(graph)
            solutionE = fitness(sol)
            f.write("{},".format(solutionE))
            solutions = annealing(generator(), get_neighbor, fitness, v)
            for e in solutions:
                f.write("{},".format(e))
            f.write("\n")
        f.close()


# test_init_temperature()