import random
import networkx as nx

import sys
sys.path.append("../")
import genetics

# traveling salesman is a graph problem
# generate a complete graph

# TODO: maybe generate a sparse graph later and compare performance?

def generate_random_complete(nodes):
    '''
    :return: (g, None), g = Random complete weighted graph
    '''
    g = nx.fast_gnp_random_graph(n=nodes, p=1.0, seed=None, directed=False)
    for (u,v) in g.edges():
        g.edge[u][v]['weight'] = random.randint(0,100)
    return g, None

def generate_random_complete_with_solution(nodes):
    '''
    :param nodes: Number of nodes in random graph
    :return: a random weighted graph, and optimal solution expressed as list(g.edges)
    '''
    threshold = 10
    g = nx.Graph()
    for i in range(nodes):
        g.add_node(i)

    # add minimum weight to each path
    for i in range(nodes):
        g.add_edge(i, i+1, weight = random.randint(0, threshold))

    solution = list(g.edges())

    # add the rest of the edges with weights higher than threshold
    for i in range(nodes):
        for j in range(nodes):
            if j != i + 1:
                g.add_edge(i, j, weight = threshold + random.randint(1, 100))

    return g, solution

num_cities = 100

# represents a single value in the DNA
# this could be a bit, in our case a city, etc. Must declare a mutate_value function
# that specifies how this value would be randomized
class LetterComponent(genetics.DNAComponent):
    def mutate_value(self):
        return random.choice(letters)


