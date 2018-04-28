import random
import networkx as nx
import matplotlib.pyplot as plt
import copy

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

    solution = list(g.edges())

    # add the rest of the edges with weights higher than threshold
    for i in range(nodes):
        for j in range(nodes):
            if i != j and (i,j) not in g.edges():
                #print("adding {},{}".format(i,j))
                g.add_edge(i, j, weight = threshold + random.randint(1, 100))

    return g, solution

def mutate_swap_lambda(p, func):
    '''
    Returns a mutation function (dna, iter => dna) that swaps bases in the dna with
    probability given according to func(p, iterations) where p is some base probability
    :param p: base probability
    :param func: function that determines cutoff of probability
    :return: mutation function
    '''
    def mutate(dna, iter):
        prob = func(p, iter)
        # for each element in the dna, swap it with random probability with any other element
        for i in range(len(dna)):
            if random.random() < prob:
                swap = random.randint(0, len(dna) - 1)
                temp = dna[i]
                dna[i] = dna[swap]
                dna[swap] = temp
    return mutate

def PMX_crossover(p1, p2):
    '''
    Performs PMX crossover on two pieces and returns a list of children
    :param p1:
    :param p2:
    :return:
    '''
    start = random.randint(0, len(p1)-1)
    end = start + random.randint(0, len(p1) - start)

    # swaps from start (inclusive) to end (non-inclusive)

    swap_1 = {}
    swap_2 = {}
    child_1 = copy.deepcopy(p1)
    child_2 = copy.deepcopy(p2)

    for i in range(start, end):
        swap_1[p2[i]] = p1[i]
        swap_2[p1[i]] = p2[i]
        child_1[i] = p2[i]
        child_2[i] = p1[i]

    # performs the swaps
    for i in range(len(p1)):
        if i < start or i >= end:
            while child_1[i] in swap_1:
                child_1[i] = swap_1[p1[i]]
            while child_2[i] in swap_2:
                child_2[i] = swap_2[p2[i]]

    return child_1, child_2

def score(dna):
    total = 0
    print(graph.edges)
    for i in range(len(dna)):
        u = dna[i]
        v = dna[(i+1) % nodes]
        es = graph.edges
        e = graph.edges()[u,v]
        total += e['weight']
    return total

def generate_random_population(size):
    init_population = []
    for i in range(size):
        dna = [i for i in range(nodes)]
        random.shuffle(dna)
        init_population.append(dna)
    return init_population

nodes = 10
VISUALIZE = False
graph, solution = generate_random_complete_with_solution(nodes)
if VISUALIZE:
    pos = nx.spring_layout(graph)
    nx.draw(graph,pos)
    labels = nx.get_edge_attributes(graph,'weight')
    nx.draw_networkx_labels(graph, pos)
    nx.draw_networkx_edge_labels(graph,pos,edge_labels=labels)
    plt.show()

sim = genetics.DiscreteSimulation(
    init_population= generate_random_population(10),
    mutate=mutate_swap_lambda(0.05, lambda x,y: x),  # Mutate at a 5% rate
    crossover=PMX_crossover,
    select_breeders=genetics.tournament(2),
    elite_size=2,
    reproduction_rate=2,
    fitness_function=score,
    num_breeders=8)

def dna_stats(population):
    '''Best DNA, best score, average score'''
    best_dna = max(population, key=lambda x: score(x))
    best_score = score(best_dna)
    average_score = sum(score(member) for member in population) / len(population)

    return best_dna, best_score, average_score

while True:
    best, best_score, average_score = dna_stats(sim.population)

    print('{} | Average score: {}'.format(str(best), average_score))

    if str(best) == solution:
        break

    population = sim.step()
