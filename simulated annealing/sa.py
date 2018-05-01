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

def generate(statespace):
    points = {}
    data = [[uniform(0,10) for _ in range(len(statespace))], [uniform(0,10) for _ in range(len(statespace))]]
    for i in range(len(statespace)):
        if statespace[i] not in points:
            points[statespace[i]] = (data[0][i],data[1][i])
    return points

statespace = [1,2,3,4,5]
state = [1,4,5,3,2,1]

points = generate(statespace)
print(points)

def distance(a, b):
    """Calculates distance between two latitude-longitude coordinates."""
    ans = (a[0] - b[0])**2 + (a[1] - b[1])**2
    return math.sqrt(ans)

def energy(points,state): #fitness function
    energy = 0
    for i in range(len(state) - 1):
        a = points[state[i]]
        b = points[state[i+1]]
        dist = distance(a,b)
        energy += dist
    return energy

def annealing(state, statespace, T):
    """Minimizes the energy of a system by simulated annealing.
    Parameters
    state : an initial arrangement of the system
    Returns
    (state, energy): the best state and energy found.
    """
    initial = T
    #print(initial,'initial')
    iteration = 0
    #print(iteration,'iteration is zero')

    #Tfactor = -1/(math.log(1/T))



    accepts = 0
    improves = 0

    new_statespace = copy.deepcopy(statespace)
    new_statespace.remove(state[0])
    m = len(new_statespace)

    # Attempt moves to new states
    while T > 0 and iteration < 1000:
        currentE = energy(points,state)

        new_state = copy.deepcopy(state)
        new_statespace = copy.deepcopy(statespace)
        new_statespace.remove(state[0])
        m = len(new_statespace)

        for i in range(1,m + 1):
            new_state[i] = random.choice(new_statespace)
            new_statespace.remove(new_state[i])


        iteration += 1
        #print(iteration,'this is first iteration')

        newE = energy(points,new_state)

        dE = newE - currentE
        #print(currentE,newE,dE)
        #print()

        if dE == 0:
            accepts += 1
            state = new_state
        elif dE < 0:
            state = new_state
            accepts += 1
            improves += 1
        else:
            #print(dE)
            acceptance = math.exp((-dE)/T)
            if acceptance > uniform(0,1):
                #print(acceptance)
                #print()
                accepts += 1
                state = new_state
            #print()
        #print(new_state,newE)

        #print(state)
        Tfactor = initial/(math.log(1+iteration))
        T = Tfactor
        #print(iteration,'this is iteration')
        #print(T,'This is time')

    print(accepts,improves)
    print(state, currentE)
    return state, currentE

annealing(state,statespace,5)