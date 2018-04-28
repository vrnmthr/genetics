from collections.abc import Sequence

def pairwise(iterable):
    '''
    Given an iterable, yield the items in it in pairs. For instance: list(pairwise([1,2,3,4])) == [(1,2), (3,4)]
    '''
    x = iter(iterable)
    return zip(x, x)


class DiscreteSimulation:
    def __init__(self, init_population, mutate, crossover,
                 select_breeders, elite_size, fitness_function, num_breeders, reproduction_rate):

        self.generation = 0
        self.population = init_population
        self.population_size = len(init_population)
        # func(DNA,...) = DNA; performs randomized mutation on a single piece and returns it
        self.mutate = mutate
        # func(p1,p2) = list[DNA]; performs crossover and returns any number of children
        self.crossover = crossover
        # number of breeders to select for the next generation
        self.num_breeders = num_breeders
        # func(list[tuple[score,population_member]], number) = list[member]
        # selects number individuals to breed from the list
        self.select_breeders = select_breeders
        # number of children produced by every *couple*
        self.reproduction_rate = reproduction_rate
        # number that remains exactly the same from one generation to the next
        self.elite_size = elite_size
        # def(member) = num :: scoring function for a given member of population
        self.fitness_function = fitness_function

        if elite_size + num_breeders/2*reproduction_rate != self.population_size:
            raise ValueError("Population size is inconsistent with given parameters")

    def find_parents(self, scored_population):
        '''
        :param scored_population: scored_population to use to select breeders
        :return: List[members] of the population that are to be bred
        '''
        return self.select_breeders(
            scored_population,
            self.num_breeders)

    def find_scores(self, population):
        '''
        Created a scored population, which is a list of (score, member) pairs,
        from a population.
        '''
        for member in population:
            yield self.fitness_function(member), member

    def step_generator(self):
        '''
        Run a whole genetic step on a scored population, and yield the new
        population members
        '''

        #new_population = []
        # Score and sort current population
        scored_population = sorted(self.find_scores(self.population), reverse=False,
            key=lambda member: member[0])

        # take elite members and keep them for subsequent trials
        for elite in scored_population[:self.elite_size]:
            #new_population.append(elite[1])
            yield elite[1]

        # Generate parents
        couples = pairwise(self.find_parents(scored_population))
        for parent1, parent2 in couples:
            # generate number of children
            children = 0
            while children < self.reproduction_rate:
                # crossover parents
                for child in self.crossover(parent1, parent2):
                    # mutate
                    if children < self.reproduction_rate:
                        self.mutate(child, self.generation)
                        yield child
                        children += 1
                    else:
                        break

    def step(self):
        '''
        Run a genetic step on a population and return the new population as a
        list.
        '''
        self.generation += 1
        self.population = list(self.step_generator())
        return self.population

