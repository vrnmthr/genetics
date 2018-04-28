import random
import bisect
import itertools


def tournament(tournament_size):
    '''
    :param tournament_size: Number of members it compares at a given time
    :return: a function that given a population and a number returns the best
    number of members of the population (using the max function)
    '''
    def tournament_selector(population, num_parents):
        for _ in range(num_parents):
            sample = random.sample(population, tournament_size)
            best = max(sample, key= lambda x:x[0])
            # return a single member, not the entire tuple
            yield best[1]
    return tournament_selector


def roulette(population, num_parents):
    #Uses code taken from
    #http://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice

    cumulative_scores = list(
        itertools.accumulate(member.score for member in population))
    total = cumulative_scores[-1]

    for _ in range(num_parents):
        yield population[
            bisect.bisect(cumulative_scores, random.uniform(0, total))]


def stochastic(population, num_parents):
    cumulative_scores, total = _accumulate_scores(population)
    average = total / num_parents

    rand = random.random() * average

    for float_index, in itertools.islice(itertools.count(rand, average), num_parents):
        yield population[bisect.bisect(cumulative_scores, float_index)]
