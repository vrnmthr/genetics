import random


def mutation_rate(rate):
    '''
    :param rate: probability that a single piece will get mutated
    :return: an iterable of booleans representing whether each piece should get mutated
    '''
    def mutation_mask(length):
        return (random.random() < rate for _ in range(length))
    return mutation_mask
