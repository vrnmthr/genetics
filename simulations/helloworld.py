import string
import random

import genetics

# letters is a set that contains all values in the state space of string characters
letters = string.ascii_uppercase + string.ascii_lowercase + string.punctuation + ' '
for i in range(10):
    letters += str(i)

# optimal solution
solution = 'ankvag69'

# any subclass of DNAComponent represents a single value in the DNA
# this could be a bit, in our case a city, etc. Must declare a mutate_value function
# that specifies how this value would be randomized
class LetterComponent(genetics.DNAComponent):
    def mutate_value(self):
        return random.choice(letters)


# represents an entire solution
class WordDNA(genetics.arrayed_segment(len(solution), LetterComponent)):
    # scores based on number of matching elements
    def score(self):
        return sum(comp.value == letter for comp, letter in zip(self, solution))

    # toString method
    def __str__(self):
        return ''.join(comp.value for comp in self)


pop_size = 100
# random selection occurs via mutate_value in LetterComponent
init_population = [WordDNA() for _ in range(pop_size)]

sim = genetics.DiscreteSimulation(
    init_population=init_population,
    # func(length) = list of booleans determining mutation
    mutation_mask=genetics.mutation_rate(0.05),  # Mutate at a 5% rate
    # func(length) = list of booleans determining which elements should be taken, etc.
    crossover_mask=genetics.two_point_crossover,
    select_breeders=genetics.tournament(2),
    elite_size=2,
    reproduction_rate=2,
    fitness_function=WordDNA.score,
    num_breeders=98)

def dna_stats(population):
    '''Best DNA, best score, average score'''
    best_dna = max(population, key=lambda x: x.score())
    best_score = best_dna.score()
    average_score = sum(member.score() for member in population) / len(population)

    return best_dna, best_score, average_score


while True:
    best, best_score, average_score = dna_stats(sim.population)

    print('{} | Average score: {}'.format(str(best), average_score))

    if str(best) == solution:
        break

    population = sim.step()