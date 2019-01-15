"""
This module provides multiple functions for Genetic Algorithms such as
combination or mutation functions
"""

import random
import numpy as np
import matplotlib.pyplot as plt




# TODO population_size has to be a multiple of 4
def minimal_weight_matching(X, Y, population_size= 12, nr_iterations=50,
    save_best_nr=1, setup_replacement = 'delete-all', norm_order=2, m_rate=0.05):
    """
    Function that performance minimal weigth perfect matching between
    two set of points of length n. Done by a genetic algorithm.

    Args:
        X: set of X points
        Y: set of Y points
        norm: either 1 or 2 for L1 or L2 norm

    Returns:
        List, where every element is a tuple of indices of X and the Y index
        of one pair
    """
    # to save best members
    history = np.zeros((nr_iterations, save_best_nr))
    history_avg = np.zeros((nr_iterations, save_best_nr))
    best_tuple = [None, 0]
    # --------------------- Initialization
    population = initial_population(len(X), population_size)
    # todo
    stagnation_counter = 0
    for ite in range(nr_iterations):
        #print(population)
        # --------------------- Evaluation
        # get list of fitness scores for memebrs in population
        fitness_scores = evaluation(X, Y, population, norm_order)
        #print(fitness_scores)
        # get the values of the best save_best_nr members and save it in the
        # history
        maxmax = max(fitness_scores)
        if maxmax > best_tuple[1]:
            best_tuple = population[fitness_scores.index(maxmax)], maxmax
        history[ite] = sorted(fitness_scores)[-save_best_nr:]
        history_avg[ite] = np.mean(fitness_scores)
        # todo: if we dont move for 10 steps, half the mutation rate
        if history[ite] == history[ite-1]:
            stagnation_counter += 1
            if stagnation_counter == 10:
                m_rate *= 0.25
                stagnation_counter = 0
        else:
            stagnation_counter = 0
            if m_rate <= 0.001:
                m_rate = 0.05
        print("--- Iteration {} --> Best:[{}] ||| Avg:[{}] ||| Stagnation Level {} ||| MR {}".format(ite+1, history[ite], history_avg[ite], stagnation_counter, m_rate))
        # --------------------- Selection
        # select members based on roulette_wheel_selection
        #selected_indx = roulette_wheel_selection(fitness_scores, population_size//2)
        selected_indx = simple_selection(fitness_scores, population_size//2)
        # given indexes, get the selecter members (POPULATION_SIZE//2)
        selected_members = population[selected_indx]
        #print(selected_members)
        # shuffle
        np.random.shuffle(selected_members)
        # create empty array to save children in
        children = np.empty((population_size//2, population.shape[1])).astype(int)

        # --------------------- Crossover
        for i in range(0, population_size//2, 2):
            # parent one is in selected_members in row 1, parent two in
            # row 2 ...
                off1, off2 = k_point_crossover(selected_members[i], selected_members[i+1])

                # --------------------- Mutation
                # save created children in children array
                children[i], children[i+1] = \
                    mutation(off1, p_m=m_rate), mutation(off2, p_m=m_rate)


        # ---------------------- Replacement
        population = replacement(population, children, mode=setup_replacement, n=children.shape[0],
            based_on_fitness=True, fitness_old=fitness_scores, fitness_new=evaluation(X, Y, children, norm_order)).astype(int)

    # add the best to the population at place 1 [not good...]
    population[0] = best_tuple[0]

    return population, best_tuple, history, history_avg


def mutation(chrom,  p_m=0.05):
    """
    Randomly swap 2 element
    Args:
        chrom      : chromosome list
        p_m        : mutation probability
    Returns:
        chrom      : mutated chromose list
    """

    for i in range(len(chrom)):
        # generate number between 0 and 1, responsible for deciding
        # whether to appply mutation
        r = random.uniform(0, 1)

        # if r is smaller/equal the mutation probability, mutate!
        if r <= p_m:
            # generate to random indices
            idx = range(len(chrom))
            i1, i2 = random.sample(idx,2)
            # swap two alleles (randomly)
            chrom[i1], chrom[i2] = chrom[i2], chrom[i1]

    return chrom



def from_inversion_sequence(inversion_sequence):
    position = np.zeros(len(inversion_sequence))

    for i in range(len(inversion_sequence)-1, -1, -1):
        for m in range(len(inversion_sequence)):
            if position[m] >= inversion_sequence[i]:
                position[m] += 1
        position[i] = int(inversion_sequence[i])


    permutation = np.zeros(len(inversion_sequence))
    for i in range(len(inversion_sequence)):

        permutation[int(position[i])] = i

    return permutation


def to_inversion_sequence(permutation):
    # start by setting the inversion sequence of all values to zero

    inversion_sequence = np.zeros(len(permutation))

    for i in range(len(permutation)):
        for element in permutation:
            # now go through every element, until you encounter the element
            # with value i. up until there, count how many of the values
            # are greater than i. the resulting value is a measure of how much
            # i is out of order in the permutation.
            if element == i:
                break
            else:
                if element > i:
                    inversion_sequence[i] += 1

    return inversion_sequence


def k_point_crossover(chrom1, chrom2, k=1):
    """
    k Point Crossover
    Args:
        chrom1: chromosome of parent 1
        chrom2: chromosome of parent 2
        k     : number of crossovers
    Returns:
        offspring1: first offspring
        offspring2: second offspring
    """

    if k > (len(chrom1)-1) or k < 1:
        raise Exception("""[-] k was chosen to be {}, but k is only allowed to
            be chosen from the interval [1; {}]!""".format(k, len(chrom1)-1))


    # translate chromosome into inversion space, so we get feasible children
    # see http://user.ceng.metu.edu.tr/~ucoluk/research/publications/tspnew.pdf
    inverse1 = to_inversion_sequence(chrom1)
    inverse2 = to_inversion_sequence(chrom2)

    # generate empty chromosomes for offsprings
    offspring1 = np.empty((len(chrom1),))
    offspring2 = np.empty((len(chrom1),))

    # randomly generate sections for crossover, by generating the indexing,
    # at which the next section start (returns a list)
    crossover_indx = random.sample(range(1, len(inverse1)), k) # vor dem value
    # for easier iteration later, add the amount of genes to this list, since
    # one can interprete it either as the end of our last section
    # or the "start" of the next (not exisiting) section
    crossover_indx.append(len(inverse1))
    # sort them, so we can iterate properly
    crossover_indx.sort()

    # iterator for gene reference
    gene = 0

    # k+1 because, e.g. if k=1, we have two sections
    for section in range(k+1):

        # iterate through all genes in the current section
        while gene < crossover_indx[section]:
            # first section is (in pictures) not swapped and from there on
            # alternatively
            if section%2 == 1:
                offspring1[gene] = inverse2[gene]
                offspring2[gene] = inverse1[gene]
            # else do not swap
            else:
                offspring1[gene] = inverse1[gene]
                offspring2[gene] = inverse2[gene]

            # reference next gene
            gene += 1

    return from_inversion_sequence(offspring1), from_inversion_sequence(offspring2)





def evaluation(X, Y, population, norm_order):
    """
    Returns list: for every member we calculate a loss value, by calculating
    the sum of the distant vector between two points. the lower the better.
    """
    fitness = np.zeros((population.shape[0], ))

    for i, member in enumerate(population):
        for x_indx, y_indx in enumerate(member):
            # calculate the length of the vector between both points and add it
            # to our sum variable

            #fitness[i] += np.linalg.norm(X[x_indx]-Y[y_indx], ord=norm_order)
            fitness[i] += np.linalg.norm([255, 255, 255], ord=norm_order) - np.linalg.norm(X[x_indx]-Y[y_indx], ord=norm_order)
    # we want to make this a maximization problem, thus..
    #maximum = max(fitness)
    #fitness = [1/fit for fit in fitness]
    fitness = [fit for fit in fitness]
    return fitness




def initial_population(datapoints_nr, population_size):
    """
    Creates an initial population for our genetic algorithm.

    A member of our population looks like this:
    Given that we have two set of points X and Y, with each of them having
    N datapoints, a member of our population is a vector of length N, whose
    element x_i at indice i, assigns the datapoint of X with indice i the
    element of Y at indice x_i.
    """

    # row = members
    population = np.empty(shape=(population_size, datapoints_nr))

    for member in range(population_size):
        # each member consist of the indices from 0 to datapoints_nr-1 and is
        # some random assignment between them
        population[member] = random.sample(range(datapoints_nr), datapoints_nr)

    return population.astype(int)




# ---
def simple_selection(fitness, n):
    return np.argsort(fitness)[-n:]


def roulette_wheel_selection(fitness, n):
    """
    Roulette Wheel Selection (Fitness Proportion Selection)
    Args:
        fitness: fitness scores
        n: number of members to be selected
    Returns:
        indx_list: indexes of members to be selected
    """

    # calculate standard propabilites in regard to fitness scores
    sum_of_fitness = np.sum(fitness)

    # since smaller is better, inverse it
    probabilities = [fit/sum_of_fitness for fit in fitness]

    # build cummulative probabilites
    cum_propabilites = [sum(probabilities[:i]) for i in range(1, len(probabilities)+1)]

    # list of indexes of selected members
    indx_list = []

    while len(indx_list) != n:

        # generate random number pepresenting the ball in the roulette
        r = random.uniform(0, 1)

        for indx, prob in enumerate(cum_propabilites):
            # we found the place the ball fell down
            if r <= prob:
                indx_list.append(indx)
                break

    return indx_list




def replacement(old_pop, new_pop, mode="delete-all", n=None,
    based_on_fitness=True, fitness_old=[], fitness_new=[]):
    """
    Replacement of old population through new population
    Args:
        old_pop: old population
        new_pop: new population
        mode: chosen from options: ['delete-all', 'steady-state']
            * 'delete-all': replace part of old population through new ones
            * 'steady-state': replace n members of the old population by n
                members of the new population
        n: number of members to be replaced if 'stead-state' is chosen as mode
        based_on_fitness: boolean, if chosen, you will replace the n worst
            members of the old population the n best members of the new
            population
        fitness_old: corresponding fitness values for the old population
        fitness_new: corresponding fitness values for the new population
    Returns:
        population: replaced population
    """

    if mode == "delete-all":
        if fitness_old == []:
            raise ValueError("[!] 'fitness_old' has to be filled!")

        population = np.empty(old_pop.shape)

        # take over all members of new population
        for i in range(len(new_pop)):
            population[i] = new_pop[i]

        # list of members sorted from best to worse
        sorted_members = [member for _, member in sorted(zip(fitness_old, old_pop), key=lambda pair: -pair[0])][:n]

        # fill up rest with best of old population
        for i in range(len(new_pop), len(old_pop)):
            population[i] = sorted_members[i-len(new_pop)]

        return population

    # if not, check if n was supplied
    if n is None:
        raise Exception("[-] Please supply n when using steady-state modes!")

    # generate list for the resulting population, starting as the old_pop
    population = old_pop[:]

    # here we generate lists of indx for the old population and values to
    # replace them with from the new population

    # for the "steady-state mode" ...
    if mode == "steady-state" and not based_on_fitness:
        # choose n random indexes from the old_pop
        # replace=False ensures no duplicates
        indx_list = np.random.choice(range(len(old_pop)), size=n, replace=False)
        # and n random values from new_pop
        value_list = np.random.choice(new_pop, size=n, replace=False)

    elif mode == "steady-state" and based_on_fitness:
        """
        check if fitness lsits are defined
        build two correpsonding lists of indx and values to replace
        """

        if len(fitness_old) != len(old_pop) or len(fitness_new) != len(new_pop):
            raise Exception("""[-] Both 'fitness_old' and 'fitness_new' need to be
            the same length as 'old_pop' and 'new_pop'""")
        # check if sort the right way
        # sort populations, based on their fitness score as sort key
        # (for old population we want the n worst member, thus default sorted
        # [small to big] does just fine, for the new population we want the
        # n best member, thus we include a -)
        # for old population, sort the indexes to be replace
        # for new population, sort the member to replace with
        indx_list = [indx_old for _, indx_old in \
            sorted(zip(fitness_old, range(len(old_pop))), \
            key=lambda pair: pair[0])][:n]
        value_list = [member for _, member in \
            sorted(zip(fitness_new, new_pop), \
            key=lambda pair: -pair[0])][:n]

    # now that we got our lists, simply replace them
    for indx, val in zip(indx_list, value_list):
        population[indx] = val

    return population

# ---

if __name__=="__main__":
    X = np.array([[100,200,300], [153,223,350], [2,-3,2], [12,2,4],[5,2,5], [-1,2,4], [100,200,300], [2,-3,2], [1,2,4],[5,2,5], [-1,2,4], [100,200,300], [2,-3,2], [1,2,4],[5,2,5], [-1,2,4], [100,200,300], [2,-3,2], [1,2,4],[5,2,5], [-1,2,4]])
    Y = np.array([[101,201,301], [500, -400, 200], [100,200,300], [2,-3,2], [1,2,4],[5,2,5], [-1,2,4], [100,200,300], [2,-3,2], [1,2,4],[5,2,5], [-1,2,4], [100,200,300], [2,-3,2], [1,2,4],[5,2,5], [-1,2,4], [5,1,2],[-221,202,-124], [-2,-5,-1], [5,2,1]])

    pop, best, hist, hist_avg = minimal_weight_matching(X, Y)
    plt.figure()
    plt.plot(hist, color="blue")
    plt.plot(hist_avg, color="red")
    plt.show()
