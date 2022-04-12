r'''
DISCLAIMER! THIS FILE IS NOT ORIGINALLY CREATED BY ME. IT IS USED FOR
PRACTICE PURPOSES. ALL CREDITS GO TO:

https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
'''

import numpy as np

r'''
 First we set an evaluation function. This function takes the negative sum of all elements in a parent
 We artificially say that the lower the better.
 Individuals will be lists of n_bits bits, if an individual has only 1's, it is perfect
''' 
def OneMax(x):
    return(-sum(x))

r'''
 This is a selection mechanism where parents can be chosen multiple times dependent on performance
 we sample a parent and check its value against k - 1 other parents
 out of these 3 parents, the score of the best parent is returned
 this particular form of selection is called a tournament selection
'''
def Selection(pop, scores, k = 3):
    selection_ix = np.random.randint(len(pop)) #obtain random parent
    for ix in np.random.randint(0,len(pop), k - 1): # for k -1 other parents
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return(pop[selection_ix])

r'''
 the probability of a crossover happening is 85%
 if crossover does not happen, the children get same bits as their parents.
 when it does, child1 gets the last pt = n random bits from parent1 and the first pt random bits from p2
 c2 gets last pt from parent 2 and first pt from p1
'''
def Crossover(p1, p2, r_cross = 0.85):
    c1, c2 = p1.copy(), p2.copy()
    if np.random.rand() < r_cross:
        pt = np.random.randint(1, len(p1) - 2)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return([c1, c2])

# This function loops over each bit in an individual and will mutate the bit 5% of the time
# in our case this means the bit is flipped from 0 to 1 or other way around.
def Mutation(individual, r_mut = 0.05):
    for gene in range(len(individual)):
        if np.random.rand() < r_mut:
            individual[gene] = 1 - individual[gene]

#this is the entire genetic algorithm
def GeneticAlgorithm(n_pop, n_bits, evaluate, n_iter, r_cross, r_mut):
    pop = [np.random.randint(0,2,n_bits).tolist() for ind in range(n_pop)] #define population
    best, best_eval = 0, evaluate(pop[0]) # set initial best member and it evaluation 
    for gen in range(n_iter): # iterate over generations
        scores = [evaluate(ind) for ind in pop] # evaluate current population
        for i in range(n_pop): # for each member
            if scores[i] < best_eval: # check if its score is better than the current best score
                best, best_eval = pop[i], scores[i] #if it is, set new best member and its evaluation
                print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i])) #Print new solution
        selected = [Selection(pop, scores) for i in range(n_pop)] # select parents
        children = [] # create space for children
        for i in range(0, n_pop, 2): #for every two parents
            p1, p2 = selected[i], selected[i+1] # obtain parent bits
            for child in Crossover(p1, p2, r_cross): #for each child after crossover
                Mutation(child, r_mut) # mutate the child
                children.append(child) # append child to children
        pop = children
    return([best, best_eval]) #return best member along with its evaluation

GeneticAlgorithm(100, 20, OneMax, 20, 0.85, 0.05)

        
