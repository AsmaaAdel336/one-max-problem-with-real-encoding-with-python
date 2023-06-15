import numpy as np
import random as rd
import math

rd.seed(10)


def population(npop, clen):
    pop = np.random.randint(0, 2, (npop, clen))

    return pop


def standerd_decoding(binaryInd, R_min, R_max, vLength=5):
    l = np.size(binaryInd)
    binsum = 0
    for i in range(l):
        binsum = binsum + binaryInd[i] * (2**(l-i-1))

    x = R_min+(binsum/(2**(l)))*(R_max-R_min)

    return x


def convert_to_real_with_standard_decoding(pop, pop_size, num_of_variables, x_length, x_min, x_max):
    real = np.zeros((pop_size, num_of_variables))
    iterator = 0  # to know from where to start slicing each variable
    for i in range(pop_size):
        for j in range(num_of_variables):
            real[i][j] = standerd_decoding(
                pop[i][iterator:iterator+x_length[j]], x_min[j], x_max[j])
            iterator += x_length[j]
        iterator = 0
    return real


def fitness(real, pop_size):  # only for two variables as the objective function in assignment 2
    fit = np.zeros((pop_size, 1))

    for i in range(pop_size):
        fit[i] = 8-(real[i][0]+0.0317)**2 + (real[i][1])**2

    return fit


def fitness_with_constraint(real, pop_size):
    fit = np.zeros((pop_size, 1))

    for i in range(pop_size):
        fit[i] = (8-(real[i][0]+0.0317)**2 + (real[i][1])**2) - \
            abs(real[i][0]+real[i][1] - 1)

    return fit


def gray_decoding(binaryInd, x_min, x_max):
    l = np.size(binaryInd)
    binsum = 0
    for j in range(l-1):
        for i in range(j):
            binsum = binsum + binaryInd[i]
        binsum = binsum % 2
    return x_min + ((binsum * (2**(l-j-1)))/(2**l)) * (x_max-x_min)


def convert_to_real_with_gray_decoding(pop, pop_size, num_of_variables, x_length, x_min, x_max):
    gray = np.zeros((pop_size, num_of_variables))
    iterator = 0
    for i in range(pop_size):
        for j in range(num_of_variables):
            gray[i][j] = gray_decoding(
                pop[i][iterator:iterator+x_length[j]], x_min[j], x_max[j])
            iterator += x_length[j]
        iterator = 0
    return gray


def probability_of_selection(fitness):
    # calculate the probability of selection
    probability_of_selection = []
    sum_of_fitness = sum(fitness)
    if sum_of_fitness != 0:
        for i in range(len(fitness)):
            probability_of_selection.append(fitness[i]/sum_of_fitness)
    else:
        probability_of_selection = len(fitness)*[0]
    return probability_of_selection


def cumulative_probability_of_selection(prob_of_selection):
    #  calculate the cumulative probability of selection

    cumulative = np.zeros_like(prob_of_selection)
    cumulative[0] = prob_of_selection[0]

    for i in range(1, len(prob_of_selection)):
        cumulative[i] = cumulative[i-1]+prob_of_selection[i]

    return cumulative


def Roulette_Wheel(cumulative):
    R = np.random.random()
    for i in range(len(cumulative)):
        if R <= cumulative[i]:
            return i


def Roulette_Wheel_selection(cumulative, population):

    parents_matrix = np.zeros((2, np.size(population, 1)))
    parent1_indx = Roulette_Wheel(cumulative)
    parent2_indx = Roulette_Wheel(cumulative)
    parents_matrix[0] = population[parent1_indx]
    parents_matrix[1] = population[parent2_indx]
    return parents_matrix


def binCross(twoParents, pcross, clen):
    R = rd.random()
    print(R)
    twochildren = twoParents.copy()  # deep copy not shallow copy
    if R < pcross:
        cutting_point = rd.randint(1, 5)
       #cutting_point = math.floor(cutting_point)
        print("cutting point: ", cutting_point)
        for i in range(1, clen):
            if i == cutting_point:
                twochildren[0, i:] = twoParents[1, i:]
                twochildren[1, i:] = twoParents[0, i:]
                return twochildren
    else:
        return twochildren


def binMutate(individual, pmute, clen):
    mutatedInd = individual.copy()
    for i in range(clen):
        R = rd.random()
        if R < pmute:
            mutatedInd[i] = 1-individual[i]
    return mutatedInd


def Elitism(pop=[], fitness=[]):
    elitism = np.zeros((2, np.size(pop, 1)))
    fit_copy = fitness.copy()
    fit_copy.sort(reverse=True)
    max_fitness = fit_copy[0]
    second_max_fitness = fit_copy[1]
    for i in range(len(fitness)):
        if fitness[i] == max_fitness:
            indx1 = i
        if fitness[i] == second_max_fitness:
            indx2 = i
    elitism[0, :] = pop[indx1]
    elitism[1, :] = pop[indx2]
    fitness.remove(max_fitness)
    fitness.remove(second_max_fitness)
    # to delete row(axis 0 = horizontal = row) of indx1 in matrix pop
    pop = np.delete(pop, indx1, 0)
    pop = np.delete(pop, indx2-1, 0)

    return elitism


# without elitism & with standard decodig
def run_real_GA(npop, clen, num_of_variables, x_length, x_min, x_max, ngen, pcross, pmute):

    best_hist = []
    avg_best_hist = 0
    pop = population(npop, clen)
    real = convert_to_real_with_standard_decoding(
        pop, npop, num_of_variables, x_length, x_min, x_max)
    new_generation = np.zeros_like(pop)
    fit = fitness(real, npop)
    for i in range(ngen):
        probs = probability_of_selection(fit)
        cumulative = cumulative_probability_of_selection(probs)
        for j in range(npop, 2):
            twoparents = Roulette_Wheel_selection(cumulative, pop)
            twochildren = binCross(twoparents, pcross, clen)
            new_generation[i, :] = twochildren[0, :]
            new_generation[i+1, :] = twochildren[1, :]
        for k in range(npop):
            mutedInd = binMutate(new_generation[k, :], pmute, clen)
            new_generation[k, :] = mutedInd
        pop = new_generation.copy()
        fit = fitness(real, npop)
        best_hist.append(max(fit))
        avg_best_hist = (sum(best_hist)/len(best_hist))
    real = convert_to_real_with_standard_decoding(
        pop, npop, num_of_variables, x_length, x_min, x_max)
    return real, best_hist, avg_best_hist


def run_real_GA_with_Elitism(npop, clen, num_of_variables, x_length, x_min, x_max, ngen, pcross, pmute):

    best_hist = []
    avg_best_hist = 0
    pop = population(npop, clen)
    real = convert_to_real_with_standard_decoding(
        pop, npop, num_of_variables, x_length, x_min, x_max)
    new_generation = np.zeros_like(pop)
    fit = fitness(real, npop)
    elitism = Elitism(pop, fit)
    new_generation[0, :] = elitism[0, :]
    new_generation[1, :] = elitism[1, :]
    for i in range(ngen):
        probs = probability_of_selection(fit)
        cumulative = cumulative_probability_of_selection(probs)
        for j in range(npop-2, 2):
            twoparents = Roulette_Wheel_selection(cumulative, pop)
            twochildren = binCross(twoparents, pcross, clen)
            new_generation[i+2, :] = twochildren[0, :]
            new_generation[i+3, :] = twochildren[1, :]
        for k in range(npop-2):
            mutedInd = binMutate(new_generation[k, :], pmute, clen)
            new_generation[k+2, :] = mutedInd
        pop = new_generation.copy()
        fit = fitness(real, npop)
        best_hist.append(max(fit))
        avg_best_hist = math.floor(sum(best_hist)/len(best_hist))
        elitism = Elitism(pop, fit)
        new_generation[0, :] = elitism[0, :]
        new_generation[1, :] = elitism[1, :]
    real = convert_to_real_with_standard_decoding(
        pop, npop, num_of_variables, x_length, x_min, x_max)
    return real, best_hist, avg_best_hist


print(run_real_GA(20, 25, 5, [5, 5, 5, 5, 5], [-2, -2, -2, -2, -2],
      [2, 2, 2, 2, 2], 100, 0.6, 0.05))  # to run the code without elitism


"""""
Different precisions:

print(run_real_GA(20, 35, 5, [10, 5, 5, 10, 5], [-2, -2, -2, -2, -2],[2, 2, 2, 2, 2], 100, 0.6, 0.05))
print(run_real_GA(20, 15, 5, [3, 3, 3, 3, 3], [-2, -2, -2, -2, -2],[2, 2, 2, 2, 2], 100, 0.6, 0.05))
print(run_real_GA(20, 25, 5, [3, 3, 7, 7, 5], [-2, -2, -2, -2, -2],[2, 2, 2, 2, 2], 100, 0.6, 0.05))
print(run_real_GA(20, 20, 5, [3, 4, 3, 5, 5], [-2, -2, -2, -2, -2],[2, 2, 2, 2, 2], 100, 0.6, 0.05))
print(run_real_GA(20, 10, 5, [2, 2, 2, 2, 2], [-2, -2, -2, -2, -2],[2, 2, 2, 2, 2], 100, 0.6, 0.05))


"""
