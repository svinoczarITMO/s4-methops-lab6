import math
import random

class SimulatedAnnealing:
    def __init__(self, temp, alpha, stopping_temp, stopping_iter, dist_matrix):
        self.sample_size = len(dist_matrix)
        self.temp = temp
        self.alpha = alpha
        self.stopping_temp = stopping_temp
        self.stopping_iter = stopping_iter
        self.iteration = 1
        self.dist_matrix = dist_matrix
        self.curr_solution = [i for i in range(0, len(dist_matrix))]
        self.best_solution = self.curr_solution
        self.solution_history = [self.curr_solution]
        self.curr_weight = self.weight(self.curr_solution)
        self.initial_weight = self.curr_weight
        self.min_weight = self.curr_weight
        self.weight_list = [self.curr_weight]

    def weight(self, sol):
        return sum([self.dist_matrix[i][j] for i, j in zip(sol, sol[1:] + [sol[0]])])])

    def acceptance_probability(self, candidate_weight):
        return math.exp(-abs(candidate_weight - self.curr_weight) / self.temp)

    def accept(self, candidate):
        candidate_weight = self.weight(candidate)
        if candidate_weight < self.curr_weight:
            self.curr_weight = candidate_weight
            self.curr_solution = candidate
            if candidate_weight < self.min_weight:
                self.min_weight = candidate_weight
                self.best_solution = candidate
        elif random.random() < self.acceptance_probability(candidate_weight):
            self.curr_weight = candidate_weight
            self.curr_solution = candidate

    def anneal(self):
        while self.temp >= self.stopping_temp and self.iteration < self.stopping_iter:
            candidate = list(self.curr_solution)
            l = random.randint(2, self.sample_size - 1)
            i = random.randint(0, self.sample_size - l)
            candidate[i : (i + l)] = reversed(candidate[i : (i + l)])
            self.accept(candidate)
            self.temp *= self.alpha
            self.iteration += 1
            self.weight_list.append(self.curr_weight)
            self.solution_history.append(self.curr_solution)

def run_algorithm(params, matrix):
    temp, alpha, stopping_temp, stopping_iter = params
    sa = SimulatedAnnealing(temp, alpha, stopping_temp, stopping_iter, matrix.values.tolist())
    sa.anneal()
    best_fitness = sa.min_weight
    params = params
    logbook = sa.weight_list
    best_path = sa.best_solution
    return best_fitness, params, logbook, best_path

if __name__ == '__main__':
    params = input("input params: ")
    matrix = input("input matrix: ")
    run_algorithm(params, matrix)