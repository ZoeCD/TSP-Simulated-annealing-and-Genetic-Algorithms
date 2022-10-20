

from Helpers import nearestNeighbourSolution, vectorToDistMatrix, animateTSP
import numpy as np
import random

from datetime import datetime
import math
import random
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, cities):
        ''' animate the solution over time
            Parameters
            ----------
            coords: array_like
                list of coordinates
            temp: float
                initial temperature
            alpha: float
                rate at which temp decreases
            stopping_temp: float
                temerature at which annealing process terminates
            stopping_iter: int
                interation at which annealing process terminates
        '''

        self.cities = cities
        self.n_cities = len(cities)
        self.n_population = 25
        self.mutation_rate = 0.4

    def initiate(self):
        population_set = self.genesis(self.cities["name"], self.n_population)
        fitnes_list = self.get_all_fitnes(population_set)
        progenitor_list = self.progenitor_selection(population_set, fitnes_list)
        new_population_set = self.mate_population(progenitor_list)
        mutated_pop = self.mutate_population(new_population_set)

        best_solution = [-1, np.inf, np.array([])]
        for i in range(100):
            if i % 10 == 0: print(i, fitnes_list.min(), fitnes_list.mean(), datetime.now().strftime("%d/%m/%y %H:%M"))
            fitnes_list = self.get_all_fitnes(mutated_pop)

            # Saving the best solution
            if fitnes_list.min() < best_solution[1]:
                best_solution[0] = i
                best_solution[1] = fitnes_list.min()
                best_solution[2] = np.array(mutated_pop)[fitnes_list.min() == fitnes_list]

            progenitor_list = self.progenitor_selection(population_set, fitnes_list)
            new_population_set = self.mate_population(progenitor_list)

            mutated_pop = self.mutate_population(new_population_set)
        return best_solution


    # Function to compute the distance between two points
    def compute_city_distance_coordinates(self, a, b):
        return ((a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5

    def compute_city_distance_names(self, city_a, city_b):
        city_a = self.cities[self.cities["name"] == city_a]
        city_b = self.cities[self.cities["name"] == city_b]
        return self.compute_city_distance_coordinates(city_a.values[0], city_b.values[0])

    # First step: Create the first population set
    def genesis(self, city_list, n_population):
        population_set = []
        for i in range(n_population):
            # Randomly generating a new solution
            sol_i = city_list[np.random.choice(list(range(self.n_cities)), self.n_cities, replace=False)]
            population_set.append(sol_i)
        return np.array(population_set)

    def fitness_eval(self, city_list):
        total = 0
        for i in range(self.n_cities - 1):
            a = city_list[i]
            b = city_list[i + 1]
            total += self.compute_city_distance_names(a, b)
        return total

    def get_all_fitnes(self, population_set):
        fitnes_list = np.zeros(self.n_population)

        # Looping over all solutions computing the fitness for each solution
        for i in range(self.n_population):
            fitnes_list[i] = self.fitness_eval(population_set[i])

        return fitnes_list

    def progenitor_selection(self, population_set, fitnes_list):
        total_fit = fitnes_list.sum()
        prob_list = fitnes_list / total_fit

        # Notice there is the chance that a progenitor. mates with oneself
        progenitor_list_a = np.random.choice(list(range(len(population_set))), len(population_set), p=prob_list,
                                             replace=True)
        progenitor_list_b = np.random.choice(list(range(len(population_set))), len(population_set), p=prob_list,
                                             replace=True)

        progenitor_list_a = population_set[progenitor_list_a]
        progenitor_list_b = population_set[progenitor_list_b]

        return np.array([progenitor_list_a, progenitor_list_b])


    def mate_progenitors(self, prog_a, prog_b):
        offspring = prog_a[0:5]

        for city in prog_b:

            if not city in offspring:
                offspring = np.concatenate((offspring, [city]))

        return offspring


    def mate_population(self, progenitor_list):
        new_population_set = []
        for i in range(progenitor_list.shape[1]):
            prog_a, prog_b = progenitor_list[0][i], progenitor_list[1][i]
            offspring = self.mate_progenitors(prog_a, prog_b)
            new_population_set.append(offspring)

        return new_population_set

    def mutate_offspring(self, offspring):
        for q in range(int(self.n_cities * self.mutation_rate)):
            a = np.random.randint(0, self.n_cities)
            b = np.random.randint(0, self.n_cities)

            offspring[a], offspring[b] = offspring[b], offspring[a]

        return offspring

    def mutate_population(self, new_population_set):
        mutated_pop = []
        for offspring in new_population_set:
            mutated_pop.append(self.mutate_offspring(offspring))
        return mutated_pop

