import argparse
import pandas as pd
from Algorithms import SimulatedAnnealing
from Helpers import CityGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default=15, dest='annealing', type=int)
    args = parser.parse_args()



    '''set the simulated annealing algorithm params'''
    temp = 1000
    stopping_temp = 0.00000001
    alpha = 0.9995
    stopping_iter = 10000000

    '''set the dimensions of the grid'''
    size_width = 200
    size_height = 200

    '''set the number of nodes'''
    population_size = 32

    '''generate random list of nodes'''
    cities = CityGenerator(size_width, size_height, population_size).generate()

    '''run simulated annealing algorithm with 2-opt'''
    sa = SimulatedAnnealing(cities, temp, alpha, stopping_temp, stopping_iter)
    sa.anneal()

    '''animate'''
    sa.animateSolutions()

    '''show the improvement over time'''
    sa.plotLearning()


if __name__ == '__main__':
    main()
