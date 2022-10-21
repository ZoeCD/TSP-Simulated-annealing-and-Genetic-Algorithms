import argparse
import pandas as pd
from Algorithms import SimulatedAnnealing, GeneticAlgorithm
from Helpers import CityGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default='genetic',
                        dest='algorithm', type=str)
    args = parser.parse_args()

    cities = pd.read_csv("Data/cities.csv")

    if args.algorithm == 'annealing':
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
        cities = CityGenerator(size_width, size_height,
                               population_size).generate()

        '''run simulated annealing algorithm with 2-opt'''
        sa = SimulatedAnnealing(cities, temp, alpha,
                                stopping_temp, stopping_iter)
        sa.anneal()

        '''animate'''
        sa.animateSolutions()

        '''show the improvement over time'''
        sa.plotLearning()

    elif args.algorithm == 'genetic':
        gen = GeneticAlgorithm(cities)
        try:
            gen.initiate()
        except KeyboardInterrupt:
            pass
        print("\nBest route:", gen.best_route)
        '''animate'''
        gen.animateSolutions()
    else:
        raise ValueError("Please set a valid option")


if __name__ == '__main__':
    main()
