
import random
from datetime import datetime
from typing import List
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .models import City, Population, Route


class GeneticAlgorithm:
    def __init__(self, cities: DataFrame):
        self.cities = [
            City(city["name"], city["x"], city["y"], city["GC"], city["GM"])
            for _, city in cities.iterrows()]
        self.n_cities = len(cities)
        self.n_routes = 200
        self.max_generations = 900
        self.mutation_rate = 0.04
        self.population: Population = None
        self.solution_history: List[Route] = []
        self.best_route: Route = None

    def initiate(self):
        self.population = self.genesis()
        self.best_route = self.population.routes[0]

        for i in range(self.max_generations):
            self.population.calculate_fitnesses()

            # Saving the best solution
            if self.population.min_fitness < self.best_route.fitness:
                self.best_route = Route([*self.population.best_route.cities])

            if i % 10 == 0:
                print(i, self.population.best_route.fitness, self.population.avg_fitness,
                      datetime.now().strftime("%d/%m/%y %H:%M"))

            self.solution_history.append(
                Route([*self.population.best_route.cities]))
            self.population.next_generation(self.mutation_rate)

    # First step: Create the first population set
    def genesis(self) -> Population:
        def random_route():
            random.shuffle(self.cities)
            return Route(self.cities)

        return Population([random_route() for _ in range(self.n_routes)])

    def animateSolutions(self):
        self.animate(150)

    def animate(self, speed: int):
        ''' animate the solution over time
            Parameters
            ----------
            hisotry : list
                history of the solutions chosen by the algorith
            points: array_like
                points with the coordinates
        '''

        ''' approx 1500 frames for animation '''
        key_frames_mult = max(1, len(self.solution_history) // speed)

        fig, ax = plt.subplots()
        ''' path is a line coming through all the nodes '''
        line, = plt.plot([], [], lw=2)

        def init():
            ''' initialize node dots on graph '''
            x = [city.x for city in self.solution_history[0].cities]
            y = [city.y for city in self.solution_history[0].cities]
            plt.plot(x, y, 'co')
            for city in self.cities:
                plt.text(city.x, city.y, '({})'.format(city.name))

            ''' draw axes slighty bigger  '''
            extra_x = (max(x) - min(x)) * 0.05
            extra_y = (max(y) - min(y)) * 0.05
            ax.set_xlim(min(x) - extra_x, max(x) + extra_x)
            ax.set_ylim(min(y) - extra_y, max(y) + extra_y)

            '''initialize solution to be empty '''
            line.set_data([], [])
            return line,

        def update(frame):
            ''' for every frame update the solution on the graph '''
            x = [city.x for city in self.solution_history[frame].cities] + \
                [self.solution_history[frame].cities[0].x]
            y = [city.y for city in self.solution_history[frame].cities] + \
                [self.solution_history[frame].cities[0].y]
            line.set_data(x, y)
            return line

        ''' animate precalulated solutions '''

        anim = FuncAnimation(
            fig,
            update,
            frames=range(0, len(self.solution_history), key_frames_mult),
            init_func=init,
            interval=3,
            repeat=False)

        plt.show()
