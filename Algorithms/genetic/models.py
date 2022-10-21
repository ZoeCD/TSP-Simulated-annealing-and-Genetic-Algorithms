from dataclasses import dataclass
from functools import lru_cache
import math
import random
from typing import List, Tuple


@dataclass
class City:
    name: str
    x: float
    y: float
    GC: int
    GM: int

    def __hash__(self):
        # El hash nos permite memoizar cálculos con las ciudades
        return int(''.join(map(str, map(ord, self.name))))


# memoizamos la distancia para optimizar rendimiento
@lru_cache(None)
def distance(city_a: City, city_b: City) -> int:
    return math.sqrt(((city_a.x - city_b.x) ** 2) + ((city_a.y - city_b.y) ** 2))


@dataclass
class Route:
    def __init__(self, cities: List[City]):
        self.cities = tuple(cities)
        self.n = len(cities)

    @property
    def fitness(self) -> float:
        @lru_cache(None)
        def memoized_fitness(cities: Tuple[City]):
            return sum(distance(a, b) for a, b in tuple(
                zip(cities, [*cities[1:], cities[0]])))
        return memoized_fitness(self.cities)

    def mutate(self):
        self.cities = list(self.cities)
        a = random.randint(0, self.n - 1)
        b = random.randint(0, self.n - 1)
        self.cities[a], self.cities[b] = self.cities[b], self.cities[a]
        self.cities = tuple(self.cities)

    def __repr__(self) -> str:
        return f'{self.fitness:.4}'


class Population:
    def __init__(self, routes: List[Route] = None):
        self.routes = routes
        self.n = len(routes)

    def calculate_fitnesses(self) -> float:
        return sum(route.fitness for route in self.routes)

    def select_progenitors(self) -> List[Route]:
        total_fitness = self.calculate_fitnesses()
        min_fit = self.min_fitness
        weights = tuple(total_fitness / (route.fitness - min_fit + 1)
                        for route in self.routes)
        return random.choices(population=self.routes, weights=weights, k=self.n)

    def next_generation(self, mutation_rate: float = 0.05) -> None:
        def mate(entity1: Route, entity2: Route):
            new_route = list(entity1.cities[:5])
            for city in entity2.cities:
                if city not in new_route:
                    new_route.append(city)
            return Route(new_route)

        mates = self.select_progenitors()

        self.routes = [
            mate(a, b)
            for a, b in zip(mates[::2], mates[1::2])
        ]

        for route in self.routes:
            if random.random() < mutation_rate:
                route.mutate()

    @ property
    def min_fitness(self) -> float:
        return min(route.fitness for route in self.routes)

    @ property
    def avg_fitness(self) -> float:
        return self.calculate_fitnesses() / len(self.routes)

    @ property
    def best_route(self) -> Route:
        min_dist = self.min_fitness
        for route in self.routes:
            if route.fitness == min_dist:
                return route
