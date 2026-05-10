import random
from typing import Any, Callable
import networkx as nx

from .annealing import State


class GeneticAlgorithm:
    def __init__(
        self,
        K: nx.Graph,
        source: Any,
        population_size: int = 80,
        generations: int = 200,
        mutation_rate: float = 0.1
    ):
        self.K = K
        self.source = source
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

        self.nodes = list(K.nodes())
        self.nodes.remove(source)

    def create_route(self):
        route = self.nodes[:]
        random.shuffle(route)
        return [self.source] + route + [self.source]

    def fitness(self, route):
        return nx.path_weight(self.K, route, "length")

    def crossover(self, parent1, parent2):
        middle1 = parent1[1:-1]
        middle2 = parent2[1:-1]

        start = random.randint(0, len(middle1) - 1)
        end = random.randint(start, len(middle1) - 1)

        child_middle = middle1[start:end + 1]

        for node in middle2:
            if node not in child_middle:
                child_middle.append(node)

        return [self.source] + child_middle + [self.source]

    def mutate(self, route):
        new_route = route[:]

        if random.random() < self.mutation_rate and len(new_route) > 3:
            i, j = random.sample(range(1, len(new_route) - 1), 2)
            new_route[i], new_route[j] = new_route[j], new_route[i]

        return new_route

    def run(self, callback: Callable[[State], None] | None = None) -> State:
        population = [self.create_route() for _ in range(self.population_size)]

        for generation in range(self.generations):
            population.sort(key=self.fitness)

            best_route = population[0]

            if callback:
                callback(State(self.K, best_route))

            best_routes = population[:10]
            new_population = best_routes[:]

            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(best_routes, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

        population.sort(key=self.fitness)
        best_route = population[0]

        if callback:
            callback(State(self.K, best_route))

        return State(self.K, best_route)
