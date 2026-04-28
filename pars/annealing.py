import copy
import random
from typing import Any, Callable
import networkx as nx
import math


class State:
    def __init__(self, K: nx.Graph, route: list[Any]):
        self.K = K
        self.route: list[Any] = route
        self.energy: float = nx.path_weight(K, route, "length")

    @staticmethod
    def initial(K: nx.Graph, source: Any) -> "State":
        route = [source]
        nodes = list(K.nodes())
        nodes.remove(source)
        route.extend(nodes)
        route.append(source)
        return State(K, route)

    def next(self) -> "State":
        new_route = copy.deepcopy(self.route)

        if len(self.route) <= 3:
            return State(self.K, self.route)

        i = random.randint(1, len(new_route) - 2)
        j = random.randint(1, len(new_route) - 2)

        new_route[i], new_route[j] = new_route[j], new_route[i]

        return State(self.K, new_route)


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

        if random.random() < self.mutation_rate:
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


class AntColonyOptimization:
    def __init__(
        self,
        K: nx.Graph,
        source: Any,
        ants: int = 40,
        iterations: int = 150,
        alpha: float = 1,
        beta: float = 2,
        evaporation_rate: float = 0.5,
        pheromone_deposit: float = 100
    ):
        self.K = K
        self.source = source
        self.ants = ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit

        self.nodes = list(K.nodes())

        self.pheromone = {
            (u, v): 1.0
            for u in self.nodes
            for v in self.nodes
            if u != v
        }

    def distance(self, u, v):
        return self.K[u][v]["length"]

    def choose_next_node(self, current, unvisited):
        scores = []
        total = 0

        for node in unvisited:
            pheromone_value = self.pheromone[(current, node)] ** self.alpha
            visibility = (1 / self.distance(current, node)) ** self.beta
            score = pheromone_value * visibility

            scores.append((node, score))
            total += score

        if total == 0:
            return random.choice(list(unvisited))

        pick = random.uniform(0, total)
        current_sum = 0

        for node, score in scores:
            current_sum += score

            if current_sum >= pick:
                return node

        return scores[-1][0]

    def build_route(self):
        route = [self.source]
        unvisited = set(self.nodes)
        unvisited.remove(self.source)

        current = self.source

        while unvisited:
            next_node = self.choose_next_node(current, unvisited)
            route.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        route.append(self.source)

        return route

    def evaporate_pheromone(self):
        for edge in self.pheromone:
            self.pheromone[edge] *= (1 - self.evaporation_rate)

    def add_pheromone(self, route):
        route_length = nx.path_weight(self.K, route, "length")

        if route_length == 0:
            return

        deposit = self.pheromone_deposit / route_length

        for i in range(len(route) - 1):
            u = route[i]
            v = route[i + 1]
            self.pheromone[(u, v)] += deposit

    def run(self, callback: Callable[[State], None] | None = None) -> State:
        best_route = None
        best_distance = float("inf")

        for iteration in range(self.iterations):
            routes = []

            for _ in range(self.ants):
                route = self.build_route()
                route_length = nx.path_weight(self.K, route, "length")
                routes.append(route)

                if route_length < best_distance:
                    best_distance = route_length
                    best_route = route

            if callback and best_route is not None:
                callback(State(self.K, best_route))

            self.evaporate_pheromone()

            for route in routes:
                self.add_pheromone(route)

        if callback and best_route is not None:
            callback(State(self.K, best_route))

        return State(self.K, best_route)


class SimulatedAnnealing:
    def __init__(
        self,
        K: nx.Graph,
        source: Any,
        temperature: float = 10_000,
        decay_rate: float = .995
    ):
        self.K = K
        self.temperature = temperature
        self.decay_rate = decay_rate
        self.current_state = State.initial(K, source)
        self.best_state = self.current_state

    def pas(self, next_energy: int | float) -> bool:
        if next_energy < self.current_state.energy:
            return True

        probability = math.exp(
            -(next_energy - self.current_state.energy) / self.temperature
        )

        return random.random() < probability

    def run(self, callback: Callable[["SimulatedAnnealing"], None] | None = None) -> State:
        while self.temperature > 1:
            next_state = self.current_state.next()

            if self.pas(next_state.energy):
                self.current_state = next_state

                if next_state.energy < self.best_state.energy:
                    self.best_state = self.current_state

                if callback:
                    callback(self)

            self.temperature *= self.decay_rate

        return self.best_state