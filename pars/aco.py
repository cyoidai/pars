import random
from typing import Any, Callable
import networkx as nx

from .annealing import State


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
