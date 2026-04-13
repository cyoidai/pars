import copy
import random
from typing import Any, Callable
import networkx as nx
import math


class State:
    def __init__(self, K: nx.Graph, route: list[Any]):
        self.K = K
        self.route: list[Any] = route
        self.energy: float = nx.path_weight(K, route, 'length')

    @staticmethod
    def initial(K: nx.Graph, source: Any) -> 'State':
        route = [source]
        nodes = list(K.nodes())
        nodes.remove(source)
        route.extend(nodes)
        route.append(source)
        return State(K, route)

    def next(self) -> 'State':
        new_route = copy.deepcopy(self.route)
        if len(self.route) <= 3:
            return State(self.K, self.route)
        i = random.randint(1, len(new_route) - 2)
        j = random.randint(1, len(new_route) - 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        return State(self.K, new_route)


class SimulatedAnnealing:
    def __init__(self, K: nx.Graph, source: Any, temperature: float=10_000, decay_rate: float=.995):
        self.K = K
        self.temperature = temperature
        self.decay_rate = decay_rate
        self.current_state = State.initial(K, source)
        self.best_state = self.current_state

    def pas(self, next_energy: int | float) -> bool:
        """Probability acceptance function"""
        if next_energy < self.current_state.energy:
            return True
        probability = math.exp(-(next_energy - self.current_state.energy) / self.temperature)
        return random.random() < probability


    def run(self, callback: Callable[['SimulatedAnnealing'], None] | None=None) -> State:
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
