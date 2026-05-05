#!/usr/bin/env python3

import math
import random
import sys
import os
from typing import Any

import networkx as nx
import osmnx as ox

IMAGE_COUNTER = 0


def euclidean_distance(G, n1, n2, attr='pos'):
    x1, y1 = G.nodes[n1][attr]
    x2, y2 = G.nodes[n2][attr]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def path_weight_sum(G, routes):
    return sum(nx.path_weight(G, r, "length") for r in routes)


def expand_route(route, routings):
    full = [route[0]]
    for i in range(len(route)-1):
        full += routings[(route[i], route[i+1])][1:]
    return full


def road_network_to_complete_graph(G, nodes):
    K = nx.DiGraph()
    K.add_nodes_from(nodes)

    routings = {}

    for src in nodes:
        dist, paths = nx.single_source_dijkstra(G, src, weight="length")

        for dst in nodes:
            if src == dst or dst not in paths:
                continue

            routings[(src, dst)] = paths[dst]
            K.add_edge(src, dst, length=dist[dst])

    return K, routings


def graph_from_address(address, dist=5000):
    print("Downloading road network...")
    G = ox.graph_from_address(address, dist=dist, network_type="drive")
    G = ox.project_graph(G)

    scc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(scc).copy()

    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()}
    nx.set_node_attributes(G, pos, "pos")

    print("Graph ready:", len(G.nodes()), "nodes")
    return G


def draw_routes(G, routes, customers, warehouse, name="route"):
    global IMAGE_COUNTER

    os.makedirs("out", exist_ok=True)

    base_colors = [
        "red", "blue", "green", "orange", "purple",
        "cyan", "magenta", "yellow", "lime", "pink",
        "teal", "gold", "coral", "navy"
    ]
    route_colors = [base_colors[i % len(base_colors)] for i in range(len(routes))]

    node_colors = []
    node_sizes = []

    for n in G.nodes():
        if n == warehouse:
            node_colors.append("yellow")
            node_sizes.append(60)
        elif n in customers:
            node_colors.append("white")
            node_sizes.append(40)
        else:
            node_colors.append("lightgrey")
            node_sizes.append(0)

    ox.plot_graph_routes(
        G,
        routes,
        route_colors=route_colors,
        route_linewidths=[4]*len(routes),
        node_size=node_sizes,
        node_color=node_colors,
        bgcolor="black",
        show=False,
        save=True,
        filepath=f"out/{name}_{IMAGE_COUNTER}.png"
    )

    IMAGE_COUNTER += 1


def draw_single_route(G, route, customers, warehouse, name, color):
    os.makedirs("out", exist_ok=True)

    node_colors = []
    node_sizes = []

    for n in G.nodes():
        if n == warehouse:
            node_colors.append("yellow")
            node_sizes.append(70)
        elif n in customers:
            node_colors.append("white")
            node_sizes.append(50)
        else:
            node_colors.append("lightgrey")
            node_sizes.append(0)

    ox.plot_graph_routes(
        G,
        [route],
        route_colors=[color],
        route_linewidths=[4],
        node_size=node_sizes,
        node_color=node_colors,
        bgcolor="black",
        show=False,
        save=True,
        filepath=f"out/{name}.png"
    )


class State:
    def __init__(self, K, route):
        self.K = K
        self.route = route
        self.energy = nx.path_weight(K, route, "length")

    @staticmethod
    def initial(K, source):
        nodes = list(K.nodes())
        nodes.remove(source)
        return State(K, [source] + nodes + [source])

    def next(self):
        r = self.route[:]
        i, j = random.sample(range(1, len(r)-1), 2)
        r[i], r[j] = r[j], r[i]
        return State(self.K, r)


class SimulatedAnnealing:
    def __init__(self, K, source):
        self.K = K
        self.temperature = 10000
        self.decay = 0.995
        self.current = State.initial(K, source)
        self.best = self.current

    def accept(self, next_energy):
        if next_energy < self.current.energy:
            return True
        prob = math.exp(-(next_energy - self.current.energy) / self.temperature)
        return random.random() < prob

    def run(self):
        while self.temperature > 1:
            nxt = self.current.next()

            if self.accept(nxt.energy):
                self.current = nxt
                if nxt.energy < self.best.energy:
                    self.best = nxt

            self.temperature *= self.decay

        return self.best


class GeneticAlgorithm:
    def __init__(self, K, source):
        self.K = K
        self.source = source
        self.population_size = 120
        self.generations = 400
        self.mutation_rate = 0.2
        self.elite_size = 10

        self.nodes = list(K.nodes())
        self.nodes.remove(source)

    def create_route(self):
        r = self.nodes[:]
        random.shuffle(r)
        return [self.source] + r + [self.source]

    def fitness(self, route):
        try:
            return nx.path_weight(self.K, route, "length")
        except:
            return float("inf")

    def tournament(self, pop):
        return min(random.sample(pop, 5), key=self.fitness)

    def crossover(self, p1, p2):
        p1 = p1[1:-1]
        p2 = p2[1:-1]

        a, b = sorted(random.sample(range(len(p1)), 2))
        child = [None]*len(p1)
        child[a:b] = p1[a:b]

        ptr = 0
        for n in p2:
            if n not in child:
                while child[ptr] is not None:
                    ptr += 1
                child[ptr] = n

        return [self.source] + child + [self.source]

    def mutate(self, r):
        r = r[:]

        if random.random() < self.mutation_rate:
            i, j = random.sample(range(1, len(r)-1), 2)
            r[i], r[j] = r[j], r[i]

        if random.random() < self.mutation_rate:
            i, j = sorted(random.sample(range(1, len(r)-1), 2))
            r[i:j] = reversed(r[i:j])

        return r

    def run(self):
        pop = [self.create_route() for _ in range(self.population_size)]

        for g in range(self.generations):
            pop.sort(key=self.fitness)

            if g % 50 == 0:
                print(f"GA Gen {g} Best:", self.fitness(pop[0]))

            new_pop = pop[:self.elite_size]

            while len(new_pop) < self.population_size:
                p1 = self.tournament(pop)
                p2 = self.tournament(pop)

                child = self.crossover(p1, p2)
                child = self.mutate(child)

                new_pop.append(child)

            pop = new_pop

        pop.sort(key=self.fitness)
        return State(self.K, pop[0])


class AntColonyOptimization:
    def __init__(self, K, source):
        self.K = K
        self.source = source
        self.nodes = list(K.nodes())

        self.pheromone = {(u, v): 1 for u in self.nodes for v in self.nodes if u != v}

    def distance(self, u, v):
        return self.K[u][v]["length"]

    def choose(self, cur, unvisited):
        scores = []
        total = 0

        for n in unvisited:
            s = self.pheromone[(cur, n)] * (1/self.distance(cur, n))
            scores.append((n, s))
            total += s

        pick = random.uniform(0, total)
        curr = 0

        for n, s in scores:
            curr += s
            if curr >= pick:
                return n

        return scores[-1][0]

    def run(self):
        best = None
        best_d = float("inf")

        for _ in range(150):
            for _ in range(40):
                route = [self.source]
                unvisited = set(self.nodes)
                unvisited.remove(self.source)

                cur = self.source

                while unvisited:
                    nxt = self.choose(cur, unvisited)
                    route.append(nxt)
                    unvisited.remove(nxt)
                    cur = nxt

                route.append(self.source)

                d = nx.path_weight(self.K, route, "length")

                if d < best_d:
                    best = route
                    best_d = d

        return State(self.K, best)


def cluster_nodes(nodes, capacity):
    return [nodes[i:i+capacity] for i in range(0, len(nodes), capacity)]


def pars(G, warehouse, customers, trucks, cap, algo):
    clusters = cluster_nodes(customers, cap)

    nodes = customers + [warehouse]
    K, routings = road_network_to_complete_graph(G, nodes)

    routes = []

    for cluster in clusters:
        sub_nodes = cluster + [warehouse]
        sub_K = K.subgraph(sub_nodes).copy()

        if algo == "sa":
            best = SimulatedAnnealing(sub_K, warehouse).run()
        elif algo == "ga":
            best = GeneticAlgorithm(sub_K, warehouse).run()
        elif algo == "aco":
            best = AntColonyOptimization(sub_K, warehouse).run()
        else:
            raise ValueError("Use sa / ga / aco")

        route = expand_route(best.route, routings)
        routes.append(route)

    return routes


def main():
    algo = sys.argv[1] if len(sys.argv) > 1 else "ga"

    ADDRESS = "1400 Washington Ave, Albany, NY"
    DIST = 5000
    CUSTOMERS = 1000
    TRUCKS = 10
    CAPACITY = 6

    print("Running:", algo)

    G = graph_from_address(ADDRESS, DIST)

    nodes = list(G.nodes())
    warehouse = random.choice(nodes)
    customers = random.sample(nodes, CUSTOMERS)

    routes = pars(G, warehouse, customers, TRUCKS, CAPACITY, algo)

    total = path_weight_sum(G, routes)
    print("Total distance:", total)

    colors = [
        "red", "blue", "green", "orange", "purple",
        "cyan", "magenta", "yellow", "lime", "pink"
    ]

    for i, route in enumerate(routes):
        color = colors[i % len(colors)]
        draw_single_route(G, route, customers, warehouse, f"{algo}_route_{i}", color)

    draw_routes(G, routes, customers, warehouse, f"{algo}_final")


if __name__ == "__main__":
    main()