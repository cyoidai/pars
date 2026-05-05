import math
import random
import networkx as nx



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


# ================= ACO =================
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