#!/usr/bin/env python3

from typing import Iterable, Any
import csv
import math
import random
import textwrap
import sys
import os

import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox

from pars.utils import *
from pars.annealing import SimulatedAnnealing, GeneticAlgorithm, AntColonyOptimization


IMAGE_COUNTER = 0


def draw_routes(G: nx.Graph, routes: list[list[Any]], customers: list, warehouse, show: bool, save: bool, algorithm="route"):
    global IMAGE_COUNTER

    os.makedirs("out", exist_ok=True)

    ROUTE_COLORS = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'orange', 'pink']

    node_colors = []
    node_sizes = []

    for node in G.nodes():
        if node == warehouse or node in customers:
            node_colors.append('white')
            node_sizes.append(32)
        else:
            node_colors.append('lightgrey')
            node_sizes.append(0)

    route_colors = [ROUTE_COLORS[i % len(ROUTE_COLORS)] for i in range(len(routes))]

    ox.plot_graph_routes(
        G,
        routes,
        route_colors=route_colors,
        node_size=node_sizes,
        node_color=node_colors,
        show=show,
        save=save,
        filepath=f'out/{algorithm}_route{IMAGE_COUNTER}.png'
    )

    IMAGE_COUNTER += 1


def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


def nearest_neighbor(G: nx.Graph, src, dst_nodes: Iterable | None = None):
    min_dist = float('inf')
    nearest = None

    if dst_nodes is None:
        dst_nodes = G.nodes()

    for dst in dst_nodes:
        if dst == src:
            continue

        dist = euclidean_distance(G, src, dst)

        if dist < min_dist:
            min_dist = dist
            nearest = dst

    return nearest, min_dist


def generate_city() -> nx.Graph:
    G = nx.waxman_graph(48, .3, .15)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    weights = {}
    for u, v in G.edges():
        weights[(u, v)] = euclidean_distance(G, u, v)

    nx.set_edge_attributes(G, weights, 'length')
    return G


def generate_grid_city(m: int, n: int) -> nx.Graph:
    G: nx.Graph = nx.grid_2d_graph(m, n)

    pos = {(x, y): (x, y) for x, y in G.nodes()}
    nx.set_node_attributes(G, pos, 'pos')

    weights = {}
    for u, v in G.edges():
        weights[(u, v)] = euclidean_distance(G, u, v)

    nx.set_edge_attributes(G, weights, 'length')

    G.remove_edges_from(random.sample(list(G.edges()), round(.25 * len(G.edges()))))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    return G


def assign_nodes(G: nx.Graph, num_customers: int) -> tuple[Any, list[Any], list[Any]]:
    if num_customers + 1 > len(G.nodes):
        raise ValueError(textwrap.dedent(f'''
            Cannot assign {num_customers} plus a central warehouse on a graph
            with only {len(G.nodes)} nodes
        '''))

    remaining_nodes = list(G.nodes())

    warehouse = random.choice(remaining_nodes)
    remaining_nodes.remove(warehouse)

    customers = random.sample(remaining_nodes, k=num_customers)

    for customer in customers:
        remaining_nodes.remove(customer)

    return warehouse, customers, remaining_nodes


def tsp_nn_heuristic(G: nx.Graph, source, nodes: Iterable) -> list:
    nodes_left = list(nodes)
    current_node = source
    full_path = [current_node]

    while nodes_left:
        next_node, _dist = nearest_neighbor(G, current_node, nodes_left)

        path = nx.astar_path(
            G,
            current_node,
            next_node,
            lambda u, v: euclidean_distance(G, u, v),
            'length'
        )

        full_path += path[1:]
        nodes_left.remove(next_node)
        current_node = next_node

    return full_path


def cluster_graph_sweep(
    G: nx.Graph,
    max_cluster_size: int,
    center: tuple[int | float, int | float] = (0, 0),
    node_list: Iterable | None = None,
    pos_attr: str = 'pos'
) -> list[list[Any]]:

    if node_list is None:
        node_list = G.nodes()

    clusters = []
    center_x, center_y = center
    ref_angles = {}

    for node in node_list:
        node_x, node_y = G.nodes[node][pos_attr]

        x = node_x - center_x
        y = node_y - center_y

        angle = math.atan2(y, x)
        ref_angles[node] = angle

    nodes_sorted = sorted(node_list, key=lambda n: ref_angles[n])

    for cluster in batched(nodes_sorted, max_cluster_size):
        clusters.append(list(cluster))

    return clusters


def pars(G, warehouse, customers, trucks, truck_capacity, algorithm="sa"):
    if not customers:
        return []

    max_packages = trucks * truck_capacity

    if len(customers) > max_packages:
        raise ValueError("Too many customers for trucks/capacity")

    routes = []

    clusters = cluster_graph_sweep(
        G,
        truck_capacity,
        G.nodes[warehouse]["pos"],
        customers
    )

    nonrouting_nodes = customers.copy()
    nonrouting_nodes.append(warehouse)

    K, routings = road_network_to_complete_graph(
        G,
        nonrouting_nodes,
        "length"
    )

    for cluster in clusters:
        subgraph_nodes = cluster.copy()
        subgraph_nodes.append(warehouse)

        sub_K = K.subgraph(subgraph_nodes).copy()

        if algorithm == "sa":
            optimizer = SimulatedAnnealing(sub_K, warehouse)

            best_state = optimizer.run(
                lambda s: draw_routes(
                    G,
                    [expand_route(s.current_state.route, routings)],
                    customers,
                    warehouse,
                    False,
                    True,
                    algorithm
                )
            )

        elif algorithm == "ga":
         optimizer = GeneticAlgorithm(sub_K, warehouse)
         best_state = optimizer.run(
        lambda state: draw_routes(
            G,
            [expand_route(state.route, routings)],
            customers,
            warehouse,
            False,
            True,
            algorithm
        )
    )

        elif algorithm == "aco":
         optimizer = AntColonyOptimization(sub_K, warehouse)
         best_state = optimizer.run(
        lambda state: draw_routes(
            G,
            [expand_route(state.route, routings)],
            customers,
            warehouse,
            False,
            True,
            algorithm
        )
    )

        else:
            raise ValueError("Invalid algorithm. Use: sa, ga, or aco")

        route = expand_route(best_state.route, routings)
        routes.append(route)

    return routes


def main():
    os.makedirs("out", exist_ok=True)

    algorithm = sys.argv[1] if len(sys.argv) > 1 else "sa"

    if algorithm not in ["sa", "ga", "aco"]:
        raise ValueError("Use: python main.py sa OR python main.py ga OR python main.py aco")

    print("Running algorithm:", algorithm)

    ADDRESS = '1400 Washington Ave, Albany, NY 12222'
    DISTANCE = 10_000
    ITERATIONS = 1
    SHOW_ROUTES = False
    SAVE_ROUTES = True
    WRITE_TO_FILE = True
    CUSTOMERS = 10
    TRUCKS = 1
    TRUCK_CAPACITY = 10

    output_data: list = [
        ('algorithm', 'address', 'distance', 'customers', 'trucks', 'truck_capacity', 'total_distance')
    ]

    G = graph_from_address(ADDRESS, DISTANCE)

    for _ in range(ITERATIONS):
        warehouse, customers, remaining_nodes = assign_nodes(G, CUSTOMERS)

        routes = pars(G, warehouse, customers, TRUCKS, TRUCK_CAPACITY, algorithm)

        total_distance = path_weight_sum(G, routes)

        stats = (
            algorithm,
            ADDRESS,
            DISTANCE,
            CUSTOMERS,
            TRUCKS,
            TRUCK_CAPACITY,
            total_distance / 1000
        )

        output_data.append(stats)
        print(stats)

        if SHOW_ROUTES or SAVE_ROUTES:
            draw_routes(G, routes, customers, warehouse, SHOW_ROUTES, SAVE_ROUTES, algorithm)

    if WRITE_TO_FILE:
       os.makedirs("data", exist_ok=True)

       with open(f'data/sweep_{algorithm}.tsv', 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.writer(output_file, delimiter='\t')
        writer.writerows(output_data)


if __name__ == '__main__':
    main()