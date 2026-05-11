#!/usr/bin/env python3

from typing import Any
import csv
import math
import random
import textwrap
import itertools
import networkx as nx
import osmnx as ox
# from sklearn.cluster import KMeans
from pars import *


def pars(G: nx.Graph, warehouse, customers: list, trucks: int, truck_capacity: int, algorithm: str) -> list[list[Any]]:
    if not customers:
        return []
    max_packages = trucks * truck_capacity
    if len(customers) > max_packages:
        raise ValueError(textwrap.dedent(f'''
            Fleet is unable to support {len(customers)} customers (customers >
            trucks * truck_capacity). Either increase trucks, truck capacity or
            reduce the number of overall customers.'''))
    routes: list[list[Any]] = []

    clusters = cluster_graph_sweep(G, truck_capacity, G.nodes[warehouse]['pos'], customers)

    nonrouting_nodes = customers.copy()
    nonrouting_nodes.append(warehouse)

    K, routings = road_network_to_complete_graph(G, nonrouting_nodes, 'length')

    for cluster in clusters:
        tsp_nodes = cluster.copy()
        tsp_nodes.append(warehouse)
        K_tsp = K.subgraph(tsp_nodes).copy()

        if algorithm == 'nn':
            optimizer = NearestNeighbor(K_tsp, warehouse)
            best_state = optimizer.run()
            # best_state = optimizer.run(lambda r:draw_routes(G, routes + [expand_route(r, routings)], customers, warehouse, False, True))
            route = expand_route(best_state, routings)
            routes.append(route)

        elif algorithm == 'sa':
            optimizer = SimulatedAnnealing(K_tsp, warehouse)
            best_state = optimizer.run()
            # best_state = optimizer.run(lambda s:draw_routes(G, routes + [expand_route(s.current_state.route, routings)], customers, warehouse, False, True))
            route = expand_route(best_state.route, routings)
            routes.append(route)

        elif algorithm == 'aco':
            optimizer = AntColonyOptimization(K_tsp, warehouse)
            best_state = optimizer.run()
            # best_state = optimizer.run(lambda s:draw_routes(G, routes + [expand_route(s.route, routings)], customers, warehouse, False, True))
            routes.append(expand_route(best_state.route, routings))

        elif algorithm == 'ga':
            optimizer = GeneticAlgorithm(K_tsp, warehouse)
            best_state = optimizer.run()
            # best_state = optimizer.run(lambda s:draw_routes(G, routes + [expand_route(s.route, routings)], customers, warehouse, False, True))
            routes.append(expand_route(best_state.route, routings))

        else:
            raise ValueError('Unknown algorithm')

    return routes


def draw_routes(G: nx.Graph, routes: list[list[Any]], customers: list, warehouse, show: bool, save: bool):
    global IMAGE_COUNTER
    ROUTE_COLORS = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'orange', 'pink']
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node == warehouse:
            node_colors.append('white')
            node_sizes.append(32)
        elif node in customers:
            node_colors.append('white')
            node_sizes.append(32)
        else:
            node_colors.append('lightgrey')
            node_sizes.append(0)
    route_colors = []
    for i in range(len(routes)):
        route_colors.append(ROUTE_COLORS[i % len(ROUTE_COLORS)])
    ox.plot_graph_routes(G, routes, route_colors=route_colors
        , node_size=node_sizes, node_color=node_colors, show=show, save=save
        , filepath=f'out/route{IMAGE_COUNTER}.png')
    IMAGE_COUNTER += 1


def main():
    ADDRESS = '1400 Washington Ave, Albany, NY 12222'
    # ADDRESS = '20 W 34th St., New York, NY 10001'
    # ADDRESS = '1600 Pennsylvania Ave NW, Washington, DC 20500'
    # ADDRESS = '633 W 5th St, Los Angeles, CA 90071'
    # ADDRESS = 'London SW1A 0AA, United Kingdom'
    DISTANCE = 10_000
    ITERATIONS = 1
    DRAW_ROUTES_TO_SCREEN = True
    DRAW_ROUTES_TO_FILE = False
    WRITE_STATS_TO_FILE = False
    CUSTOMERS = 10
    TRUCKS = 1
    TRUCK_CAPACITY = 10

    output_data: list = [('address', 'distance', 'customers', 'trucks', 'truck_capacity', 'total_distance')]

    G = graph_from_address(ADDRESS, DISTANCE)

    # lat = nx.get_node_attributes(G, 'y').values()
    # lon = nx.get_node_attributes(G, 'x').values()
    # print(max(lat), max(lon))
    # print(min(lat), min(lon))

    for _ in range(ITERATIONS):
        # CUSTOMERS = random.randint(1, TRUCKS * TRUCK_CAPACITY)
        warehouse, customers, remaining_nodes = assign_nodes(G, CUSTOMERS)

        # nonrouting_nodes = customers.copy()
        # nonrouting_nodes.append(warehouse)
        # K, routes = road_network_to_complete_graph(G, nonrouting_nodes, 'length')
        # nx.draw(K)
        # plt.show()

        routes = pars(G, warehouse, customers, TRUCKS, TRUCK_CAPACITY)
        total_distance = path_weight_sum(G, routes)
        stats = (ADDRESS, DISTANCE, CUSTOMERS, TRUCKS, TRUCK_CAPACITY, total_distance / 1000)
        output_data.append(stats)
        print(stats)

        if DRAW_ROUTES_TO_SCREEN or DRAW_ROUTES_TO_FILE:
            draw_routes(G, routes, customers, warehouse, DRAW_ROUTES_TO_SCREEN, DRAW_ROUTES_TO_FILE)

    if WRITE_STATS_TO_FILE:
        with open('output.tsv', 'w', newline='', encoding='utf-8') as output_file:
            writer = csv.writer(output_file, delimiter='\t')
            writer.writerows(output_data)


if __name__ == '__main__':
    main()
