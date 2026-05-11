#!/usr/bin/env python3

from typing import Any
import csv
import argparse
import os
import random
import textwrap
import yaml
import networkx as nx
import osmnx as ox
import time
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

        elif algorithm == 'rl':
            with open('config.yaml') as f:
                config = yaml.safe_load(f)
            optimizer = ReinforcementLearningTSP(K_tsp, warehouse, config)
            optimizer.train()
            time.sleep(1) # <-- avoid file busy error
            best_state = optimizer.evaluate()
            routes.append(expand_route(best_state, routings))

        else:
            raise ValueError('Unknown algorithm')

    return routes


def draw_routes(G: nx.Graph, routes: list[list[Any]], customers: list, warehouse, show: bool, save: bool):
    ROUTE_COLORS = ['red', 'magenta', 'green', 'blue', 'cyan', 'orange', 'yellow', 'pink']
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node == warehouse or node in customers:
            node_colors.append('white')
            node_sizes.append(32)
        else:
            node_colors.append('lightgray')
            node_sizes.append(0)
    route_colors = []
    for i in range(len(routes)):
        route_colors.append(ROUTE_COLORS[i % len(ROUTE_COLORS)])

    counter = 0
    filepath = f'out/route{counter}.png'
    while os.path.exists(filepath):
        counter += 1
        filepath = f'out/route{counter}.png'
    ox.plot_graph_routes(G, routes, route_colors=route_colors
        , node_size=node_sizes, node_color=node_colors, show=show, save=save
        , filepath=filepath)


def main():
    parser = argparse.ArgumentParser(
        prog='pars',
        description='Package assignment and routing system (PARS)'
    )
    parser.add_argument('--algorithm', '-a', type=str, help='TSP optimizer', choices=['nn', 'sa', 'aco', 'ga', 'rl'])
    parser.add_argument('--distance', '-d', type=int, default=10_000, help='Distance')
    parser.add_argument('--trucks', '-t', type=int, default=4, help='Number of trucks')
    parser.add_argument('--capacity', '-c', type=int, default=8, help='Capacity of each truck')
    parser.add_argument('--customers', '-C', type=int, default=32, help='Total customers')
    parser.add_argument('--draw-route', type=bool, default=True, help='Draw the output route to the screen')
    parser.add_argument('--write-route', type=bool, default=False, help='Draw the output route to a file')
    parser.add_argument('--write-stats', type=bool, default=False, help='Write route statistics to a file')
    parser.add_argument('--iterations', '-i', type=int, default=1, help='Number of iterations to perform')
    parser.add_argument('address', type=str, help='Reference address')
    args = parser.parse_args()

    ALGORITHM = args.algorithm
    DISTANCE = args.distance
    TRUCKS = args.trucks
    TRUCK_CAPACITY = args.capacity
    CUSTOMERS = args.customers
    DRAW_ROUTES_TO_SCREEN = args.draw_route
    DRAW_ROUTES_TO_FILE = args.write_route
    WRITE_STATS_TO_FILE = args.write_stats
    ITERATIONS = args.iterations
    ADDRESS = args.address

    # overrides
    # PLACE = 'Albany, NY, USA'
    # PLACE = 'New York, NY, USA'
    # PLACE = 'Seoul, South Korea'
    # ADDRESS = '1400 Washington Ave, Albany, NY 12222'
    # ADDRESS = '1901 W Madison St, Chicago, IL 60612'
    # ADDRESS = '1325 Pearl St, Boulder, CO 80302'
    # DISTANCE = 10_000
    # ITERATIONS = 1
    # DRAW_ROUTES_TO_SCREEN = True
    # DRAW_ROUTES_TO_FILE = False
    # WRITE_STATS_TO_FILE = False
    # TRUCKS = 10
    # TRUCK_CAPACITY = 100
    # CUSTOMERS = TRUCKS * TRUCK_CAPACITY

    output_header = ('address', 'distance', 'customers', 'trucks', 'truck_capacity', 'algorithm', 'total_distance')
    print(output_header)

    G = graph_from_address(ADDRESS, DISTANCE)
    # G = graph_from_place(PLACE, DISTANCE)

    for _ in range(ITERATIONS):
        # CUSTOMERS = random.randint(1, TRUCKS * TRUCK_CAPACITY)
        warehouse, customers, remaining_nodes = assign_nodes(G, CUSTOMERS)

        try:
            routes = pars(G, warehouse, customers, TRUCKS, TRUCK_CAPACITY, ALGORITHM)
        except RuntimeError as e:
            print(e)
            continue
        total_distance = path_weight_sum(G, routes)
        stats = (ADDRESS, DISTANCE, CUSTOMERS, TRUCKS, TRUCK_CAPACITY, ALGORITHM, total_distance / 1000)
        print(stats)

        if DRAW_ROUTES_TO_SCREEN or DRAW_ROUTES_TO_FILE:
            draw_routes(G, routes, customers, warehouse, DRAW_ROUTES_TO_SCREEN, DRAW_ROUTES_TO_FILE)

        if WRITE_STATS_TO_FILE:
            max_retries = 5
            output_filepath = 'output.tsv'
            write_header = not os.path.exists(output_filepath)
            for retries in range(max_retries):
                try:
                    with open(output_filepath, 'a', newline='', encoding='utf-8') as output_file:
                        writer = csv.writer(output_file, delimiter='\t')
                        if write_header:
                            writer.writerows([output_header, stats])
                        else:
                            writer.writerow(stats)
                    break
                except PermissionError as e:
                    print('Unable to write stats, %s. Retrying... %s/%s', e, retries, max_retries)


if __name__ == '__main__':
    main()
