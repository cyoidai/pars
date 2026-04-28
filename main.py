#!/usr/bin/env python3

from typing import Iterable, Any
import csv
import math
import random
import textwrap
import itertools
import networkx as nx
import osmnx as ox
# from sklearn.cluster import KMeans
from pars import *


def nearest_neighbor(G: nx.Graph, src, dst_nodes: Iterable | None=None):
    """
    Finds the nearest neighbor from a source node `src` to a set of destination
    nodes `node_list`. If `node_list` is None, we assume all nodes in `G`.
    """
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
    return nearest, dist


# def generate_city() -> nx.Graph:
#     G = nx.waxman_graph(48, .3, .15)
#     G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
#     weights = {}
#     for u, v in G.edges():
#         weights[(u, v)] = euclidean_distance(G, u, v)
#     nx.set_edge_attributes(G, weights, 'length')
#     return G


# def generate_grid_city(m: int, n: int) -> nx.Graph:
#     G: nx.Graph = nx.grid_2d_graph(m, n)
#     pos = {(x,y): (x,y) for x, y in G.nodes()}
#     nx.set_node_attributes(G, pos, 'pos')
#     weights = {}
#     for u, v in G.edges():
#         weights[(u, v)] = euclidean_distance(G, u, v)
#     nx.set_edge_attributes(G, weights, 'length')
#     G.remove_edges_from(random.sample(list(G.edges()), round(.25 * len(G.edges()))))
#     G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
#     return G


def assign_nodes(G: nx.Graph, num_customers: int) -> tuple[Any, list[Any], list[Any]]:
    """
    Given a graph, return a random node for the warehouse, of the remaining
    nodes a list of nodes designated as customers, and a list of unassigned
    leftover nodes.
    """
    if num_customers + 1 > len(G.nodes):
        raise ValueError(textwrap.dedent(f'''
            Cannot assign {num_customers} plus a central warehouse on a graph
            with only {len(G.nodes)} nodes'''))
    remaining_nodes = list(G.nodes())
    warehouse = random.choice(remaining_nodes)
    remaining_nodes.remove(warehouse)
    customers = random.sample(remaining_nodes, k=num_customers)
    for customer in customers:
        remaining_nodes.remove(customer)
    return warehouse, customers, remaining_nodes


def tsp_nn_heuristic(G: nx.Graph, source, nodes: Iterable) -> list:
    """
    Solves a road network TSP using nearest neighbor heuristic for picking the
    next node to visit and A* search for routing between nodes.
    """
    nodes_left = list(nodes)
    current_node = source
    full_path = [current_node]
    while nodes_left:
        next_node, _dist = nearest_neighbor(G, current_node, nodes_left)
        path = nx.astar_path(G, current_node, next_node, lambda u,v:euclidean_distance(G, u, v), 'length')
        full_path += path[1::]
        nodes_left.remove(next_node)
        current_node = next_node
    return full_path


# def cluster_graph_kmeans(G: nx.Graph, customers: list[Any], trucks: int, capacity: int, attr: str='pos') -> list[list[Any]]:
#     partitions = min(trucks, len(customers))
#     k_means = KMeans(partitions)
#     coords = [
#         G.nodes[customer][attr] for customer in customers
#     ]
#     labels = k_means.fit_predict(coords)
#     partitions = [[] for _ in range(partitions)]
#     for node, label in zip(customers, labels):
#         G.nodes[node]['cluster'] = label  # Optional: save back to graph
#         partitions[label].append(node)
#     return partitions


def cluster_graph_sweep(G: nx.Graph, max_cluster_size: int, center: tuple[int | float, int | float]=(0, 0), node_list: Iterable | None=None, pos_attr: str='pos') -> list[list[Any]]:
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
    # nx.set_node_attributes(G, ref_angles, 'ref_angle')
    nodes_sorted = sorted(node_list, key=lambda n:ref_angles[n])
    for cluster in itertools.batched(nodes_sorted, max_cluster_size):
        clusters.append(list(cluster))
    return clusters


def pars(G: nx.Graph, warehouse, customers: list, trucks: int, truck_capacity: int) -> list[list[Any]]:
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

    # performs nearest-neighbor heuristic
    # for cluster in clusters:
    #     path = []
    #     if len(cluster) == 1:
    #         path.extend(nx.astar_path(G, warehouse, cluster[0], lambda n1,n2:euclidean_distance(G, n1, n2), 'length'))
    #         path.extend(nx.astar_path(G, cluster[0], warehouse, lambda n1,n2:euclidean_distance(G, n1, n2), 'length')[1::])
    #     else:
    #         # warehouse to first customer
    #         path.extend(nx.astar_path(G, warehouse, cluster[0], lambda n1,n2:euclidean_distance(G, n1, n2), 'length'))
    #         # first customer to last customer
    #         path.extend(tsp_nn_heuristic(G, cluster[0], cluster[1::])[1::])
    #         # last customer back to warehouse
    #         path.extend(nx.astar_path(G, path[-1], warehouse, lambda n1,n2:euclidean_distance(G, n1, n2))[1::])
    #     routes.append(path)

    # performs simulated annealing
    nonrouting_nodes = customers.copy()
    nonrouting_nodes.append(warehouse)
    K, routings = road_network_to_complete_graph(G, nonrouting_nodes, 'length')
    for cluster in clusters:
        subgraph_nodes = cluster.copy()
        subgraph_nodes.append(warehouse)
        annealing = SimulatedAnnealing(K.subgraph(subgraph_nodes), warehouse)
        best_state = annealing.run()
        route = expand_route(best_state.route, routings)
        routes.append(route)

    return routes


def graph_from_address(address: str, dist: int=10_000) -> nx.MultiDiGraph:
    G = ox.graph_from_address(address, dist, network_type='drive')
    G = ox.project_graph(G)
    # ensure every node can route to every other node
    max_scc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(max_scc)
    pos = { node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes() }
    nx.set_node_attributes(G, pos, 'pos')
    return G


IMAGE_COUNTER = 0


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
