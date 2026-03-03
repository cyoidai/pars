#!/usr/bin/env python3

import math
import random
from typing import Iterable, Any
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox

def euclidean_distance(G: nx.Graph, n1, n2) -> float:
    x1, y1 = G.nodes[n1]['pos']
    x2, y2 = G.nodes[n2]['pos']
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y2 - y1, 2))

# def euclidean_distance(x1: int | float, y1: int | float, x2: int | float, y2: int | float) -> float:
#     return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y2 - y1, 2))

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
    # x_min, y_min, x_max, y_max = domain
    # node_index = 0
    # node_index_up = None
    # node_index_left = None
    # G = nx.Graph()
    # for y_index in range(y_count):
    #     y = (y_index + 1) * (y_max - y_min / y_count)
    #     for x_index in range(x_count):
    #         x = (x_index + 1) * (x_max - x_min / x_count)
    #         G.add_node(node_index)
    #         nx.set_node_attributes(G, {node_index: (x, y)}, 'pos')
    #         node_index_left = node_index
    #         node_index += 1
    pos = {(x,y): (x,y) for x, y in G.nodes()}
    nx.set_node_attributes(G, pos, 'pos')
    weights = {}
    for u, v in G.edges():
        weights[(u, v)] = euclidean_distance(G, u, v)
    nx.set_edge_attributes(G, weights, 'length')
    G.remove_edges_from(random.sample(list(G.edges()), round(.25 * len(G.edges()))))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G

def assign_nodes(G: nx.Graph, max_customers: int) -> tuple[Any, list[Any], list[Any]]:
    """
    Given a graph, returns a random node for the warehouse, of the remaining
    nodes a list of nodes designated as customers, and a list of unassigned
    leftover nodes.
    """
    remaining_nodes = list(G.nodes())
    if len(remaining_nodes) <= 1:
        raise ValueError('Graph must have at least 2 nodes')
    warehouse = random.choice(remaining_nodes)
    remaining_nodes.remove(warehouse)
    customers = random.sample(remaining_nodes, k=min(max_customers, len(remaining_nodes)))
    for customer in customers:
        nx.set_node_attributes(G, { customer: 'blue' }, 'viz_color')
        remaining_nodes.remove(customer)
    return warehouse, customers, remaining_nodes


def tsp_nn_heuristic(G: nx.Graph, src, destinations: Iterable) -> list:
    dst = list(destinations)
    current_node = src
    full_path = [src]
    # out_graph = nx.DiGraph(G)
    # out_graph.clear_edges()
    while dst:
        next_node, _dist = nearest_neighbor(G, current_node, dst)
        path = nx.astar_path(G, current_node, next_node, lambda u,v:euclidean_distance(G, u, v))
        full_path += path[1::]
        dst.remove(next_node)
        current_node = next_node
    # for i in range(len(full_path) - 1):
    #     out_graph.add_edge(full_path[i], full_path[i + 1])
    return full_path

def pars(G: nx.Graph, warehouse, customers: list, m: int, k: int) -> list[list[Any]]:
    if not customers:
        return []

    routes: list[list[Any]] = []

    path = nx.astar_path(G, warehouse, customers[0], lambda n1,n2:euclidean_distance(G, n1, n2))
    path.extend(tsp_nn_heuristic(G, customers[0], customers[1::])[1::])
    path.extend(nx.astar_path(G, path[-1], warehouse, lambda n1,n2:euclidean_distance(G, n1, n2))[1::])
    routes.append(path)

    return routes


def main():
    MAX_CUSTOMERS = 16
    # G = generate_city()
    # G = generate_grid_city(5, 5)
    _G = ox.graph_from_place('Albany, NY, USA', network_type='drive')
    G = ox.project_graph(_G)
    nx.set_node_attributes(G, {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}, 'pos')

    warehouse, customers, remaining_nodes = assign_nodes(G, MAX_CUSTOMERS)
    print(f'Warehouse: {warehouse}')
    print(f'Customers: {customers}')
    routes = pars(G, warehouse, customers, 0, 0)
    total_distance = 0
    for route in routes:
        total_distance += nx.path_weight(G, route, 'length')
    # print(f'{PLACE}\t{len(customers)}\t{total_distance}')
    print(f'Total distance traveled: {total_distance}')

    # pos = nx.get_node_attributes(G, 'pos')

    # ox.plot_graph_route(G, route)

    node_colors = []
    for node in G.nodes():
        if node == warehouse:
            node_colors.append('yellow')
        elif node in customers:
            node_colors.append('blue')
        else:
            node_colors.append('lightgrey')

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ox.plot_graph(G)
    for route in routes:
        ox.plot_graph_route(G, route, node_color=node_colors)
    # nx.draw_networkx_nodes(G, pos, [warehouse], node_color='red', node_size=.1, ax=ax1)
    # nx.draw_networkx_nodes(G, pos, customers, node_color='blue', node_size=.1, ax=ax1)
    # nx.draw_networkx_nodes(G, pos, remaining_nodes, node_color='grey', node_size=.1, ax=ax1)
    # nx.draw_networkx_edges(G, pos, G.edges(), ax=ax1)
    # ox.plot_graph_route(G, nn_path)
    # use this for waxman graph
    # labels = {node: node for node in G.nodes()}
    # use this for grid graph
    # labels = {node: i + 1 for i, node in enumerate(G.nodes())}
    # nx.draw_networkx_labels(G, pos, labels, ax=ax1)
    # nx.draw_networkx_edges(G, pos, G.edges(), ax=ax1)

    # nx.draw(nn_graph, nx.get_node_attributes(nn_graph, 'pos'), node_size=.05, ax=ax2)
    # nx.draw_networkx_edge_labels(G, pos, {(u, v): round(G.edges[u, v]['length'], 2) for u, v in G.edges()})
    # plt.show()

if __name__ == '__main__':
    main()
