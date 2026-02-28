#!/usr/bin/env python3

import math
import random
from copy import copy
from typing import Iterable, Collection
import networkx as nx
import matplotlib.pyplot as plt

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
    nx.set_edge_attributes(G, weights, 'weight')
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
    nx.set_edge_attributes(G, weights, 'weight')
    G.remove_edges_from(random.sample(list(G.edges()), round(.25 * len(G.edges()))))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G

def assign_nodes(G: nx.Graph, max_customers: int=8) -> tuple[int, list[int], list[int]]:
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
        remaining_nodes.remove(customer)
    return warehouse, customers, remaining_nodes


def tsp_nn_heuristic(G: nx.Graph, src, destinations: Iterable) -> tuple[list, nx.DiGraph]:
    dst = list(destinations)
    current_node = src
    full_path = [src]
    out_graph = nx.DiGraph(G)
    out_graph.clear_edges()
    while dst:
        next_node, _dist = nearest_neighbor(G, current_node, dst)
        path = nx.astar_path(G, current_node, next_node, lambda u,v:euclidean_distance(G, u, v))
        full_path += path[1::]
        dst.remove(next_node)
        current_node = next_node
    print(full_path)
    for i in range(len(full_path) - 1):
        out_graph.add_edge(full_path[i], full_path[i + 1])
    return full_path, out_graph


def main():
    # G = generate_city()
    G = generate_grid_city(5, 5)
    warehouse, customers, remaining_nodes = assign_nodes(G)

    print(f'Warehouse: {warehouse}')
    print(f'Customers: {customers}')

    # print(nearest_neighbor(G, warehouse, customers))
    nn_path, nn_graph = tsp_nn_heuristic(G, warehouse, customers)
    nn_path_weight = nx.path_weight(G, nn_path, 'weight')
    print(f'Nearest neighbor distance: {nn_path_weight}')

    pos = nx.get_node_attributes(G, 'pos')

    fig, (ax1, ax2) = plt.subplots(1, 2)
    nx.draw_networkx_nodes(G, pos, [warehouse], node_color='red', ax=ax1)
    nx.draw_networkx_nodes(G, pos, customers, node_color='blue', ax=ax1)
    nx.draw_networkx_nodes(G, pos, remaining_nodes, node_color='grey', ax=ax1)
    # use this for waxman graph
    # labels = {node: node for node in G.nodes()}
    # use this for grid graph
    labels = {node: i + 1 for i, node in enumerate(G.nodes())}
    nx.draw_networkx_labels(G, pos, labels, ax=ax1)
    nx.draw_networkx_edges(G, pos, G.edges(), ax=ax1)

    nx.draw(nn_graph, nx.get_node_attributes(nn_graph, 'pos'), labels=labels, ax=ax2)
    # nx.draw_networkx_edge_labels(G, pos, {(u, v): round(G.edges[u, v]['weight'], 2) for u, v in G.edges()})
    plt.show()

if __name__ == '__main__':
    main()
