import itertools
import math
from typing import Any
import networkx as nx
import osmnx as ox


def euclidean_distance(G: nx.Graph, n1, n2, attr='pos') -> float:
    x1, y1 = G.nodes[n1][attr]
    x2, y2 = G.nodes[n2][attr]
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y2 - y1, 2))


def path_weight_sum(G: nx.Graph, routes: list[list[Any]], weight_attr: str='length') -> float:
    total_distance = 0
    for route in routes:
        total_distance += nx.path_weight(G, route, weight_attr)
    return total_distance


def road_network_to_complete_graph(G: nx.MultiDiGraph, nodes: list, weight_attr: str='length') -> tuple[nx.DiGraph, dict[tuple[Any, Any], list[Any]]]:
    """
    Converts a road network graph into a complete graph which is usually given
    as input to solving a TSP/VRP.
    """
    K = nx.DiGraph()
    nodes_with_data = [(node, data) for node, data in G.nodes(data=True) if node in nodes]
    K.add_nodes_from(nodes_with_data)
    routings = {}
    for source in nodes:
        distances, paths = nx.single_source_dijkstra(G, source, weight=weight_attr)
        for target in nodes:
            if source == target:
                continue
            routings[(source, target)] = paths[target]
            K.add_edges_from([(source, target, {weight_attr: distances[target]})])
    return K, routings


def road_network_to_distance_map(G: nx.MultiDiGraph, nodes: list, weight_attr: str='length') -> tuple[list, dict[tuple[Any, Any], int | float], dict[tuple[Any, Any], list[Any]]]:
    """
    Converts a road network graph into a distance map which is usually given
    as input to solving a TSP/VRP.
    """
    routings: dict[tuple[Any, Any], list[Any]] = {}
    distances: dict[tuple[Any, Any], int | float] = {}
    for source in nodes:
        local_distances, paths = nx.single_source_dijkstra(G, source, weight=weight_attr)
        for target in nodes:
            if source == target:
                continue
            routings[(source, target)] = paths[target]
            distances[(source, target)] = local_distances[target]
    return list(G.nodes().keys()), distances, routings


def expand_route(route: list[Any], routings: dict[tuple[Any, Any], list[Any]]) -> list[Any]:
    """
    Converts a route over a complete graph to a route over a road network graph.
    """
    if len(route) < 2:
        raise ValueError('A route requires at least two nodes')
    expanded_route = [route[0]]
    for i in range(len(route) - 1):
        expanded_segment = routings[(route[i], route[i+1])][1::]
        expanded_route.extend(expanded_segment)
    return expanded_route
