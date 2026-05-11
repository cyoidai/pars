from typing import Any, Iterable
import itertools
import math
import random
import textwrap
import networkx as nx
import osmnx as ox


def euclidean_distance(G: nx.Graph, n1, n2, attr='pos') -> float:
    x1, y1 = G.nodes[n1][attr]
    x2, y2 = G.nodes[n2][attr]
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y2 - y1, 2))


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


def cluster_graph_sweep(
    G: nx.Graph,
    max_cluster_size: int,
    center: tuple[int | float, int | float]=(0, 0),
    node_list: Iterable | None=None, pos_attr: str='pos'
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
    # nx.set_node_attributes(G, ref_angles, 'ref_angle')
    nodes_sorted = sorted(node_list, key=lambda n:ref_angles[n])
    for cluster in itertools.batched(nodes_sorted, max_cluster_size):
        clusters.append(list(cluster))
    return clusters


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


def graph_from_address(address: str, dist: int=10_000) -> nx.MultiDiGraph:
    G = ox.graph_from_address(address, dist, network_type='drive')
    G = ox.project_graph(G)
    # ensure every node can route to every other node
    max_scc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(max_scc)
    pos = { node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes() }
    nx.set_node_attributes(G, pos, 'pos')
    return G


def graph_from_place(query: str, dist: int=10_000) -> nx.MultiDiGraph:
    G = ox.graph_from_address(query, dist, network_type='drive')
    G = ox.project_graph(G)
    # ensure every node can route to every other node
    max_scc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(max_scc)
    pos = { node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes() }
    nx.set_node_attributes(G, pos, 'pos')
    return G


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
