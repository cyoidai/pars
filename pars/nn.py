from typing import Iterable, Callable
import networkx as nx
from .utils import euclidean_distance


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


class NearestNeighbor:
    def __init__(self, K: nx.Graph, warehouse):
        self.K = K
        self.warehouse = warehouse

    def run(self, callback: Callable[[list], None] | None=None) -> list:
        route = [self.warehouse]
        remaining_nodes = list(self.K.nodes())
        remaining_nodes.remove(self.warehouse)
        while remaining_nodes:
            next_node, _dist = nearest_neighbor(self.K, route[-1], remaining_nodes)
            route.append(next_node)
            remaining_nodes.remove(next_node)
            if callback:
                callback(route)
        route.append(self.warehouse)
        if callback:
            callback(route)
        return route



# def tsp_nn_heuristic(G: nx.Graph, source, nodes: Iterable) -> list:
#     """
#     Solves a road network TSP using nearest neighbor heuristic for picking the
#     next node to visit and A* search for routing between nodes.
#     """
#     nodes_left = list(nodes)
#     current_node = source
#     full_path = [current_node]
#     while nodes_left:
#         next_node, _dist = nearest_neighbor(G, current_node, nodes_left)
#         path = nx.astar_path(G, current_node, next_node, lambda u,v:euclidean_distance(G, u, v), 'length')
#         full_path += path[1::]
#         nodes_left.remove(next_node)
#         current_node = next_node
#     return full_path
