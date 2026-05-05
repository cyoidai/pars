import math
import networkx as nx
import osmnx as ox


def euclidean_distance(G, n1, n2, attr='pos'):
    x1, y1 = G.nodes[n1][attr]
    x2, y2 = G.nodes[n2][attr]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def path_weight_sum(G, routes):
    return sum(nx.path_weight(G, r, "length") for r in routes)


def expand_route(route, routings):
    full = [route[0]]
    for i in range(len(route)-1):
        full += routings[(route[i], route[i+1])][1:]
    return full


def road_network_to_complete_graph(G, nodes):
    K = nx.DiGraph()
    K.add_nodes_from(nodes)

    routings = {}

    for src in nodes:
        dist, paths = nx.single_source_dijkstra(G, src, weight="length")

        for dst in nodes:
            if src == dst or dst not in paths:
                continue

            routings[(src, dst)] = paths[dst]
            K.add_edge(src, dst, length=dist[dst])

    return K, routings


def graph_from_address(address, dist=5000):
    print("Downloading road network...")
    G = ox.graph_from_address(address, dist=dist, network_type="drive")
    G = ox.project_graph(G)

    scc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(scc).copy()

    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()}
    nx.set_node_attributes(G, pos, "pos")

    print("Graph ready:", len(G.nodes()), "nodes")
    return G