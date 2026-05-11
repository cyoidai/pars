import numpy as np
import random


def generate_sparse_connected_adj(size=10, extra_edges=5):
    # 1. Initialize matrix with zeros
    adj = np.zeros((size, size))

    # 2. Ensure Connectivity (Create a random spanning tree)
    # This connects every node to at least one other node
    nodes = list(range(size))
    random.shuffle(nodes)
    connected_nodes = [nodes[0]]
    remaining_nodes = nodes[1:]

    while remaining_nodes:
        u = random.choice(connected_nodes)
        v = remaining_nodes.pop()
        # Assign a random weight (e.g., between 0.1 and 1.0)
        weight = round(random.uniform(0.1, 1.0), 2)
        adj[u][v] = adj[v][u] = weight
        connected_nodes.append(v)

    # 3. Add extra edges to control "sparseness"
    # Total possible edges in undirected graph is (n*(n-1))/2
    attempts = 0
    added = 0
    while added < extra_edges and attempts < 100:
        u, v = random.sample(range(size), 2)
        if adj[u][v] == 0:
            weight = round(random.uniform(0.1, 1.0), 2)
            adj[u][v] = adj[v][u] = weight
            added += 1
        attempts += 1

    return adj


# Generate a 10x10 matrix
matrix = generate_sparse_connected_adj(10, extra_edges=8)
print(matrix)