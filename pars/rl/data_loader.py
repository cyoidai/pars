import yaml
import numpy as np
import random
import os
from adjacencyMatrix import generate_sparse_connected_adj
from adjacencyMatrixManual import get_manual_data


def get_consistent_data():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set seeds for consistent random generation
    random.seed(config['env_params']['seed'])
    np.random.seed(config['env_params']['seed'])

    if config['env_params']['mode'] == "manual":
        matrix, goals = get_manual_data()
    else:
        matrix = generate_sparse_connected_adj(
            size=config['env_params']['size'],
            extra_edges=config['env_params']['extra_edges']
        )
        goals = config['env_params']['required_cities']

    # --- Simplified Output: Single Overwritten File ---
    if not os.path.exists("output"):
        os.makedirs("output")

    filename = "output/matrix_log.txt"

    with open(filename, "w") as f:
        f.write(f"Current Mode: {config['env_params']['mode']}\n")
        f.write(f"Goals: {goals}\n")
        f.write("-" * 20 + "\n")
        f.write("Adjacency Matrix:\n")
        # Ensure the matrix prints fully without truncation
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        f.write(np.array2string(matrix, separator=', '))

    print(f"Matrix log updated at: {filename}")
    # -------------------------------------------------

    return matrix, goals, config