import numpy as np
import yaml
import re
import networkx as nx
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from tsp_env import TSPEnv


def load_matrix_from_file(filepath):
    """
    Parses the matrix from the log file using a more robust string-to-array conversion.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # Isolate the data after the header
    if 'Adjacency Matrix:' not in content:
        raise ValueError("Could not find 'Adjacency Matrix:' in the log file.")

    matrix_str = content.split('Adjacency Matrix:')[-1].strip()

    # Remove brackets and replace newlines/multiple spaces with a single comma
    # This makes the string readable by np.fromstring
    clean_str = matrix_str.replace('[', '').replace(']', '').replace('\n', ' ')

    # Use numpy's built-in string parser (handles spaces and commas automatically)
    flat_matrix = np.fromstring(clean_str, sep=', ')

    # Reshape back into a square matrix (N x N)
    size = int(np.sqrt(len(flat_matrix)))
    return flat_matrix.reshape((size, size))


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Load the same matrix used in training
    try:
        matrix = load_matrix_from_file("output/matrix_log.txt")
        print("--- Loaded Matrix from File ---")
        print(matrix)
    except Exception as e:
        print(f"Error loading matrix: {e}")
        exit()

    # 2. Setup Environment and Model
    goals = config['env_params']['required_cities']
    env = TSPEnv(matrix, must_visit_nodes=goals)

    model_path = config['train_params']['save_path']
    if not model_path.endswith(".zip"):
        model_path += ".zip"

    print(f"Loading model: {model_path}")
    model = MaskablePPO.load(model_path, env=env)

    # 3. Deterministic Test Run
    obs, _ = env.reset()
    done = False
    max_moves = 50
    move_count = 0

    print("Starting test run...")
    while not done and move_count < max_moves:
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        move_count += 1
        done = terminated or truncated

    # 4. Results
    print("-" * 30)
    print(f"Visit History: {env.visited}")
    print(f"Status: {'SUCCESS (Returned to Start)' if terminated else 'FAILED'}")
    print(f"Total Moves: {move_count}")

    # 5. Visualization
    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    pos = nx.spring_layout(G, seed=42)

    # Draw base graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            edge_color='gray', alpha=0.3, arrowsize=15)

    # Highlight the path taken by the agent
    path_edges = list(zip(env.visited, env.visited[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                           edge_color='blue', width=2.5, arrowsize=20)

    # Highlight goal nodes
    nx.draw_networkx_nodes(G, pos, nodelist=goals, node_color='orange')

    plt.title(f"TSP Agent Path (Status: {'Success' if terminated else 'Failed'})")
    plt.show()