from typing import Callable
import networkx as nx
import numpy as np
import os
import multiprocessing
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from .tsp_env import TSPEnv
import matplotlib.pyplot as plt


def make_env(matrix, goals):
    """
    Wrapper to create an environment instance for a subprocess.
    """
    def _init():
        # Each worker gets the exact same matrix object passed from the main process
        return TSPEnv(matrix, must_visit_nodes=goals)
    return _init



class ReinforcementLearningTSP:
    """
    Mostly a wrapper class handling the mapping of data to and from this RL
    package with the rest of PARS
    """
    def __init__(self, K: nx.Graph, warehouse, config):
        self.K = K
        self.matrix = nx.to_numpy_array(K, weight='length')
        self.matrix = self.matrix / np.max(self.matrix)
        self.warehouse = warehouse

        self.config = config
        self.train_params = config['train_params']
        self.ppo_params = config['ppo_params']

        self.goals = [i for i in range(0, len(K.nodes))]

    def train(self):

        core_count = self.train_params.get('num_cpu', 8)
        env = SubprocVecEnv([make_env(self.matrix, self.goals) for _ in range(core_count)])
        # 3. Configure Neural Network
        policy_kwargs = dict(
            net_arch=dict(
                pi=[self.config['network_arch']['num_neurons']] * self.config['network_arch']['num_layers'],
                vf=[self.config['network_arch']['num_neurons']] * self.config['network_arch']['num_layers']
            )
        )

        # 4. Initialize MaskablePPO
        # Checks config for tensorboard to avoid the ImportError
        use_tb = self.train_params.get('use_tensorboard', False)

        self.model = MaskablePPO(
            env=env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.train_params['log_dir'] if use_tb else None,
            **self.ppo_params
        )

        print(f"Starting training for {self.train_params['total_timesteps']} steps...")
        try:
            self.model.learn(total_timesteps=self.train_params['total_timesteps'])

            # 6. Save results
            self.model.save(self.train_params['save_path'])
            print(f"Success! Model saved as: {self.train_params['save_path']}")

        except KeyboardInterrupt:
            print("\nManual stop detected. Saving model before exit...")
            self.model.save(f"{self.train_params['save_path']}_interrupted")
        finally:
            env.close()

    def evaluate(self) -> list:
        # 2. Setup Environment and Model
        env = TSPEnv(self.matrix, must_visit_nodes=self.goals)

        model_path = self.config['train_params']['save_path']
        if not model_path.endswith(".zip"):
            model_path += ".zip"

        print(f"Loading model: {model_path}")
        model = MaskablePPO.load(model_path, env=env)

        # 3. Deterministic Test Run
        obs, _ = env.reset()
        done = False
        max_moves = len(self.K.nodes) * 1.50
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

        if not terminated:
            raise RuntimeError('terminated')

        # Convert matrix indices back to graph node ids
        route = []
        node_list = list(self.K.nodes.keys())
        for node in env.visited:
            route.append(node_list[node])
        return route

        # # 5. Visualization
        # G = nx.from_numpy_array(self.matrix, create_using=nx.DiGraph)
        # pos = nx.spring_layout(G, seed=42)

        # # Draw base graph
        # nx.draw(G, pos, with_labels=True, node_color='lightblue',
        #         edge_color='gray', alpha=0.3, arrowsize=15)

        # # Highlight the path taken by the agent
        # path_edges = list(zip(env.visited, env.visited[1:]))
        # nx.draw_networkx_edges(G, pos, edgelist=path_edges,
        #                     edge_color='blue', width=2.5, arrowsize=20)

        # # Highlight goal nodes
        # nx.draw_networkx_nodes(G, pos, nodelist=self.goals, node_color='orange')

        # plt.title(f"TSP Agent Path (Status: {'Success' if terminated else 'Failed'})")
        # plt.show()
