import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TSPEnv(gym.Env):
    def __init__(self, adj_matrix, must_visit_nodes):
        super(TSPEnv, self).__init__()
        self.adj_matrix = adj_matrix.astype(np.float32)
        self.num_nodes = len(adj_matrix)
        self.initial_must_visit = set(must_visit_nodes)

        self.action_space = spaces.Discrete(self.num_nodes)

        self.observation_space = spaces.Dict({
            "map": spaces.Box(low=0, high=1, shape=(self.num_nodes, self.num_nodes), dtype=np.float32),
            "checklist": spaces.Box(low=0, high=1, shape=(self.num_nodes,), dtype=np.float32),
            "current_pos": spaces.Box(low=0, high=self.num_nodes, shape=(1,), dtype=np.float32)
        })

    def action_masks(self):
        """Returns a boolean mask where True means weight > 0 (valid path)."""
        return (self.adj_matrix[self.current_node] > 0).astype(bool)

    def _get_obs(self):
        checklist = np.zeros(self.num_nodes, dtype=np.float32)
        for node in self.must_visit:
            checklist[node] = 1.0

        return {
            "map": self.adj_matrix,
            "checklist": checklist,
            "current_pos": np.array([self.current_node], dtype=np.float32)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_node = 0
        self.visited = [0]
        self.must_visit = self.initial_must_visit.copy()
        if 0 in self.must_visit:
            self.must_visit.remove(0)
        self.all_goals_met = False
        return self._get_obs(), {}

    def step(self, action):
        action = int(action)
        reward = 0.0
        terminated = False
        truncated = False

        # 1. Path Check (Fallback)
        if self.adj_matrix[self.current_node][action] == 0:
            reward = -10.0
            return self._get_obs(), float(reward), False, False, {"reason": "Blocked"}

        # 2. Valid Movement
        cost = self.adj_matrix[self.current_node][action]
        self.current_node = action
        self.visited.append(action)

        # 3. Reward Logic
        if action in self.must_visit:
            self.must_visit.remove(action)
            reward = 20.0 - cost
        else:
            reward = -cost

        # 4. Phase Transition
        if len(self.must_visit) == 0 and not self.all_goals_met:
            self.all_goals_met = True
            reward += 25.0

        # 5. Final Termination
        if self.all_goals_met and action == 0:
            terminated = True
            reward += 50.0

        return self._get_obs(), float(reward), terminated, truncated, {}