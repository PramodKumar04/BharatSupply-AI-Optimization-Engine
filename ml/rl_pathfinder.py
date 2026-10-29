"""
Reinforcement Learning Path-Finding and MCMC Speed Analysis.
Implements Framework 2: Q-Learning grid path optimization and Metropolis-Hastings MCMC
speed estimation under dynamic marine hazard and weather severity.
"""

import math
from typing import List, Tuple
import numpy as np

class MaritimeRLPathFinder:
    def __init__(self, grid_size: int = 8, episodes: int = 120, alpha: float = 0.30, gamma: float = 0.95):
        self.grid_size = grid_size
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.actions = [(0, 1), (1, 0), (1, 1), (-1, 1), (1, -1)]

    def build_cost_grid(
        self,
        weather_severity: float,
        traffic_density: float,
        chokepoint_restriction: float,
        piracy_risk: float,
        war_risk: float = 0.1,
        seed: int = 7,
    ) -> np.ndarray:
        rng = np.random.RandomState(seed)
        N = self.grid_size
        grid = np.ones((N, N)) * 0.1
        for r in range(N):
            for c in range(N):
                diag_dist = abs(r - c) / float(N)
                choke_zone = chokepoint_restriction if (N // 3 <= r <= 2 * N // 3 and N // 3 <= c <= 2 * N // 3) else 0.0
                port_zone = traffic_density if (r >= N - 2 and c >= N - 2) else 0.0
                weather_cell = weather_severity * (0.5 + 0.5 * math.sin(r + c))
                grid[r, c] = (
                    0.22 * diag_dist
                    + 0.28 * weather_cell
                    + 0.20 * choke_zone
                    + 0.15 * port_zone
                    + 0.15 * (piracy_risk + war_risk)
                    + rng.uniform(0, 0.02)
                )
        grid[N - 1, N - 1] = 0.0
        return grid

    def train_and_find_path(self, cost_grid: np.ndarray) -> Tuple[List[Tuple[int, int]], float]:
        N = self.grid_size
        q_table = np.zeros((N, N, len(self.actions)))
        for ep in range(self.episodes):
            r, c = 0, 0
            eps = max(0.05, 0.4 * (0.96 ** ep))
            steps = 0
            while (r, c) != (N - 1, N - 1) and steps < N * 3:
                steps += 1
                if np.random.rand() < eps:
                    a_idx = np.random.randint(len(self.actions))
                else:
                    a_idx = int(np.argmax(q_table[r, c]))
                dr, dc = self.actions[a_idx]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < N and 0 <= nc < N):
                    reward = -5.0
                    nr, nc = r, c
                elif (nr, nc) == (N - 1, N - 1):
                    reward = 20.0
                else:
                    reward = -(cost_grid[nr, nc] * 4.0 + 0.5 * math.hypot(dr, dc))
                q_table[r, c, a_idx] += self.alpha * (reward + self.gamma * np.max(q_table[nr, nc]) - q_table[r, c, a_idx])
                r, c = nr, nc

        path = [(0, 0)]
        r, c = 0, 0
        total_hazard = 0.0
        visited = {(0, 0)}
        for _ in range(N * 2):
            if (r, c) == (N - 1, N - 1):
                break
            moved = False
            for a_idx in np.argsort(q_table[r, c])[::-1]:
                dr, dc = self.actions[int(a_idx)]
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N and (nr, nc) not in visited:
                    path.append((nr, nc))
                    visited.add((nr, nc))
                    total_hazard += cost_grid[nr, nc]
                    r, c = nr, nc
                    moved = True
                    break
            if not moved:
                r, c = min(N - 1, r + 1), min(N - 1, c + 1)
                path.append((r, c))
        return path, total_hazard / max(1, len(path))

    @staticmethod
    def metropolis_hastings_speed(
        design_speed_knots: float,
        mean_hazard: float,
        weather_severity: float,
        n_samples: int = 250,
    ) -> float:
        target_mu = max(6.5, design_speed_knots * (1.0 - 0.32 * mean_hazard - 0.22 * weather_severity))
        target_sigma = 1.1 + 0.8 * weather_severity
        samples = []
        current = target_mu
        rng = np.random.RandomState(101)
        for _ in range(n_samples):
            proposal = current + rng.normal(0, 0.9)
            if 4.0 <= proposal <= design_speed_knots * 1.1:
                log_p_curr = -0.5 * ((current - target_mu) / target_sigma) ** 2
                log_p_prop = -0.5 * ((proposal - target_mu) / target_sigma) ** 2
                if math.log(rng.uniform(1e-9, 1.0)) < (log_p_prop - log_p_curr):
                    current = proposal
            samples.append(current)
        return float(np.mean(samples[60:]))

