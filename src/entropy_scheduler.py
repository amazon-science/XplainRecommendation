# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Adaptive entropy scheduler (extracted from the paper's PPO training loop).

Anneals the entropy coefficient from initial_coef → final_coef over
total_episodes, with two adjustments:
  * bump entropy up when training stalls (recent improvement < 0.01)
  * bump entropy down when training is improving fast (improvement > 0.05)
  * bump entropy up when action-distribution diversity is low (H < 0.5)
"""
import numpy as np


class AdaptiveEntropyScheduler:
    def __init__(self, initial_coef=0.1, final_coef=0.001, total_episodes=500):
        self.initial_coef = initial_coef
        self.final_coef = final_coef
        self.total_episodes = total_episodes
        self.performance_history = []
        self.selection_entropy_history = []

    def get_coefficient(self, episode: int) -> float:
        progress = episode / self.total_episodes
        base_coef = self.initial_coef * (1 - progress) + self.final_coef * progress
        adjustment = 1.0
        if len(self.performance_history) >= 10:
            rec = (np.mean(self.performance_history[-5:])
                   - np.mean(self.performance_history[-10:-5]))
            if rec < 0.01:
                adjustment = 1.5
            elif rec > 0.05:
                adjustment = 0.7
        if len(self.selection_entropy_history) >= 5:
            if np.mean(self.selection_entropy_history[-5:]) < 0.5:
                adjustment *= 1.3
        return float(np.clip(base_coef * adjustment, self.final_coef, self.initial_coef))

    def update(self, performance: float, selection_entropy: float):
        self.performance_history.append(performance)
        self.selection_entropy_history.append(selection_entropy)
