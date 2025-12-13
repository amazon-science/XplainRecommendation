"""
RL Environment for Path Ranking

This environment trains an RL agent to rank/select candidate paths extracted by heuristics.
Much easier task than full path generation: rank 10 paths instead of navigating 30K nodes.

Author: RL-GraphRetriever
Date: December 2025
"""

import numpy as np
from typing import List, Dict, Tuple


class PathRankingEnv:
    """
    RL Environment for learning to rank explanation paths.
    
    Task: Given K candidate paths (from heuristics), select the best one(s)
    
    State Space:
        - User embedding (768)
        - Item embedding (768)
        - Path features (10 paths × 64 = 640)
        Total: 2176-dimensional state
    
    Action Space:
        - Discrete: Select which path to use (0 to num_paths-1)
        - Can also output ranking scores
    
    Reward:
        - Path quality (reaches target, similarity, diversity)
        - Simpler than full navigation!
    """
    
    def __init__(
        self,
        data_loader,
        path_extractor,
        num_paths: int = 10
    ):
        """
        Initialize the ranking environment.
        
        Args:
            data_loader: GReferDataLoader instance
            path_extractor: HeuristicPathExtractor instance
            num_paths: Number of candidate paths
        """
        self.data_loader = data_loader
        self.path_extractor = path_extractor
        self.num_paths = num_paths
        
        # Current episode
        self.user_id = None
        self.item_id = None
        self.candidate_paths = []
        
        # State dimension
        embedding_dim = len(data_loader.get_embedding(0))
        path_feature_dim = 64  # Features per path
        self.state_dim = embedding_dim * 2 + (num_paths * path_feature_dim)
        self.action_dim = num_paths  # Select which path
        
        print(f"Initialized PathRankingEnv")
        print(f"  State dim: {self.state_dim}")
        print(f"  Action dim: {self.action_dim} (select 1 of {num_paths} paths)")
    
    def reset(self) -> np.ndarray:
        """Reset for new episode."""
        # Get random user-item pair
        sample = self.data_loader.get_sample_batch(1)[0]
        self.user_id = sample['user_id']
        self.item_id = sample['item_id']
        
        # Extract candidate paths
        self.candidate_paths = self.path_extractor.extract_paths(
            self.user_id,
            self.item_id,
            max_length=7
        )
        
        # Pad if needed
        while len(self.candidate_paths) < self.num_paths:
            self.candidate_paths.append([])
        
        # Get state
        state = self._get_state()
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action (select a path).
        
        Args:
            action: Index of path to select
            
        Returns:
            (next_state, reward, done, info)
        """
        # Get selected path
        if action >= len(self.candidate_paths):
            action = 0  # Default to first path
        
        selected_path = self.candidate_paths[action]
        
        # Compute reward based on path quality
        reward = self._compute_path_quality(selected_path)
        
        # Episode done after single selection
        done = True
        
        info = {
            'selected_path': selected_path,
            'num_candidates': len([p for p in self.candidate_paths if len(p) > 0]),
            'reaches_target': len(selected_path) > 0 and selected_path[-1] == self.item_id
        }
        
        # State doesn't change (single-step task)
        next_state = self._get_state()
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """Get current state."""
        # User and item embeddings
        user_emb = self.data_loader.get_embedding(self.user_id)
        item_emb = self.data_loader.get_embedding(self.item_id)
        
        # Path features
        path_features = []
        for path in self.candidate_paths[:self.num_paths]:
            features = self._encode_path_features(path)
            path_features.append(features)
        
        path_features = np.concatenate(path_features)
        
        # Combine
        state = np.concatenate([user_emb, item_emb, path_features])
        
        return state.astype(np.float32)
    
    def _encode_path_features(self, path: List[int]) -> np.ndarray:
        """Encode path into feature vector."""
        features = np.zeros(64)
        
        if len(path) == 0:
            return features
        
        # Feature 1-10: Path length (one-hot up to 10)
        path_len = min(len(path), 10)
        features[path_len - 1] = 1.0
        
        # Feature 11: Reaches target
        if path[-1] == self.item_id:
            features[10] = 1.0
        
        # Feature 12-43: Mean path embedding (32 dims)
        path_embs = [self.data_loader.get_embedding(n) for n in path]
        mean_emb = np.mean(path_embs, axis=0)
        features[11:43] = mean_emb[:32]
        
        # Feature 44: Node type diversity
        node_types = [1 if n < self.data_loader.num_users else 0 for n in path]
        diversity = len(set(node_types)) / len(node_types)
        features[43] = diversity
        
        # Feature 45-54: Similarity to target (for each hop)
        for i in range(min(10, len(path))):
            node_emb = self.data_loader.get_embedding(path[i])
            target_emb = self.data_loader.get_embedding(self.item_id)
            sim = np.dot(node_emb, target_emb) / (
                np.linalg.norm(node_emb) * np.linalg.norm(target_emb) + 1e-8
            )
            features[44 + i] = (sim + 1) / 2
        
        return features
    
    def _compute_path_quality(self, path: List[int]) -> float:
        """Compute quality score for a path."""
        if len(path) == 0:
            return -10.0
        
        reward = 0.0
        
        # 1. Reaches target? Big reward!
        if path[-1] == self.item_id:
            reward += 20.0
            reward += max(0, 10 - len(path))
        else:
            final_sim = self._compute_similarity(path[-1], self.item_id)
            reward += final_sim * 5.0
            reward -= 5.0
        
        # 2. Path coherence
        if len(path) > 1:
            coherence = 0
            for i in range(len(path) - 1):
                sim = self._compute_similarity(path[i], path[i+1])
                coherence += sim
            coherence /= (len(path) - 1)
            reward += coherence * 2.0
        
        # 3. Diversity
        user_nodes = sum(1 for n in path if n < self.data_loader.num_users)
        item_nodes = len(path) - user_nodes
        if user_nodes > 0 and item_nodes > 0:
            reward += 1.0
        
        return reward
    
    def _compute_similarity(self, node1: int, node2: int) -> float:
        """Compute embedding similarity between nodes."""
        emb1 = self.data_loader.get_embedding(node1)
        emb2 = self.data_loader.get_embedding(node2)
        
        sim = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
        )
        
        return (sim + 1.0) / 2.0


def main():
    """Test the ranking environment."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from data_loader import GReferDataLoader
    from heuristic_path_extractor import HeuristicPathExtractor
    
    print("\n" + "="*60)
    print("Testing PathRankingEnv")
    print("="*60)
    
    # Load data
    loader = GReferDataLoader(dataset_name='yelp')
    loader.load_all(split='trn')
    
    # Create extractor
    extractor = HeuristicPathExtractor(loader.graph, loader, num_paths=10)
    
    # Create ranking environment
    env = PathRankingEnv(loader, extractor, num_paths=10)
    
    # Test episodes
    print("\n" + "="*60)
    print("Testing Path Ranking")
    print("="*60)
    
    for episode in range(3):
        state = env.reset()
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  User {env.user_id} → Item {env.item_id}")
        print(f"  State shape: {state.shape}")
        print(f"  Candidate paths: {len([p for p in env.candidate_paths if len(p) > 0])}")
        
        # Try each path
        print("\n  Path Rewards:")
        for i, path in enumerate(env.candidate_paths):
            if len(path) == 0:
                continue
            
            # Compute reward for this path
            reward = env._compute_path_quality(path)
            reaches = path[-1] == env.item_id
            
            print(f"    Path {i}: Length {len(path)}, Reward {reward:.2f}, Reaches target: {reaches}")
        
        # Select best path (action with highest reward)
        best_action = 0
        best_reward = -float('inf')
        for i, path in enumerate(env.candidate_paths):
            reward = env._compute_path_quality(path)
            if reward > best_reward:
                best_reward = reward
                best_action = i
        
        print(f"\n  Best path: {best_action} with reward {best_reward:.2f}")
        
        # Take action
        _, reward, done, info = env.step(best_action)
        print(f"  Selected path reaches target: {info['reaches_target']}")
    
    print("\n✓ Path ranking environment test completed!")


if __name__ == '__main__':
    main()
