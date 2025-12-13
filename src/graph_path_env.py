"""
RL Environment for Graph Path Retrieval

This module implements a gym-style environment for training RL agents to retrieve
explanation paths in user-item graphs. The agent learns to navigate the graph
to find high-quality paths that can explain recommendations.

Author: RL-GraphRetriever
Date: December 2025
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from collections import deque


class GraphPathRetrievalEnv:
    """
    RL Environment for learning to retrieve explanation paths in graphs.
    
    The environment simulates navigating a user-item graph to find paths that
    explain why a user might like an item. The agent starts at the user node
    and must navigate to the item node, collecting relevant intermediate nodes.
    
    State Space:
        - Current node embedding (256)
        - Target item embedding (256)
        - Path history encoding (128)
        - Available actions mask (50)
        Total: 690-dimensional state
    
    Action Space:
        - Discrete: Select next node to visit from valid neighbors
        - Special action: STOP (terminate path)
    
    Reward:
        - Path relevance: How well the path explains the user-item connection
        - Efficiency: Shorter paths preferred
        - Diversity: Multiple diverse paths encouraged
    
    Episode:
        - Starts at user node
        - Agent selects next nodes to visit
        - Terminates when reaching item or max steps
        - Returns collected path
    """
    
    def __init__(
        self,
        data_loader,
        max_hops: int = 4,
        max_neighbors: int = 10,  # Reduced from 50 to 10 for easier learning
        reward_scale: float = 1.0,
        dataset_name: str = 'yelp'
    ):
        """
        Initialize the environment.
        
        Args:
            data_loader: GReferDataLoader instance
            max_hops: Maximum number of hops allowed
            max_neighbors: Maximum neighbors to consider per node
            reward_scale: Scaling factor for rewards
            dataset_name: Name of dataset (affects reward function)
        """
        self.data_loader = data_loader
        self.graph = data_loader.graph
        self.max_hops = max_hops
        self.max_neighbors = max_neighbors
        self.reward_scale = reward_scale
        self.dataset_name = dataset_name
        
        # Episode state
        self.current_node = None
        self.target_node = None
        self.visited_nodes = set()
        self.current_path = []
        self.step_count = 0
        
        # User and item for current episode
        self.user_id = None
        self.item_id = None
        
        # Get actual embedding dimension from data
        sample_node = 0
        sample_emb = self.data_loader.get_embedding(sample_node)
        self.embedding_dim = len(sample_emb)
        
        # State and action dimensions
        self.path_history_dim = min(128, self.embedding_dim)  # Limit to avoid huge states
        self.state_dim = (
            self.embedding_dim * 2 +  # current + target
            self.path_history_dim +    # path history
            self.max_neighbors         # action mask
        )
        self.action_dim = self.max_neighbors + 1  # +1 for STOP action
        
        # STOP action index
        self.STOP_ACTION = self.max_neighbors
        
        # Dataset-specific parameters
        self._set_dataset_params()
        
        print(f"Initialized GraphPathRetrievalEnv for {dataset_name}")
        print(f"  State dim: {self.state_dim}")
        print(f"  Action dim: {self.action_dim}")
        print(f"  Max hops: {self.max_hops}")
    
    def _set_dataset_params(self):
        """Set dataset-specific reward parameters."""
        if self.dataset_name == 'yelp':
            self.social_weight = 0.15
            self.efficiency_weight = 0.25
        elif self.dataset_name == 'amazon':
            self.coverage_weight = 0.10  # Reward visiting rare nodes
            self.efficiency_weight = 0.20
        elif self.dataset_name == 'google':
            self.spatial_weight = 0.15
            self.efficiency_weight = 0.30
        else:
            self.efficiency_weight = 0.25
    
    def reset(self, user_id: Optional[int] = None, item_id: Optional[int] = None) -> np.ndarray:
        """
        Reset the environment for a new episode.
        
        Args:
            user_id: User node to start from (random if None)
            item_id: Target item node (random if None)
            
        Returns:
            Initial state observation
        """
        # Sample user-item pair if not provided
        if user_id is None or item_id is None:
            sample = self.data_loader.get_sample_batch(batch_size=1)[0]
            self.user_id = sample['user_id']
            self.item_id = sample['item_id']
        else:
            self.user_id = user_id
            self.item_id = item_id
        
        # Initialize episode state
        self.current_node = self.user_id
        self.target_node = self.item_id
        self.visited_nodes = {self.user_id}
        self.current_path = [self.user_id]
        self.step_count = 0
        
        # Get initial state
        state = self._get_state()
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action index (0 to max_neighbors-1 for move, max_neighbors for STOP)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.step_count += 1
        
        # Check if STOP action
        if action == self.STOP_ACTION:
            reward = self._compute_reward(terminated=True)
            next_state = self._get_state()
            done = True
            info = {
                'path': self.current_path.copy(),
                'path_length': len(self.current_path),
                'reached_target': self.current_node == self.target_node,
                'stopped_early': True
            }
            return next_state, reward, done, info
        
        # Get valid neighbors
        valid_neighbors = self._get_valid_neighbors()
        
        # Invalid action check
        if action >= len(valid_neighbors):
            # Invalid action: stay at current node, negative reward
            reward = -0.5
            next_state = self._get_state()
            done = False
            info = {
                'path': self.current_path.copy(),
                'path_length': len(self.current_path),
                'invalid_action': True
            }
            return next_state, reward, done, info
        
        # Move to selected neighbor
        next_node = valid_neighbors[action]
        self.current_node = next_node
        self.visited_nodes.add(next_node)
        self.current_path.append(next_node)
        
        # Check if reached target
        reached_target = (self.current_node == self.target_node)
        
        # Check if max hops reached
        max_hops_reached = (self.step_count >= self.max_hops)
        
        # Episode done conditions
        done = reached_target or max_hops_reached
        
        # Compute reward
        reward = self._compute_reward(
            reached_target=reached_target,
            terminated=done
        )
        
        # Get next state
        next_state = self._get_state()
        
        # Info dictionary
        info = {
            'path': self.current_path.copy(),
            'path_length': len(self.current_path),
            'reached_target': reached_target,
            'max_hops_reached': max_hops_reached
        }
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state observation.
        
        Returns:
            State vector as numpy array
        """
        # Current node embedding
        current_emb = self.data_loader.get_embedding(self.current_node)
        
        # Target node embedding
        target_emb = self.data_loader.get_embedding(self.target_node)
        
        # Path history encoding (simple: mean of visited nodes)
        if len(self.visited_nodes) > 1:
            visited_embs = [self.data_loader.get_embedding(n) for n in list(self.visited_nodes)[1:]]
            path_history = np.mean(visited_embs, axis=0)
            # Reduce dimensionality for path history
            path_history = path_history[:self.path_history_dim]
        else:
            path_history = np.zeros(self.path_history_dim)
        
        # Valid actions mask
        valid_neighbors = self._get_valid_neighbors()
        action_mask = np.zeros(self.max_neighbors)
        action_mask[:len(valid_neighbors)] = 1.0
        
        # Concatenate all components
        state = np.concatenate([
            current_emb,
            target_emb,
            path_history,
            action_mask
        ])
        
        return state.astype(np.float32)
    
    def _get_valid_neighbors(self) -> List[int]:
        """
        Get valid neighbors that can be visited.
        
        Returns:
            List of valid neighbor node IDs
        """
        if self.current_node not in self.graph:
            return []
        
        # Get all neighbors
        all_neighbors = list(self.graph.neighbors(self.current_node))
        
        # Filter out visited nodes
        unvisited = [n for n in all_neighbors if n not in self.visited_nodes]
        
        # If no unvisited neighbors, allow revisiting (with penalty)
        if len(unvisited) == 0:
            unvisited = all_neighbors
        
        # Rank by relevance to target
        ranked = self._rank_neighbors_by_relevance(unvisited)
        
        # Return top-k neighbors
        return ranked[:self.max_neighbors]
    
    def _rank_neighbors_by_relevance(self, neighbors: List[int]) -> List[int]:
        """
        Rank neighbors by relevance to target node.
        
        Args:
            neighbors: List of neighbor node IDs
            
        Returns:
            Ranked list of neighbor node IDs
        """
        if len(neighbors) == 0:
            return []
        
        target_emb = self.data_loader.get_embedding(self.target_node)
        
        # Compute cosine similarity to target
        scores = []
        for neighbor in neighbors:
            neighbor_emb = self.data_loader.get_embedding(neighbor)
            # Cosine similarity
            similarity = np.dot(neighbor_emb, target_emb) / (
                np.linalg.norm(neighbor_emb) * np.linalg.norm(target_emb) + 1e-8
            )
            scores.append(similarity)
        
        # Sort by score (descending)
        ranked_indices = np.argsort(scores)[::-1]
        ranked = [neighbors[i] for i in ranked_indices]
        
        return ranked
    
    def _compute_reward(
        self,
        reached_target: bool = False,
        terminated: bool = False
    ) -> float:
        """
        Compute reward for current state.
        
        IMPROVED reward design with dense feedback:
        1. Target reached: Large positive reward
        2. Progress toward target: Reward for getting closer (dense)
        3. Path efficiency: Small penalty for long paths
        4. Dataset-specific bonuses
        
        Args:
            reached_target: Whether target node was reached
            terminated: Whether episode terminated
            
        Returns:
            Scalar reward value
        """
        reward = 0.0
        
        # 1. Target reached - BIG reward!
        if reached_target:
            # Bonus for reaching target, extra bonus for shorter paths
            path_length = len(self.current_path)
            reward += 20.0 - (0.5 * path_length)  # 20 points minus path length penalty
            return reward * self.reward_scale
        
        # 2. Progress reward - DENSE feedback for getting closer
        if len(self.current_path) > 1:
            # Compare progress from previous step
            prev_node = self.current_path[-2]
            prev_similarity = self._compute_similarity(prev_node, self.target_node)
            current_similarity = self._compute_similarity(self.current_node, self.target_node)
            
            # Reward improvement (getting closer)
            progress_delta = (current_similarity - prev_similarity) * 10.0
            reward += progress_delta
        
        # 3. Small constant reward for each step (encourages exploration)
        reward += 0.1
        
        # 4. Moderate penalty for termination without reaching target
        if terminated and not reached_target:
            reward -= 5.0
        
        # 5. Dataset-specific bonuses
        dataset_bonus = self._compute_dataset_bonus()
        reward += dataset_bonus
        
        # Scale reward
        reward *= self.reward_scale
        
        return reward
    
    def _compute_similarity(self, node1: int, node2: int) -> float:
        """
        Compute embedding similarity between two nodes.
        
        Args:
            node1: First node ID
            node2: Second node ID
            
        Returns:
            Cosine similarity (0 to 1)
        """
        emb1 = self.data_loader.get_embedding(node1)
        emb2 = self.data_loader.get_embedding(node2)
        
        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
        )
        
        # Normalize to [0, 1]
        return (similarity + 1.0) / 2.0
    
    def _compute_progress(self) -> float:
        """
        Compute progress toward target.
        
        Uses embedding similarity and graph distance.
        
        Returns:
            Progress score (0 to 1)
        """
        current_emb = self.data_loader.get_embedding(self.current_node)
        target_emb = self.data_loader.get_embedding(self.target_node)
        
        # Cosine similarity
        similarity = np.dot(current_emb, target_emb) / (
            np.linalg.norm(current_emb) * np.linalg.norm(target_emb) + 1e-8
        )
        
        # Normalize to [0, 1]
        progress = (similarity + 1.0) / 2.0
        
        return progress
    
    def _compute_dataset_bonus(self) -> float:
        """
        Compute dataset-specific reward bonus.
        
        Returns:
            Bonus reward value
        """
        bonus = 0.0
        
        if self.dataset_name == 'yelp':
            # Bonus for paths through social connections
            # (simplified: check if current node is a user)
            if self.current_node < self.data_loader.num_users:
                bonus += self.social_weight * 0.5
        
        elif self.dataset_name == 'amazon':
            # Bonus for visiting rare nodes
            degree = self.graph.degree(self.current_node)
            if degree < 10:  # Rare node
                bonus += self.coverage_weight * 0.5
        
        elif self.dataset_name == 'google':
            # Bonus for spatial coherence (simplified)
            # Would need actual location data for full implementation
            bonus += 0.0
        
        return bonus
    
    def render(self, mode: str = 'human') -> None:
        """
        Render the current state.
        
        Args:
            mode: Rendering mode
        """
        print("\n" + "="*50)
        print(f"Episode Step: {self.step_count}/{self.max_hops}")
        print(f"Current Node: {self.current_node} (type: {self.graph.nodes[self.current_node]['type']})")
        print(f"Target Node: {self.target_node}")
        print(f"Path Length: {len(self.current_path)}")
        print(f"Path: {' -> '.join(map(str, self.current_path))}")
        print(f"Visited Nodes: {len(self.visited_nodes)}")
        
        valid_neighbors = self._get_valid_neighbors()
        print(f"Valid Neighbors: {len(valid_neighbors)} (showing first 5)")
        for i, neighbor in enumerate(valid_neighbors[:5]):
            node_type = self.graph.nodes[neighbor]['type']
            print(f"  {i}: Node {neighbor} ({node_type})")
        print("="*50)
    
    def get_path_quality_metrics(self, path: List[int]) -> Dict[str, float]:
        """
        Compute quality metrics for a path.
        
        Args:
            path: List of node IDs forming a path
            
        Returns:
            Dictionary of quality metrics
        """
        if len(path) < 2:
            return {
                'length': len(path),
                'avg_similarity': 0.0,
                'diversity': 0.0,
                'coverage': 0.0
            }
        
        # Path length
        path_length = len(path)
        
        # Average embedding similarity along path
        similarities = []
        for i in range(len(path) - 1):
            emb1 = self.data_loader.get_embedding(path[i])
            emb2 = self.data_loader.get_embedding(path[i+1])
            sim = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
            )
            similarities.append(sim)
        avg_similarity = np.mean(similarities)
        
        # Node type diversity
        node_types = [self.graph.nodes[n]['type'] for n in path]
        diversity = len(set(node_types)) / len(node_types)
        
        # Coverage (unique nodes)
        coverage = len(set(path)) / len(path)
        
        return {
            'length': path_length,
            'avg_similarity': avg_similarity,
            'diversity': diversity,
            'coverage': coverage
        }


def main():
    """Test the environment."""
    from data_loader import GReferDataLoader
    
    print("\n" + "="*60)
    print("Testing GraphPathRetrievalEnv")
    print("="*60)
    
    # Load data
    loader = GReferDataLoader(dataset_name='yelp')
    loader.load_all(split='trn')
    
    # Create environment
    env = GraphPathRetrievalEnv(
        data_loader=loader,
        max_hops=4,
        dataset_name='yelp'
    )
    
    # Test one episode with random actions
    print("\n" + "="*60)
    print("Testing Episode with Random Actions")
    print("="*60)
    
    state = env.reset()
    print(f"\nInitial state shape: {state.shape}")
    print(f"Starting from user {env.user_id} to item {env.item_id}")
    
    done = False
    total_reward = 0
    step = 0
    
    while not done and step < 10:
        # Render current state
        env.render()
        
        # Take random action
        valid_neighbors = env._get_valid_neighbors()
        if len(valid_neighbors) > 0:
            action = np.random.randint(0, min(len(valid_neighbors), env.action_dim))
        else:
            action = env.STOP_ACTION
        
        # Step
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        
        print(f"\nAction: {action}, Reward: {reward:.3f}, Done: {done}")
        
        if done:
            print(f"\n{'='*50}")
            print(f"Episode finished!")
            print(f"Total reward: {total_reward:.3f}")
            print(f"Path: {info['path']}")
            print(f"Reached target: {info['reached_target']}")
            
            # Compute path quality
            quality = env.get_path_quality_metrics(info['path'])
            print(f"\nPath Quality Metrics:")
            for key, value in quality.items():
                print(f"  {key}: {value:.4f}")
    
    print("\n✓ Environment test completed!")


if __name__ == '__main__':
    main()
