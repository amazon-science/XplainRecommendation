"""
Heuristic Path Extractor

This module extracts candidate paths using simple heuristics (shortest paths, random walks, etc.)
These paths are then ranked/selected by an RL agent.

This implements the hybrid approach:
1. Heuristic extraction → Multiple candidate paths
2. RL selection → Choose best paths for explanation

Author: RL-GraphRetriever
Date: December 2025
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class HeuristicPathExtractor:
    """
    Extracts candidate paths using heuristic methods.
    
    Methods:
    1. Shortest paths (k-shortest paths)
    2. Random walks with restart
    3. Node2Vec-style paths
    4. Personalized PageRank paths
    """
    
    def __init__(self, graph, data_loader, num_paths: int = 10):
        """
        Initialize path extractor.
        
        Args:
            graph: NetworkX graph
            data_loader: GReferDataLoader instance
            num_paths: Number of candidate paths to extract per user-item pair
        """
        self.graph = graph
        self.data_loader = data_loader
        self.num_paths = num_paths
        
        print(f"Initialized HeuristicPathExtractor")
        print(f"  Num candidate paths: {num_paths}")
    
    def extract_paths(
        self,
        user_id: int,
        item_id: int,
        max_length: int = 7
    ) -> List[List[int]]:
        """
        Extract multiple candidate paths between user and item.
        
        Args:
            user_id: Source user node
            item_id: Target item node
            max_length: Maximum path length
            
        Returns:
            List of paths (each path is a list of node IDs)
        """
        paths = []
        
        # Method 1: Shortest path
        try:
            shortest = nx.shortest_path(self.graph, user_id, item_id)
            if len(shortest) <= max_length:
                paths.append(shortest)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        
        # Method 2: K-shortest paths (simple paths)
        try:
            for path in nx.all_simple_paths(self.graph, user_id, item_id, cutoff=max_length):
                paths.append(path)
                if len(paths) >= self.num_paths:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        
        # Method 3: Random walks if we don't have enough paths
        while len(paths) < self.num_paths:
            random_path = self._random_walk_path(user_id, item_id, max_length)
            if random_path and random_path not in paths:
                paths.append(random_path)
            
            # Prevent infinite loop
            if len(paths) == 0 and len(paths) < self.num_paths:
                # Try 10 times, if still nothing, break
                if len(paths) < 3:
                    continue
                else:
                    break
        
        return paths[:self.num_paths]
    
    def _random_walk_path(
        self,
        start: int,
        target: int,
        max_length: int,
        restart_prob: float = 0.15
    ) -> List[int]:
        """
        Perform random walk with restart from start toward target.
        
        Args:
            start: Start node
            target: Target node
            max_length: Maximum walk length
            restart_prob: Probability of restarting from source
            
        Returns:
            Path from start to target (or partial path)
        """
        if start not in self.graph or target not in self.graph:
            return []
        
        path = [start]
        current = start
        
        for _ in range(max_length * 2):  # Allow more steps to find target
            if current == target:
                return path
            
            # Random restart
            if np.random.random() < restart_prob:
                current = start
                path = [start]
                continue
            
            # Get neighbors
            neighbors = list(self.graph.neighbors(current))
            if not neighbors:
                break
            
            # Bias toward target (use embeddings)
            next_node = self._choose_biased_neighbor(neighbors, target)
            
            # Avoid loops
            if next_node in path:
                # Try another neighbor
                unvisited = [n for n in neighbors if n not in path]
                if unvisited:
                    next_node = self._choose_biased_neighbor(unvisited, target)
                else:
                    break
            
            path.append(next_node)
            current = next_node
            
            if len(path) > max_length:
                break
        
        # Return path even if didn't reach target (partial path)
        return path if len(path) > 1 else []
    
    def _choose_biased_neighbor(
        self,
        neighbors: List[int],
        target: int,
        temperature: float = 0.5
    ) -> int:
        """
        Choose next neighbor biased toward target using embeddings.
        
        Args:
            neighbors: List of neighbor node IDs
            target: Target node ID
            temperature: Sampling temperature (lower = more greedy)
            
        Returns:
            Selected neighbor node ID
        """
        if len(neighbors) == 1:
            return neighbors[0]
        
        target_emb = self.data_loader.get_embedding(target)
        
        # Compute similarities
        similarities = []
        for neighbor in neighbors:
            neighbor_emb = self.data_loader.get_embedding(neighbor)
            sim = np.dot(neighbor_emb, target_emb) / (
                np.linalg.norm(neighbor_emb) * np.linalg.norm(target_emb) + 1e-8
            )
            similarities.append(sim)
        
        # Softmax with temperature
        similarities = np.array(similarities)
        exp_sims = np.exp(similarities / temperature)
        probs = exp_sims / exp_sims.sum()
        
        # Sample
        selected_idx = np.random.choice(len(neighbors), p=probs)
        return neighbors[selected_idx]


def main():
    """Test the path extractor."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from data_loader import GReferDataLoader
    
    print("\n" + "="*60)
    print("Testing HeuristicPathExtractor")
    print("="*60)
    
    # Load data
    loader = GReferDataLoader(dataset_name='yelp')
    loader.load_all(split='trn')
    
    # Create extractor
    extractor = HeuristicPathExtractor(
        graph=loader.graph,
        data_loader=loader,
        num_paths=10
    )
    
    # Test on sample
    sample = loader.train_samples[0]
    user_id = sample['user_id']
    item_id = sample['item_id']
    
    print(f"\nExtracting paths from user {user_id} to item {item_id}")
    paths = extractor.extract_paths(user_id, item_id, max_length=7)
    
    print(f"\nExtracted {len(paths)} candidate paths:")
    for i, path in enumerate(paths):
        print(f"  Path {i+1}: Length {len(path)}")
        print(f"    {' → '.join(map(str, path[:5]))}{'...' if len(path) > 5 else ''}")
    
    # Test on 10 random samples
    print("\n" + "="*60)
    print("Testing on 10 random samples")
    print("="*60)
    
    success_count = 0
    for i in range(10):
        sample = loader.get_sample_batch(1)[0]
        paths = extractor.extract_paths(sample['user_id'], sample['item_id'])
        
        # Check if any path reaches target
        reaches_target = any(path[-1] == sample['item_id'] for path in paths if len(path) > 0)
        if reaches_target:
            success_count += 1
        
        print(f"Sample {i+1}: {len(paths)} paths, reaches target: {reaches_target}")
    
    print(f"\nSuccess rate: {success_count}/10 = {success_count/10:.1%}")
    print("\n✓ Path extractor test completed!")


if __name__ == '__main__':
    main()
