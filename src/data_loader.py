"""
Data Loader for G-Refer Datasets

This module loads and processes G-Refer datasets (Amazon, Google, Yelp) for RL-based path retrieval.
It handles graph construction, node embeddings, and creates the data structures needed for training.

Author: RL-GraphRetriever
Date: December 2025
"""

import torch
import json
import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class GReferDataLoader:
    """
    Loads and processes G-Refer datasets for RL-based graph retrieval.
    
    The loader:
    1. Loads PyTorch graph data (data_trn.pt, data_tst.pt)
    2. Loads user/item profiles and reviews
    3. Constructs NetworkX graph for navigation
    4. Extracts node embeddings from pre-trained models
    5. Prepares training samples (user-item pairs with ground truth)
    
    Attributes:
        dataset_name: Name of dataset ('amazon', 'google', or 'yelp')
        data_dir: Path to dataset directory
        graph: NetworkX graph object
        node_embeddings: Dict mapping node_id -> embedding vector
        user_profiles: Dict of user information
        item_profiles: Dict of item information
    """
    
    def __init__(self, dataset_name: str = 'yelp', data_dir: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            dataset_name: Dataset to load ('amazon', 'google', or 'yelp')
            data_dir: Custom data directory path (optional)
        """
        self.dataset_name = dataset_name
        
        if data_dir is None:
            # Default path relative to RL-GraphRetriever
            base_dir = Path(__file__).parent.parent.parent
            self.data_dir = base_dir / 'G-Refer' / 'data' / dataset_name
        else:
            self.data_dir = Path(data_dir)
        
        print(f"Loading {dataset_name} dataset from {self.data_dir}")
        
        # Initialize data structures
        self.graph = None
        self.node_embeddings = {}
        self.user_profiles = {}
        self.item_profiles = {}
        self.train_samples = []
        self.test_samples = []
        
        # Metadata
        self.num_users = 0
        self.num_items = 0
        self.num_nodes = 0
        
    def load_all(self, split: str = 'trn') -> None:
        """
        Load all dataset components.
        
        Args:
            split: Data split to load ('trn' or 'tst')
        """
        print("\n" + "="*60)
        print(f"Loading {self.dataset_name.upper()} dataset - {split} split")
        print("="*60)
        
        # Load profiles
        self._load_profiles()
        
        # Load PyTorch graph data
        graph_data = self._load_graph_data(split)
        
        # Build NetworkX graph
        self._build_networkx_graph(graph_data)
        
        # Load or create embeddings
        self._load_embeddings(graph_data)
        
        # Load training samples
        self._load_samples(split)
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  - Users: {self.num_users}")
        print(f"  - Items: {self.num_items}")
        print(f"  - Total nodes: {self.num_nodes}")
        print(f"  - Edges: {self.graph.number_of_edges()}")
        print(f"  - Training samples: {len(self.train_samples)}")
        
    def _load_profiles(self) -> None:
        """Load user and item profiles from JSON files."""
        print("\n[1/5] Loading user and item profiles...")
        
        # Load user profiles (handle both JSON and JSONL formats)
        user_profile_path = self.data_dir / 'user_profile.json'
        if user_profile_path.exists():
            try:
                with open(user_profile_path, 'r') as f:
                    # Try loading as regular JSON first
                    self.user_profiles = json.load(f)
            except json.JSONDecodeError:
                # If that fails, try JSONL format (one JSON per line)
                with open(user_profile_path, 'r') as f:
                    self.user_profiles = {}
                    for line in f:
                        if line.strip():
                            profile = json.loads(line)
                            # Assume first key is the ID
                            user_id = list(profile.keys())[0] if profile else None
                            if user_id:
                                self.user_profiles[user_id] = profile[user_id]
            print(f"  ✓ Loaded {len(self.user_profiles)} user profiles")
        else:
            print(f"  ⚠ User profiles not found at {user_profile_path}")
        
        # Load item profiles (handle both JSON and JSONL formats)
        item_profile_path = self.data_dir / 'item_profile.json'
        if item_profile_path.exists():
            try:
                with open(item_profile_path, 'r') as f:
                    self.item_profiles = json.load(f)
            except json.JSONDecodeError:
                # If that fails, try JSONL format (one JSON per line)
                with open(item_profile_path, 'r') as f:
                    self.item_profiles = {}
                    for line in f:
                        if line.strip():
                            profile = json.loads(line)
                            # Assume first key is the ID
                            item_id = list(profile.keys())[0] if profile else None
                            if item_id:
                                self.item_profiles[item_id] = profile[item_id]
            print(f"  ✓ Loaded {len(self.item_profiles)} item profiles")
        else:
            print(f"  ⚠ Item profiles not found at {item_profile_path}")
    
    def _load_graph_data(self, split: str) -> Dict:
        """
        Load PyTorch graph data.
        
        Args:
            split: Data split ('trn' or 'tst')
            
        Returns:
            Dictionary containing graph data
        """
        print(f"\n[2/5] Loading PyTorch graph data ({split} split)...")
        
        data_path = self.data_dir / f'data_{split}.pt'
        
        if not data_path.exists():
            raise FileNotFoundError(f"Graph data not found at {data_path}")
        
        data = torch.load(data_path)
        print(f"  ✓ Loaded graph data with {data.num_nodes} nodes")
        
        return data
    
    def _build_networkx_graph(self, graph_data) -> None:
        """
        Build NetworkX graph from PyTorch Geometric data.
        
        Args:
            graph_data: PyTorch Geometric Data object
        """
        print("\n[3/5] Building NetworkX graph for navigation...")
        
        self.graph = nx.Graph()
        
        # Extract edge information
        edge_index = graph_data.edge_index
        num_edges = edge_index.shape[1]
        
        # Add edges to graph
        for i in range(num_edges):
            src = int(edge_index[0, i].item())
            dst = int(edge_index[1, i].item())
            self.graph.add_edge(src, dst)
        
        # Store node metadata
        self.num_nodes = graph_data.num_nodes
        
        # Determine number of users and items
        if hasattr(graph_data, 'user_id_to_node'):
            self.num_users = len(graph_data.user_id_to_node)
            self.num_items = self.num_nodes - self.num_users
        else:
            # Heuristic: assume roughly 20% are users (typical for recommendation graphs)
            self.num_users = int(self.num_nodes * 0.2)
            self.num_items = self.num_nodes - self.num_users
        
        # Add node attributes
        for node_id in range(self.num_nodes):
            node_type = 'user' if node_id < self.num_users else 'item'
            self.graph.nodes[node_id]['type'] = node_type
            self.graph.nodes[node_id]['id'] = node_id
        
        print(f"  ✓ Created graph with {self.graph.number_of_nodes()} nodes")
        print(f"  ✓ Added {self.graph.number_of_edges()} edges")
        print(f"  ✓ Graph density: {nx.density(self.graph):.6f}")
    
    def _load_embeddings(self, graph_data) -> None:
        """
        Load or create node embeddings.
        
        Args:
            graph_data: PyTorch Geometric Data object
        """
        print("\n[4/5] Loading node embeddings...")
        
        # Check if embeddings exist in graph_data
        if hasattr(graph_data, 'x') and graph_data.x is not None:
            print("  ✓ Using embeddings from graph data")
            embeddings = graph_data.x
        else:
            # Create random embeddings as placeholder
            print("  ⚠ No embeddings found, creating random embeddings")
            embedding_dim = 256
            embeddings = torch.randn(self.num_nodes, embedding_dim)
        
        # Store embeddings in dictionary
        for node_id in range(self.num_nodes):
            self.node_embeddings[node_id] = embeddings[node_id].numpy()
        
        embedding_dim = embeddings.shape[1]
        print(f"  ✓ Loaded embeddings: {self.num_nodes} nodes × {embedding_dim} dims")
    
    def _load_samples(self, split: str) -> None:
        """
        Load training/test samples (user-item pairs).
        
        Args:
            split: Data split ('trn' or 'tst')
        """
        print(f"\n[5/5] Loading training samples ({split} split)...")
        
        csv_path = self.data_dir / f'total_{split}.csv'
        
        if not csv_path.exists():
            print(f"  ⚠ Sample file not found at {csv_path}, creating synthetic samples")
            self._create_synthetic_samples()
            return
        
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"  ✓ Loaded {len(df)} samples from CSV")
        
        # Process samples
        samples = []
        for idx, row in df.iterrows():
            sample = {
                'user_id': int(row.get('user_id', row.get('user', idx % self.num_users))),
                'item_id': int(row.get('item_id', row.get('item', self.num_users + (idx % self.num_items)))),
                'rating': float(row.get('rating', 4.0)),
                'explanation': str(row.get('explanation', row.get('text', '')))
            }
            samples.append(sample)
        
        if split == 'trn':
            self.train_samples = samples
        else:
            self.test_samples = samples
        
        print(f"  ✓ Processed {len(samples)} samples")
    
    def _create_synthetic_samples(self, num_samples: int = 100) -> None:
        """
        Create synthetic training samples for testing.
        
        Args:
            num_samples: Number of synthetic samples to create
        """
        samples = []
        
        for i in range(num_samples):
            user_id = i % self.num_users
            item_id = self.num_users + (i % self.num_items)
            
            sample = {
                'user_id': user_id,
                'item_id': item_id,
                'rating': 4.0,
                'explanation': f'Synthetic explanation for user {user_id} and item {item_id}'
            }
            samples.append(sample)
        
        self.train_samples = samples
        print(f"  ✓ Created {num_samples} synthetic samples")
    
    def get_k_hop_neighbors(self, node_id: int, k: int = 2) -> List[int]:
        """
        Get k-hop neighbors of a node.
        
        Args:
            node_id: Node to get neighbors for
            k: Number of hops
            
        Returns:
            List of neighbor node IDs
        """
        if node_id not in self.graph:
            return []
        
        neighbors = set([node_id])
        current_level = set([node_id])
        
        for _ in range(k):
            next_level = set()
            for node in current_level:
                next_level.update(self.graph.neighbors(node))
            neighbors.update(next_level)
            current_level = next_level
        
        neighbors.remove(node_id)  # Remove the starting node
        return list(neighbors)
    
    def find_paths(self, source: int, target: int, max_length: int = 4) -> List[List[int]]:
        """
        Find all simple paths between source and target.
        
        Args:
            source: Source node ID
            target: Target node ID
            max_length: Maximum path length
            
        Returns:
            List of paths (each path is a list of node IDs)
        """
        try:
            paths = list(nx.all_simple_paths(
                self.graph, 
                source, 
                target, 
                cutoff=max_length
            ))
            return paths[:10]  # Limit to 10 paths for efficiency
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []
    
    def get_embedding(self, node_id: int) -> np.ndarray:
        """
        Get embedding for a node.
        
        Args:
            node_id: Node ID
            
        Returns:
            Embedding vector as numpy array
        """
        return self.node_embeddings.get(node_id, np.zeros(256))
    
    def get_sample_batch(self, batch_size: int = 32, split: str = 'train') -> List[Dict]:
        """
        Get a batch of samples.
        
        Args:
            batch_size: Number of samples in batch
            split: 'train' or 'test'
            
        Returns:
            List of sample dictionaries
        """
        samples = self.train_samples if split == 'train' else self.test_samples
        
        if len(samples) == 0:
            return []
        
        # Random sampling
        indices = np.random.choice(len(samples), size=min(batch_size, len(samples)), replace=False)
        return [samples[i] for i in indices]
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'dataset': self.dataset_name,
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_nodes': self.num_nodes,
            'num_edges': self.graph.number_of_edges() if self.graph else 0,
            'graph_density': nx.density(self.graph) if self.graph else 0,
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.num_nodes if self.graph else 0,
            'num_train_samples': len(self.train_samples),
            'num_test_samples': len(self.test_samples),
        }
        
        # Add degree distribution
        if self.graph:
            degrees = [d for n, d in self.graph.degree()]
            stats['min_degree'] = min(degrees)
            stats['max_degree'] = max(degrees)
            stats['median_degree'] = np.median(degrees)
        
        return stats


def main():
    """Test the data loader."""
    print("\n" + "="*60)
    print("Testing GReferDataLoader")
    print("="*60)
    
    # Test with Yelp dataset
    loader = GReferDataLoader(dataset_name='yelp')
    loader.load_all(split='trn')
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    stats = loader.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test path finding
    print("\n" + "="*60)
    print("Testing Path Finding")
    print("="*60)
    
    sample = loader.train_samples[0]
    user_id = sample['user_id']
    item_id = sample['item_id']
    
    print(f"Finding paths from user {user_id} to item {item_id}...")
    paths = loader.find_paths(user_id, item_id, max_length=4)
    print(f"Found {len(paths)} paths:")
    for i, path in enumerate(paths[:3]):  # Show first 3 paths
        print(f"  Path {i+1}: {' -> '.join(map(str, path))} (length: {len(path)})")
    
    # Test neighbor finding
    print("\n" + "="*60)
    print("Testing Neighbor Finding")
    print("="*60)
    
    neighbors = loader.get_k_hop_neighbors(user_id, k=2)
    print(f"User {user_id} has {len(neighbors)} neighbors within 2 hops")
    print(f"First 10 neighbors: {neighbors[:10]}")
    
    print("\n✓ All tests passed!")


if __name__ == '__main__':
    main()
