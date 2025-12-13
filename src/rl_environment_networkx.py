"""
RL Environment for Graph Retrieval using NetworkX

This is a drop-in replacement for rl_environment.py that uses NetworkX instead of DGL.
Works on all platforms without binary compatibility issues.

State: Current user-item graph state, query context, retrieved paths
Action: Select next node/edge to retrieve (from structural or semantic candidates)
Reward: Explanation quality + efficiency penalty
"""

import torch
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


class GraphRetrievalEnvironment:
    """
    RL Environment for learning adaptive graph retrieval policy using NetworkX
    
    This environment models the graph retrieval task as an MDP:
    - State: Current graph state, user-item context, retrieved paths
    - Action: Select next node/edge to add to retrieved subgraph
    - Reward: Explanation quality - efficiency penalty
    """
    
    def __init__(
        self,
        graph: nx.Graph,
        max_steps: int = 10,
        max_path_length: int = 3,
        efficiency_penalty: float = 0.1,
        device: str = "cpu"
    ):
        """
        Initialize environment
        
        Parameters
        ----------
        graph : nx.Graph
            Full user-item graph (can be DiGraph for directed)
        max_steps : int
            Maximum retrieval steps per episode
        max_path_length : int
            Maximum path length to consider
        efficiency_penalty : float
            Penalty per retrieval step (encourages efficiency)
        device : str
            Device for computation
        """
        self.graph = graph
        self.max_steps = max_steps
        self.max_path_length = max_path_length
        self.efficiency_penalty = efficiency_penalty
        self.device = device
        
        # Episode state
        self.current_step = 0
        self.src_nid = None
        self.tgt_nid = None
        self.retrieved_nodes = set()
        self.retrieved_edges = []
        self.retrieved_paths = []
        
        # State representation
        self.state_dim = self._compute_state_dim()
        
    def _compute_state_dim(self) -> int:
        """Compute state dimension"""
        return 64
    
    def reset(self, src_nid: int, tgt_nid: int) -> torch.Tensor:
        """
        Reset environment for new user-item pair
        
        Parameters
        ----------
        src_nid : int
            Source node (user) ID
        tgt_nid : int
            Target node (item) ID
        
        Returns
        -------
        state : torch.Tensor
            Initial state representation
        """
        self.current_step = 0
        self.src_nid = src_nid
        self.tgt_nid = tgt_nid
        self.retrieved_nodes = {src_nid, tgt_nid}
        self.retrieved_edges = []
        self.retrieved_paths = []
        
        return self._get_state()
    
    def _get_state(self) -> torch.Tensor:
        """
        Compute current state representation
        
        Returns
        -------
        state : torch.Tensor
            State vector of dimension state_dim
        """
        features = []
        
        # 1. Progress features
        features.append(self.current_step / self.max_steps)
        features.append(len(self.retrieved_nodes) / self.graph.number_of_nodes())
        features.append(len(self.retrieved_edges) / max(1, self.graph.number_of_edges()))
        features.append(len(self.retrieved_paths) / max(1, self.max_steps))
        
        # 2. Graph topology features
        if len(self.retrieved_nodes) > 2:
            subgraph = self.graph.subgraph(self.retrieved_nodes)
            features.append(subgraph.number_of_nodes() / self.graph.number_of_nodes())
            features.append(subgraph.number_of_edges() / max(1, subgraph.number_of_nodes()))
        else:
            features.extend([0.0, 0.0])
        
        # 3. Path features
        if self.retrieved_paths:
            avg_path_length = np.mean([len(p) for p in self.retrieved_paths])
            features.append(avg_path_length / self.max_path_length)
        else:
            features.append(0.0)
        
        # 4. Coverage features
        src_neighbors = set(self.graph.neighbors(self.src_nid)) if self.src_nid in self.graph else set()
        tgt_neighbors = set(self.graph.neighbors(self.tgt_nid)) if self.tgt_nid in self.graph else set()
        
        src_coverage = len(src_neighbors & self.retrieved_nodes) / max(1, len(src_neighbors))
        tgt_coverage = len(tgt_neighbors & self.retrieved_nodes) / max(1, len(tgt_neighbors))
        
        features.extend([src_coverage, tgt_coverage])
        
        # Pad to state_dim
        while len(features) < self.state_dim:
            features.append(0.0)
        
        features = features[:self.state_dim]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def get_candidate_actions(self) -> List[Tuple[int, int]]:
        """
        Get list of candidate actions (edges to retrieve)
        
        Returns
        -------
        candidates : list of tuples
            List of (src_node, dst_node) tuples representing candidate edges
        """
        candidates = []
        
        # Structural candidates: edges from retrieved nodes
        for node in self.retrieved_nodes:
            if node not in self.graph:
                continue
                
            for neighbor in self.graph.neighbors(node):
                if neighbor not in self.retrieved_nodes:
                    candidates.append((node, neighbor))
        
        # Limit number of candidates
        if len(candidates) > 50:
            candidates = candidates[:50]
        
        return candidates
    
    def step(self, action: Tuple[int, int]) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Take action (retrieve edge) and return next state, reward, done, info
        
        Parameters
        ----------
        action : tuple
            (src_node, dst_node) edge to retrieve
        
        Returns
        -------
        next_state : torch.Tensor
            Next state representation
        reward : float
            Immediate reward
        done : bool
            Whether episode is finished
        info : dict
            Additional information
        """
        src, dst = action
        
        # Add edge and nodes to retrieved set
        self.retrieved_edges.append((src, dst))
        self.retrieved_nodes.add(src)
        self.retrieved_nodes.add(dst)
        
        # Check if we've found a new path
        self._update_paths()
        
        # Increment step
        self.current_step += 1
        
        # Compute immediate reward (negative of efficiency penalty)
        immediate_reward = -self.efficiency_penalty
        
        # Check if done
        done = (
            self.current_step >= self.max_steps or
            len(self.retrieved_paths) >= self.max_steps
        )
        
        # Get next state
        next_state = self._get_state()
        
        info = {
            'num_paths': len(self.retrieved_paths),
            'num_nodes': len(self.retrieved_nodes),
            'num_edges': len(self.retrieved_edges)
        }
        
        return next_state, immediate_reward, done, info
    
    def _update_paths(self):
        """Update list of retrieved paths from src to tgt using NetworkX"""
        try:
            # Create subgraph from retrieved nodes
            subgraph = self.graph.subgraph(self.retrieved_nodes)
            
            # Find all simple paths
            paths = list(nx.all_simple_paths(
                subgraph,
                source=self.src_nid,
                target=self.tgt_nid,
                cutoff=self.max_path_length
            ))
            
            # Limit number of paths
            self.retrieved_paths = paths[:10]
            
        except (nx.NodeNotFound, nx.NetworkXNoPath, nx.NetworkXError):
            self.retrieved_paths = []
    
    def get_retrieved_context(self) -> Dict[str, Any]:
        """
        Get retrieved context for explanation generation
        
        Returns
        -------
        context : dict
            Contains retrieved paths, nodes, edges for LLM
        """
        return {
            'paths': self.retrieved_paths,
            'nodes': list(self.retrieved_nodes),
            'edges': self.retrieved_edges,
            'num_steps': self.current_step
        }
    
    def compute_final_reward(
        self,
        explanation_quality: float,
        reference_quality: Optional[float] = None
    ) -> float:
        """
        Compute final episode reward
        
        Parameters
        ----------
        explanation_quality : float
            Quality score of generated explanation (0-10)
        reference_quality : float, optional
            Quality score of baseline explanation for comparison
        
        Returns
        -------
        reward : float
            Final episode reward
        """
        # Normalize quality to [0, 1]
        quality_reward = explanation_quality / 10.0
        
        # Add efficiency bonus (fewer steps is better)
        efficiency_bonus = (self.max_steps - self.current_step) / self.max_steps
        efficiency_bonus *= 0.2  # Scale factor
        
        # If reference quality provided, compute improvement
        if reference_quality is not None:
            quality_reward = (explanation_quality - reference_quality) / 10.0
        
        total_reward = quality_reward + efficiency_bonus
        
        return total_reward


# Test environment
if __name__ == "__main__":
    # Create dummy graph for testing
    num_users = 100
    num_items = 50
    
    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(num_users + num_items))
    
    # Add random edges
    for _ in range(200):
        u = np.random.randint(0, num_users)
        i = np.random.randint(num_users, num_users + num_items)
        G.add_edge(u, i)
    
    # Initialize environment
    env = GraphRetrievalEnvironment(G, max_steps=5)
    
    # Test episode
    src_nid = 0
    tgt_nid = num_users + 5
    
    state = env.reset(src_nid, tgt_nid)
    print(f"Initial state shape: {state.shape}")
    
    for step in range(5):
        candidates = env.get_candidate_actions()
        if len(candidates) == 0:
            print("No candidates available")
            break
        
        action = candidates[0]  # Take first candidate
        next_state, reward, done, info = env.step(action)
        
        print(f"Step {step}: Action {action}, Reward {reward:.3f}, Info {info}")
        
        if done:
            break
    
    context = env.get_retrieved_context()
    print(f"\nFinal context: {len(context['paths'])} paths found")
