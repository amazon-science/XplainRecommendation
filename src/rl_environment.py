"""
RL Environment for Graph Retrieval

State: Current user-item graph state, query context, retrieved paths
Action: Select next node/edge to retrieve (from structural or semantic candidates)
Reward: Explanation quality + efficiency penalty
"""

import torch
import dgl
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


class GraphRetrievalEnvironment:
    """
    RL Environment for learning adaptive graph retrieval policy
    
    This environment models the graph retrieval task as an MDP:
    - State: Current graph state, user-item context, retrieved paths
    - Action: Select next node/edge to add to retrieved subgraph
    - Reward: Explanation quality - efficiency penalty
    """
    
    def __init__(
        self,
        graph: dgl.DGLGraph,
        max_steps: int = 10,
        max_path_length: int = 3,
        efficiency_penalty: float = 0.1,
        device: str = "cpu"
    ):
        """
        Initialize environment
        
        Parameters
        ----------
        graph : dgl.DGLGraph
            Full user-item heterogeneous graph
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
        # State includes:
        # - Retrieved subgraph features (graph statistics)
        # - Current step / max_steps
        # - Number of retrieved paths
        # - Coverage metrics
        return 64  # Can be adjusted based on feature engineering
    
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
        # Features for state representation
        features = []
        
        # 1. Progress features
        features.append(self.current_step / self.max_steps)
        features.append(len(self.retrieved_nodes) / self.graph.num_nodes())
        features.append(len(self.retrieved_edges) / self.graph.num_edges())
        features.append(len(self.retrieved_paths) / self.max_steps)
        
        # 2. Graph topology features
        if len(self.retrieved_nodes) > 2:
            # Compute subgraph statistics
            subgraph = self.graph.subgraph(list(self.retrieved_nodes))
            features.append(subgraph.num_nodes() / self.graph.num_nodes())
            features.append(subgraph.num_edges() / max(1, subgraph.num_nodes()))
        else:
            features.extend([0.0, 0.0])
        
        # 3. Path features
        if self.retrieved_paths:
            avg_path_length = np.mean([len(p) for p in self.retrieved_paths])
            features.append(avg_path_length / self.max_path_length)
        else:
            features.append(0.0)
        
        # 4. Coverage features
        # How well we've explored the neighborhood
        src_neighbors = set(self.graph.successors(self.src_nid).tolist())
        tgt_neighbors = set(self.graph.predecessors(self.tgt_nid).tolist())
        
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
            # Outgoing edges
            successors = self.graph.successors(node).tolist()
            for succ in successors:
                if succ not in self.retrieved_nodes:
                    candidates.append((node, succ))
            
            # Incoming edges  
            predecessors = self.graph.predecessors(node).tolist()
            for pred in predecessors:
                if pred not in self.retrieved_nodes:
                    candidates.append((pred, node))
        
        # Semantic candidates: high-similarity nodes (would need embeddings)
        # For now, we'll use structural only
        
        # Limit number of candidates
        if len(candidates) > 50:
            # Sample or rank by some heuristic
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
        # Actual quality reward will be computed at episode end
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
        """Update list of retrieved paths from src to tgt"""
        # Simple path finding in retrieved subgraph
        # For efficiency, only check if last added edge creates new path
        if len(self.retrieved_edges) == 0:
            return
        
        # Use DFS to find paths from src to tgt in retrieved edges
        # This is a simplified version - full implementation would use
        # proper path finding algorithms
        
        # Build adjacency list from retrieved edges
        adj = defaultdict(list)
        for u, v in self.retrieved_edges:
            adj[u].append(v)
        
        # DFS to find paths
        def dfs(node, target, path, visited):
            if node == target:
                self.retrieved_paths.append(path[:])
                return
            
            if len(path) >= self.max_path_length:
                return
            
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, target, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        # Clear and recompute paths
        self.retrieved_paths = []
        visited = {self.src_nid}
        dfs(self.src_nid, self.tgt_nid, [self.src_nid], visited)
        
        # Limit number of paths
        if len(self.retrieved_paths) > 10:
            self.retrieved_paths = self.retrieved_paths[:10]
    
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
    
    # Create bipartite graph
    user_item_edges = []
    for i in range(200):
        u = np.random.randint(0, num_users)
        i_id = np.random.randint(num_users, num_users + num_items)
        user_item_edges.append((u, i_id))
    
    src, dst = zip(*user_item_edges)
    g = dgl.graph((src, dst))
    
    # Initialize environment
    env = GraphRetrievalEnvironment(g, max_steps=5)
    
    # Test episode
    src_nid = 0
    tgt_nid = num_users + 5
    
    state = env.reset(src_nid, tgt_nid)
    print(f"Initial state shape: {state.shape}")
    
    for step in range(5):
        candidates = env.get_candidate_actions()
        if len(candidates) == 0:
            break
        
        action = candidates[0]  # Take first candidate
        next_state, reward, done, info = env.step(action)
        
        print(f"Step {step}: Action {action}, Reward {reward:.3f}, Info {info}")
        
        if done:
            break
    
    context = env.get_retrieved_context()
    print(f"\nFinal context: {len(context['paths'])} paths found")
