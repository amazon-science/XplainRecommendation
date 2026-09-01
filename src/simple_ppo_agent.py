# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Simplified PPO Agent for Graph Path Retrieval

This module implements a lightweight PPO (Proximal Policy Optimization) agent
for learning graph navigation policies. It's designed to be simple, readable,
and effective for the path retrieval task.

Author: RL-GraphRetriever
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from collections import deque


class PolicyNetwork(nn.Module):
    """
    Actor network that outputs action probabilities.
    
    Architecture:
        Input (state_dim) -> Hidden1 (256) -> Hidden2 (128) -> Output (action_dim)
        Uses ReLU activations and softmax output
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor (batch_size, state_dim)
            
        Returns:
            Action probabilities (batch_size, action_dim)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs


class ValueNetwork(nn.Module):
    """
    Critic network that estimates state value.
    
    Architecture:
        Input (state_dim) -> Hidden1 (256) -> Hidden2 (128) -> Output (1)
        Uses ReLU activations
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor (batch_size, state_dim)
            
        Returns:
            State value (batch_size, 1)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class SimplePPOAgent:
    """
    Simplified PPO agent for graph path retrieval.
    
    This implementation uses:
    - Separate actor (policy) and critic (value) networks
    - Clipped surrogate objective for policy updates
    - Generalized Advantage Estimation (GAE) for advantage computation
    - Entropy bonus for exploration
    
    Attributes:
        policy: Policy network (actor)
        value: Value network (critic)
        optimizer_policy: Optimizer for policy network
        optimizer_value: Optimizer for value network
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr_policy: float = 3e-4,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        epsilon_clip: float = 0.2,
        entropy_coef: float = 0.01,
        device: str = 'cpu'
    ):
        """
        Initialize the PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            lr_policy: Learning rate for policy network
            lr_value: Learning rate for value network
            gamma: Discount factor
            lambda_gae: GAE lambda parameter
            epsilon_clip: PPO clipping parameter
            entropy_coef: Entropy bonus coefficient
            device: Device to run on ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.epsilon_clip = epsilon_clip
        self.entropy_coef = entropy_coef
        self.device = device
        
        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.value = ValueNetwork(state_dim, hidden_dim).to(device)
        
        # Optimizers
        self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.optimizer_value = torch.optim.Adam(self.value.parameters(), lr=lr_value)
        
        # Trajectory buffer
        self.reset_buffer()
        
        print(f"Initialized SimplePPOAgent")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Device: {device}")
        print(f"  Policy params: {sum(p.numel() for p in self.policy.parameters()):,}")
        print(f"  Value params: {sum(p.numel() for p in self.value.parameters()):,}")
    
    def reset_buffer(self):
        """Reset the trajectory buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def get_top_k_actions(
        self,
        state: np.ndarray,
        k: int = 3,
        action_mask: np.ndarray = None
    ) -> List[Tuple[int, float]]:
        """
        Get top-k actions with their probabilities.
        
        Args:
            state: Current state
            k: Number of top actions to return
            action_mask: Binary mask of valid actions (1 = valid, 0 = invalid)
            
        Returns:
            List of (action, probability) tuples, sorted by probability descending
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.policy(state_tensor)
        
        # Apply action mask if provided
        if action_mask is not None:
            mask_tensor = torch.FloatTensor(action_mask).to(self.device)
            action_probs = action_probs * mask_tensor
            action_probs = action_probs / (action_probs.sum() + 1e-10)
        
        # Get top-k actions
        top_k_probs, top_k_indices = torch.topk(action_probs[0], k)
        
        return [(idx.item(), prob.item()) for idx, prob in zip(top_k_indices, top_k_probs)]

    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray = None,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select an action given a state.
        
        Args:
            state: Current state
            action_mask: Binary mask of valid actions (1 = valid, 0 = invalid)
            deterministic: If True, select max probability action
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.policy(state_tensor)
            value = self.value(state_tensor)
        
        # Apply action mask if provided
        if action_mask is not None:
            mask_tensor = torch.FloatTensor(action_mask).to(self.device)
            # Mask invalid actions by setting their probability to very small value
            action_probs = action_probs * mask_tensor
            # Renormalize
            action_probs = action_probs / (action_probs.sum() + 1e-10)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1).item()
            log_prob = torch.log(action_probs[0, action] + 1e-10).item()
        else:
            # Sample from distribution
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action).to(self.device)).item()
        
        return action, log_prob, value.item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float
    ):
        """
        Store a transition in the buffer.
        
        Args:
            state: State
            action: Action taken
            reward: Reward received
            done: Done flag
            log_prob: Log probability of action
            value: Value estimate
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def compute_gae(self, next_value: float = 0.0) -> np.ndarray:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            next_value: Value of next state (0 if terminal)
            
        Returns:
            Array of advantages
        """
        advantages = []
        gae = 0
        
        values = self.values + [next_value]
        
        for t in reversed(range(len(self.rewards))):
            # TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
            if self.dones[t]:
                next_val = 0
            else:
                next_val = values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_val - values[t]
            
            # GAE: A_t = δ_t + (γλ)*δ_{t+1} + (γλ)^2*δ_{t+2} + ...
            gae = delta + self.gamma * self.lambda_gae * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        return np.array(advantages)
    
    def update(self, num_epochs: int = 4) -> Dict[str, float]:
        """
        Update policy and value networks using PPO.
        
        Args:
            num_epochs: Number of optimization epochs
            
        Returns:
            Dictionary of training metrics
        """
        if len(self.states) == 0:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # Compute returns and advantages
        advantages = self.compute_gae()
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(self.values).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        policy_losses = []
        value_losses = []
        entropies = []
        
        # Multiple epochs of optimization
        for epoch in range(num_epochs):
            # Forward pass
            action_probs = self.policy(states)
            values = self.value(states).squeeze()
            
            # Get log probs and entropy
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            
            # Policy loss (negative because we want to maximize)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Total loss with entropy bonus
            total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
            
            # Update policy
            self.optimizer_policy.zero_grad()
            policy_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer_policy.step()
            
            # Update value
            self.optimizer_value.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=0.5)
            self.optimizer_value.step()
            
            # Store metrics
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())
        
        # Compute metrics
        metrics = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'avg_return': returns.mean().item(),
            'avg_advantage': advantages.mean().item()
        }
        
        # Reset buffer
        self.reset_buffer()
        
        return metrics
    
    def save(self, path: str):
        """
        Save agent to file.
        
        Args:
            path: Path to save file
        """
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'optimizer_policy_state_dict': self.optimizer_policy.state_dict(),
            'optimizer_value_state_dict': self.optimizer_value.state_dict(),
        }, path)
        print(f"Agent saved to {path}")
    
    def load(self, path: str):
        """
        Load agent from file.
        
        Args:
            path: Path to load file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.optimizer_policy.load_state_dict(checkpoint['optimizer_policy_state_dict'])
        self.optimizer_value.load_state_dict(checkpoint['optimizer_value_state_dict'])
        print(f"Agent loaded from {path}")


def main():
    """Test the PPO agent."""
    print("\n" + "="*60)
    print("Testing SimplePPOAgent")
    print("="*60)
    
    # Create agent
    state_dim = 690
    action_dim = 51
    
    agent = SimplePPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        device='cpu'
    )
    
    # Test action selection
    print("\n" + "="*60)
    print("Testing Action Selection")
    print("="*60)
    
    state = np.random.randn(state_dim)
    action, log_prob, value = agent.select_action(state)
    
    print(f"State shape: {state.shape}")
    print(f"Selected action: {action}")
    print(f"Log probability: {log_prob:.4f}")
    print(f"State value: {value:.4f}")
    
    # Test trajectory storage and update
    print("\n" + "="*60)
    print("Testing Trajectory Storage and Update")
    print("="*60)
    
    # Simulate a short trajectory
    for i in range(10):
        state = np.random.randn(state_dim)
        action, log_prob, value = agent.select_action(state)
        reward = np.random.randn()
        done = (i == 9)
        
        agent.store_transition(state, action, reward, done, log_prob, value)
    
    print(f"Stored {len(agent.states)} transitions")
    
    # Update agent
    metrics = agent.update(num_epochs=2)
    
    print("\nTraining Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test save/load
    print("\n" + "="*60)
    print("Testing Save/Load")
    print("="*60)
    
    save_path = "/tmp/test_agent.pt"
    agent.save(save_path)
    
    new_agent = SimplePPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device='cpu'
    )
    new_agent.load(save_path)
    
    print("\n✓ All tests passed!")


if __name__ == '__main__':
    main()
