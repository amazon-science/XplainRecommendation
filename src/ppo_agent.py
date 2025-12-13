"""
PPO Agent for RL-GraphRetriever

Proximal Policy Optimization for learning adaptive graph retrieval policy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict, Optional


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO
    
    Actor: Policy network that selects actions
    Critic: Value network that estimates state values
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        """
        Initialize Actor-Critic network
        
        Parameters
        ----------
        state_dim : int
            Dimension of state representation
        action_dim : int
            Maximum number of actions (will use masking for variable actions)
        hidden_dim : int
            Hidden layer dimension
        """
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Parameters
        ----------
        state : torch.Tensor
            State representation
        
        Returns
        -------
        action_logits : torch.Tensor
            Logits for action selection
        value : torch.Tensor
            State value estimate
        """
        features = self.shared(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Parameters
        ----------
        state : torch.Tensor
            Current state
        action_mask : torch.Tensor, optional
            Mask for valid actions (1 for valid, 0 for invalid)
        
        Returns
        -------
        action : int
            Selected action index
        log_prob : torch.Tensor
            Log probability of selected action
        value : torch.Tensor
            State value estimate
        """
        action_logits, value = self.forward(state)
        
        # Apply mask if provided
        if action_mask is not None:
            action_logits = action_logits.masked_fill(action_mask == 0, -1e9)
        
        # Sample action
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions (for PPO update)
        
        Parameters
        ----------
        states : torch.Tensor
            Batch of states
        actions : torch.Tensor
            Batch of actions taken
        action_masks : torch.Tensor, optional
            Batch of action masks
        
        Returns
        -------
        log_probs : torch.Tensor
            Log probabilities of actions
        values : torch.Tensor
            State value estimates
        entropy : torch.Tensor
            Policy entropy
        """
        action_logits, values = self.forward(states)
        
        # Apply masks if provided
        if action_masks is not None:
            action_logits = action_logits.masked_fill(action_masks == 0, -1e9)
        
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy


class PPOAgent:
    """PPO Agent for training retrieval policy"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize PPO agent
        
        Parameters
        ----------
        state_dim : int
            State dimension
        action_dim : int
            Action dimension
        lr : float
            Learning rate
        gamma : float
            Discount factor
        gae_lambda : float
            GAE lambda parameter
        clip_epsilon : float
            PPO clipping parameter
        value_coef : float
            Value loss coefficient
        entropy_coef : float
            Entropy bonus coefficient
        max_grad_norm : float
            Maximum gradient norm for clipping
        device : str
            Device for computation
        """
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Initialize actor-critic
        self.actor_critic = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # Training stats
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'entropy': []
        }
    
    def select_action(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select action using current policy"""
        with torch.no_grad():
            return self.actor_critic.get_action(state, action_mask)
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[torch.Tensor],
        dones: List[bool],
        next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Returns
        -------
        advantages : torch.Tensor
            Advantage estimates
        returns : torch.Tensor
            Discounted returns
        """
        advantages = []
        gae = 0
        
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + torch.stack(values[:-1])
        
        return advantages, returns
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        n_epochs: int = 4,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        PPO update step
        
        Parameters
        ----------
        states : torch.Tensor
            Batch of states
        actions : torch.Tensor
            Batch of actions
        old_log_probs : torch.Tensor
            Log probs from old policy
        advantages : torch.Tensor
            Advantage estimates
        returns : torch.Tensor
            Discounted returns
        action_masks : torch.Tensor, optional
            Action masks
        n_epochs : int
            Number of update epochs
        batch_size : int
            Mini-batch size
        
        Returns
        -------
        stats : dict
            Training statistics
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = states.size(0)
        indices = np.arange(dataset_size)
        
        epoch_stats = {'policy_loss': [], 'value_loss': [], 'entropy': []}
        
        for _ in range(n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_masks = action_masks[batch_indices] if action_masks is not None else None
                
                # Evaluate actions
                log_probs, values, entropy = self.actor_critic.evaluate_actions(
                    batch_states, batch_actions, batch_masks
                )
                
                # Compute ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                # Policy loss (negative because we want to maximize)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy bonus (encourage exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Record stats
                epoch_stats['policy_loss'].append(policy_loss.item())
                epoch_stats['value_loss'].append(value_loss.item())
                epoch_stats['entropy'].append(-entropy_loss.item())
        
        # Average stats
        stats = {k: np.mean(v) for k, v in epoch_stats.items()}
        
        # Update training stats
        for k, v in stats.items():
            self.training_stats[k].append(v)
        
        return stats
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_stats = checkpoint['training_stats']
