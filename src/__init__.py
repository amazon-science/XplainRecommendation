"""
RL-GraphRetriever: Reinforcement Learning-based Graph Retrieval for Explainable Recommendations

This package implements an RL-based adaptive retrieval policy using PPO that learns to 
select optimal graph paths based on downstream explanation quality.

Key Improvements over G-Refer:
1. RL-based adaptive retrieval (replaces heuristic PaGELink)
2. Amazon Bedrock integration (replaces OpenAI)
3. PPO-based learning for personalized retrieval strategies
"""

from .bedrock_llm import BedrockLLM
from .rl_environment_networkx import GraphRetrievalEnvironment
from .ppo_agent import PPOAgent, ActorCritic

__version__ = "0.1.0"
__all__ = [
    "BedrockLLM",
    "GraphRetrievalEnvironment", 
    "PPOAgent",
    "ActorCritic"
]
