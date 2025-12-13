"""
Hybrid Training Script: Heuristic Path Extraction + RL Ranking

This script implements Option 2:
1. Extract candidate paths using heuristics (shortest paths, random walks)
2. Train RL agent to rank/select the best paths

Advantage: Much easier than full navigation (10 choices vs 30K nodes)
Expected: 80-95% success rate

Usage:
    python train_hybrid_ranker.py --dataset yelp --episodes 500

Author: RL-GraphRetriever
Date: December 2025
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import GReferDataLoader
from heuristic_path_extractor import HeuristicPathExtractor
from path_ranker_env import PathRankingEnv
from simple_ppo_agent import SimplePPOAgent


def train_hybrid(args):
    """Train hybrid path ranker."""
    
    print("\n" + "="*80)
    print("Hybrid Path Ranking Training")
    print("="*80)
    print(f"Approach: Heuristic Extraction + RL Ranking")
    print(f"Dataset: {args.dataset}")
    print(f"Episodes: {args.episodes}")
    print("="*80 + "\n")
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"hybrid_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    data_loader = GReferDataLoader(dataset_name=args.dataset)
    data_loader.load_all(split='trn')
    
    # Create heuristic path extractor
    print("\nCreating heuristic path extractor...")
    extractor = HeuristicPathExtractor(
        graph=data_loader.graph,
        data_loader=data_loader,
        num_paths=args.num_candidate_paths
    )
    
    # Create ranking environment
    print("Creating path ranking environment...")
    env = PathRankingEnv(
        data_loader=data_loader,
        path_extractor=extractor,
        num_paths=args.num_candidate_paths
    )
    
    # Create RL agent
    print("Creating PPO agent for ranking...")
    agent = SimplePPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=args.hidden_dim,
        lr_policy=args.lr_policy,
        lr_value=args.lr_value,
        gamma=args.gamma,
        device='cpu'
    )
    
    # Training metrics
    episode_rewards = []
    success_rates = []
    best_path_selections = []
    
    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    start_time = time.time()
    best_avg_reward = -float('inf')
    
    for episode in range(args.episodes):
        # Reset
        state = env.reset()
        
        # Agent selects path
        action, log_prob, value = agent.select_action(state)
        
        # Evaluate selection
        next_state, reward, done, info = env.step(action)
        
        # Store transition
        agent.store_transition(state, action, reward, done, log_prob, value)
        
        # Track metrics
        episode_rewards.append(reward)
        success_rates.append(1.0 if info['reaches_target'] else 0.0)
        best_path_selections.append(1.0 if action == 0 else 0.0)  # Path 0 is shortest
        
        # Update agent
        if (episode + 1) % args.update_freq == 0:
            metrics = agent.update(num_epochs=args.ppo_epochs)
            
            # Compute stats
            recent_rewards = episode_rewards[-args.update_freq:]
            recent_success = success_rates[-args.update_freq:]
            recent_best_select = best_path_selections[-args.update_freq:]
            
            avg_reward = np.mean(recent_rewards)
            success_rate = np.mean(recent_success)
            best_select_rate = np.mean(recent_best_select)
            
            print(
                f"Episode {episode+1}/{args.episodes} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Success Rate: {success_rate:.1%} | "
                f"Best Path %: {best_select_rate:.1%} | "
                f"Loss: {metrics.get('policy_loss', 0):.4f}"
            )
            
            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(output_dir / 'best_ranker.pt')
        
        # Periodic saves
        if (episode + 1) % args.save_freq == 0:
            agent.save(output_dir / f'ranker_episode_{episode+1}.pt')
            plot_metrics(episode_rewards, success_rates, best_path_selections, output_dir)
    
    # Training complete
    training_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Total time: {training_time/60:.2f} minutes")
    print(f"Episodes per second: {args.episodes/training_time:.2f}")
    
    # Save final model
    agent.save(output_dir / 'final_ranker.pt')
    plot_metrics(episode_rewards, success_rates, best_path_selections, output_dir)
    
    # Print final statistics
    final_100_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    final_100_success = np.mean(success_rates[-100:]) if len(success_rates) >= 100 else np.mean(success_rates)
    final_100_best = np.mean(best_path_selections[-100:]) if len(best_path_selections) >= 100 else np.mean(best_path_selections)
    
    print("\nFinal Statistics:")
    print(f"  Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"  Success Rate: {np.mean(success_rates):.1%}")
    print(f"  Selects Best Path: {np.mean(best_path_selections):.1%}")
    print(f"  Final 100 Avg Reward: {final_100_reward:.2f}")
    print(f"  Final 100 Success Rate: {final_100_success:.1%}")
    print(f"  Final 100 Best Path %: {final_100_best:.1%}")
    
    # Save results
    results = {
        'avg_reward': float(np.mean(episode_rewards)),
        'success_rate': float(np.mean(success_rates)),
        'selects_best_path_rate': float(np.mean(best_path_selections)),
        'final_100_avg_reward': float(final_100_reward),
        'final_100_success_rate': float(final_100_success),
        'final_100_best_path_rate': float(final_100_best)
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print("✓ Training completed successfully!")
    
    return results


def plot_metrics(rewards, success_rates, best_selections, output_dir):
    """Plot training metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Rewards
    axes[0].plot(rewards, alpha=0.4)
    if len(rewards) > 50:
        window = 50
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(rewards)), smoothed, linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Path Selection Reward')
    axes[0].grid(True, alpha=0.3)
    
    # Success rate
    window = min(50, len(success_rates))
    if window > 1:
        success_ma = np.convolve(success_rates, np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(success_rates)), success_ma, linewidth=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Success Rate')
    axes[1].set_title(f'Success Rate ({window}-episode MA)')
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(True, alpha=0.3)
    
    # Best path selection rate
    window = min(50, len(best_selections))
    if window > 1:
        best_ma = np.convolve(best_selections, np.ones(window)/window, mode='valid')
        axes[2].plot(range(window-1, len(best_selections)), best_ma, linewidth=2)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Selects Best Path')
    axes[2].set_title(f'Best Path Selection ({window}-episode MA)')
    axes[2].set_ylim([0, 1.05])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train hybrid path ranker')
    
    parser.add_argument('--dataset', type=str, default='yelp',
                       choices=['amazon', 'google', 'yelp'])
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--num_candidate_paths', type=int, default=10)
    parser.add_argument('--update_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--lr_policy', type=float, default=3e-4)
    parser.add_argument('--lr_value', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    np.random.seed(args.seed)
    
    try:
        results = train_hybrid(args)
        return 0
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
