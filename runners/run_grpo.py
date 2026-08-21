"""
GRPO (Group-Relative Policy Optimization) trainer for the same
explanation-selection bandit as the paper's PPO method.

Key differences from PPO:
  1. NO value network. Advantages are computed from within-group F1
     statistics: A(s, a) = (R_sem[i, a] - mean_a R_sem[i]) / std_a R_sem[i].
     This is the group-relative advantage from DeepSeekMath (Shao et al.,
     2024) adapted to a single-step bandit.
  2. NO GAE. Since the episode is 1 step, advantage is just the
     normalized reward.
  3. All K=40 candidates contribute to the advantage computation each
     episode — not just the sampled one. This densifies the supervision
     signal vs PPO (which only sees the sampled action's reward).
  4. Same clipped surrogate loss as PPO:
         L = -E[min(ρ·A, clip(ρ, 1-ε, 1+ε)·A)] - β·H
     with ρ = π_θ(a|s) / π_θ_old(a|s).
  5. Reward is the same median-centered semantic reward that won the
     reward-shaping ablation (variant C):
         r_i = R_sem[i, a] - median_a R_sem[i]
     (This is the Variant C reward; GRPO's group-normalization then
      divides the advantage by the within-group std.)

Everything else — policy network size, learning rate, entropy scheduler,
state vector, pool — identical to the paper's PPO.

The test set is G-Refer's held-out 600-pair split (test.jsonl);
we load the cached K=40 state/reward tables at
    results/rag_bandit/pool_cache/paper_ppo_tables_tr800_te600_K40_seed42.npz
built by scripts/paper_ppo_on_pool.py.

Usage:
    python3 run_grpo.py --seed 42 \
        --result_out results/seed42.json \
        --history_out results/seed42_history.json
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import FINAL_ROOT, setup_sys_path  # noqa: E402
setup_sys_path()

from src.simple_ppo_agent import PolicyNetwork  # noqa: E402
from src.entropy_scheduler import AdaptiveEntropyScheduler  # noqa: E402


class GRPOAgent:
    """Policy-only agent. Value network removed — advantages come from
    within-group normalization of the per-candidate rewards.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 lr_policy=3e-4, epsilon_clip=0.2, entropy_coef=0.1,
                 device='cpu'):
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.epsilon_clip = epsilon_clip
        self.entropy_coef = entropy_coef
        self.device = device
        self.action_dim = action_dim
        print(f"Initialized GRPOAgent")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Policy params: {sum(p.numel() for p in self.policy.parameters()):,}")

    @torch.no_grad()
    def action_probs(self, states):
        """states: (B, D) → probs (B, K)"""
        return self.policy(torch.from_numpy(states).float().to(self.device))

    @torch.no_grad()
    def select_action(self, state, deterministic=False):
        p = self.policy(torch.from_numpy(state).unsqueeze(0).float().to(self.device)).squeeze(0)
        if deterministic:
            return int(torch.argmax(p).item()), None
        dist = torch.distributions.Categorical(p)
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item())


def train_grpo(S, Rsem, Rstr, alpha, episodes, seed,
               beta_init, beta_final, fixed_beta=None,
               group_samples=1, verbose=True):
    """
    Group-relative policy optimization.

    Reward design (Variant C + GRPO normalization):
      r_i,a_sampled = alpha*Rstr[i,a] + (1-alpha)*(Rsem[i,a] - median_a Rsem[i])
    Advantage:
      A_i,a = (r_i,a - mean_a r_i) / (std_a r_i + eps)

    Every rollout samples one action per (state, group). The clipped
    surrogate uses the log-prob of that sampled action and the group-
    normalized advantage of that action. We also add a KL-like
    reference anchoring term by keeping π_θ_old fresh before each
    update (implicit in PPO's ρ = new/old).

    group_samples > 1: sample multiple actions per state within each
    episode, giving more gradient signal per batch.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    Rsem_median = np.median(Rsem, axis=1, keepdims=True)  # (N, 1)

    # Per-state full reward table (not just the sampled one): used for
    # advantage normalization across the K candidates.
    # reward[i, a] = alpha * Rstr[i, a] + (1 - alpha) * (Rsem[i, a] - median_a Rsem[i])
    full_reward = (alpha * Rstr + (1.0 - alpha) * (Rsem - Rsem_median))  # (N, K)
    reward_mean = full_reward.mean(axis=1, keepdims=True)   # (N, 1)
    reward_std = full_reward.std(axis=1, keepdims=True) + 1e-6  # (N, 1)
    # Advantage table: group-normalized
    A_table = (full_reward - reward_mean) / reward_std       # (N, K)

    state_dim = S.shape[1]
    action_dim = Rsem.shape[1]
    agent = GRPOAgent(
        state_dim=state_dim, action_dim=action_dim, hidden_dim=256,
        lr_policy=3e-4, epsilon_clip=0.2,
        entropy_coef=(fixed_beta if fixed_beta is not None else beta_init),
        device='cpu',
    )
    scheduler = AdaptiveEntropyScheduler(
        initial_coef=beta_init, final_coef=beta_final,
        total_episodes=episodes,
    )

    n = len(S)
    rng = np.random.RandomState(seed)
    history = []

    S_t = torch.from_numpy(S).float()

    for ep in range(episodes):
        beta = fixed_beta if fixed_beta is not None else \
            scheduler.get_coefficient(ep)
        agent.entropy_coef = beta

        # Snapshot the policy at the start of the episode (π_θ_old).
        with torch.no_grad():
            old_probs = agent.policy(S_t).detach()  # (N, K)

        # Rollout: sample `group_samples` actions per state.
        order = rng.permutation(n)

        # Build the training tensors
        sampled_s, sampled_a, sampled_A, sampled_logp_old = [], [], [], []
        ep_rewards = []
        for i in order:
            probs_i = old_probs[i]
            dist = torch.distributions.Categorical(probs_i)
            for _ in range(group_samples):
                a = int(dist.sample().item())
                sampled_s.append(i)
                sampled_a.append(a)
                sampled_A.append(float(A_table[i, a]))
                sampled_logp_old.append(float(torch.log(probs_i[a] + 1e-12).item()))
                ep_rewards.append(float(full_reward[i, a]))

        # Tensorize
        sampled_s = np.array(sampled_s)
        sampled_a = torch.tensor(sampled_a, dtype=torch.long)
        sampled_A_t = torch.tensor(sampled_A, dtype=torch.float32)
        sampled_logp_old_t = torch.tensor(sampled_logp_old, dtype=torch.float32)

        # Normalize advantages in-batch for stability
        adv = sampled_A_t
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Forward
        states_batch = S_t[sampled_s]

        # Four inner epochs, as in the paper's PPO
        for epoch in range(4):
            action_probs = agent.policy(states_batch)
            dist = torch.distributions.Categorical(action_probs)
            logp_new = dist.log_prob(sampled_a)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp_new - sampled_logp_old_t)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - agent.epsilon_clip, 1 + agent.epsilon_clip) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            loss = policy_loss - agent.entropy_coef * entropy

            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_norm=0.5)
            agent.optimizer.step()

        # Track metrics
        mean_r = float(np.mean(ep_rewards))
        with torch.no_grad():
            probs = agent.policy(S_t)
            sel_entropy = float(-(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item())
        if fixed_beta is None:
            scheduler.update(mean_r, sel_entropy)

        history.append({
            'ep': ep + 1, 'reward': mean_r, 'beta': beta,
            'sel_entropy': sel_entropy,
            'policy_loss': float(policy_loss.item()),
            'entropy': float(entropy.item()),
        })

        if verbose and ((ep + 1) % 50 == 0 or ep == 0):
            print(f"  [ep {ep+1:3d}] R={mean_r:+7.3f}  beta={beta:.4f}  "
                  f"H={sel_entropy:.3f}  loss={policy_loss.item():+.4f}")
    return agent, history


def evaluate(agent, S, Rsem, oracle, heuristic, grefer=0.4611):
    picked = []
    actions = []
    for i in range(len(S)):
        a, _ = agent.select_action(S[i], deterministic=True)
        picked.append(Rsem[i, a] / 100.0)
        actions.append(int(a))
    picked = np.array(picked)
    print(f"\n=== Test BERTScore-F1 (n={len(S)}) ===")
    print(f"  GRPO (this run)            : {picked.mean():.4f} +- {picked.std():.4f}")
    print(f"  Heuristic (argmax R_struct): {heuristic.mean():.4f}")
    print(f"  Pool oracle                : {oracle.mean():.4f}")
    print(f"  G-Refer (held-out 600)     : {grefer:.4f}")
    print(f"  delta vs G-Refer           : {picked.mean() - grefer:+.4f}")
    print(f"  delta vs PPO (0.4816 on 600): {picked.mean() - 0.4816:+.4f}")
    return picked, actions


def main():
    ap = argparse.ArgumentParser()
    default_tables = FINAL_ROOT / "results" / "paper_ppo_tables.npz"
    ap.add_argument("--tables", type=str, default=str(default_tables))
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--fixed_beta", type=str, default="none")
    ap.add_argument("--beta_init", type=float, default=0.1)
    ap.add_argument("--beta_final", type=float, default=0.01)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group_samples", type=int, default=1,
                    help="Actions sampled per state per episode (1 = bandit).")
    ap.add_argument("--grefer", type=float, default=0.4611,
                    help="G-Refer F1 on held-out 600 for delta reporting.")
    ap.add_argument("--result_out", type=str, default=None)
    ap.add_argument("--history_out", type=str, default=None)
    args = ap.parse_args()

    fixed_beta = None if args.fixed_beta.lower() == "none" else float(args.fixed_beta)

    z = np.load(args.tables)
    S_trn, Rsem_trn, Rstr_trn = z["S_trn"], z["Rsem_trn"], z["Rstr_trn"]
    S_tst, Rsem_tst, Rstr_tst = z["S_tst"], z["Rsem_tst"], z["Rstr_tst"]
    print(f"Loaded tables: S_trn={S_trn.shape}  S_tst={S_tst.shape}  K={Rsem_trn.shape[1]}")
    print(f"  alpha={args.alpha}  beta={args.beta_init}->{args.beta_final}  "
          f"episodes={args.episodes}  seed={args.seed}  group_samples={args.group_samples}")

    oracle = Rsem_tst.max(axis=1) / 100.0
    heuristic_idx = Rstr_tst.argmax(axis=1)
    heuristic = Rsem_tst[np.arange(len(heuristic_idx)), heuristic_idx] / 100.0

    agent, history = train_grpo(
        S_trn, Rsem_trn, Rstr_trn,
        alpha=args.alpha, episodes=args.episodes, seed=args.seed,
        beta_init=args.beta_init, beta_final=args.beta_final,
        fixed_beta=fixed_beta, group_samples=args.group_samples,
    )
    picked, actions = evaluate(agent, S_tst, Rsem_tst, oracle, heuristic,
                                 grefer=args.grefer)

    if args.history_out:
        Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.history_out, "w") as f:
            json.dump(history, f, indent=2)
        print(f"  Saved history -> {args.history_out}")

    if args.result_out:
        out = {
            "method": "grpo",
            "tables": args.tables,
            "alpha": float(args.alpha),
            "fixed_beta": fixed_beta,
            "beta_init": float(args.beta_init),
            "beta_final": float(args.beta_final),
            "episodes": int(args.episodes),
            "seed": int(args.seed),
            "group_samples": int(args.group_samples),
            "n_test": int(len(S_tst)),
            "k_cands": int(Rsem_trn.shape[1]),
            "grpo_f1_mean": float(picked.mean()),
            "grpo_f1_std": float(picked.std()),
            "pool_oracle_f1": float(oracle.mean()),
            "heuristic_f1_mean": float(heuristic.mean()),
            "grefer_heldout_600": float(args.grefer),
            "ppo_heldout_600_5seed_mean": 0.4816,
            "picks": actions,
        }
        Path(args.result_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.result_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved result -> {args.result_out}")


if __name__ == "__main__":
    main()
