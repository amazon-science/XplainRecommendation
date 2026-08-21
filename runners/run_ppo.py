"""
Shared reward-shaping runner for PPO ablation experiments in
`approach_ppo_0.4864/experiments/`.

Forks the paper training loop (scripts/paper_ppo_train_only.py) but replaces
the reward expression inside the rollout. Three modes are supported:

  baseline  -- paper's reward: r = alpha*Rstr[i,a] + (1-alpha)*Rsem[i,a]
               (Rsem is 100 × BERTScore-F1, Rstr is the structural bonus.)

  zscore    -- Variant A: within-group rank-normalized semantic reward.
               r = (Rsem[i,a] - mean_a Rsem[i]) / (std_a Rsem[i] + eps)
               + alpha * Rstr[i,a]
               This removes per-sample scale noise: a sample where all
               candidates score 0.20–0.25 F1 now has unit-variance reward,
               same as a sample where candidates span 0.45–0.85.

  alpha0    -- Variant B: drop the structural term entirely (alpha=0).
               r = Rsem[i,a]  (identical to baseline at alpha=0, just a
               convenience rename for the ablation JSON.)

  median    -- Variant C: baseline-subtracted reward.
               r = Rsem[i,a] - median_a Rsem[i] + alpha * Rstr[i,a]
               Advantage is positive iff the picked candidate beats a
               coin-flip pick on this sample. Doesn't change scale noise
               but gives the value network a zero-centered target.

The PPO agent, adaptive entropy scheduler, and every hyperparameter are
identical to the paper's headline run. Only the scalar reward expression
differs.

Usage from inside one of the variant folders, e.g.:
    python3 ../_shared/run_variant.py \\
        --reward_mode zscore --alpha 0.1 --seed 42 \\
        --result_out results/seed42.json \\
        --history_out results/seed42_history.json \\
        2>&1 | tee logs/seed42.log

The cached tables at
    results/rag_bandit/pool_cache/paper_ppo_tables_tr800_te200_K40_seed42.npz
are loaded via the --tables flag (defaults to that path).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import FINAL_ROOT, setup_sys_path  # noqa: E402
setup_sys_path()

from src.simple_ppo_agent import SimplePPOAgent  # noqa: E402
from src.entropy_scheduler import AdaptiveEntropyScheduler  # noqa: E402


def train_with_reward(S, Rsem, Rstr, alpha, reward_mode,
                       episodes, seed, initial_beta, final_beta,
                       fixed_beta=None):
    """PPO training loop with pluggable reward shape.

    Pre-compute per-sample statistics so the reward expression is O(1) at
    rollout time.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Per-sample statistics computed from the CACHED TRAIN F1 labels only.
    # These labels are BERTScore-F1 of train candidates vs train refs — no
    # test-reference leakage.
    Rsem_mean = Rsem.mean(axis=1, keepdims=True)          # (N, 1)
    Rsem_std = Rsem.std(axis=1, keepdims=True) + 1e-6     # (N, 1)
    Rsem_median = np.median(Rsem, axis=1, keepdims=True)  # (N, 1)

    state_dim = S.shape[1]
    action_dim = Rsem.shape[1]
    agent = SimplePPOAgent(
        state_dim=state_dim, action_dim=action_dim, hidden_dim=256,
        lr_policy=3e-4, lr_value=1e-3, gamma=0.99, lambda_gae=0.95,
        epsilon_clip=0.2,
        entropy_coef=(fixed_beta if fixed_beta is not None else initial_beta),
        device='cpu',
    )
    scheduler = AdaptiveEntropyScheduler(
        initial_coef=initial_beta, final_coef=final_beta,
        total_episodes=episodes,
    )

    def compute_reward(i, a):
        sem = Rsem[i, a]
        struct = Rstr[i, a]
        if reward_mode == "baseline":
            return alpha * struct + (1.0 - alpha) * sem
        if reward_mode == "zscore":
            # per-sample z-score over the K semantic scores
            z = (sem - Rsem_mean[i, 0]) / Rsem_std[i, 0]
            return alpha * struct + (1.0 - alpha) * z
        if reward_mode == "alpha0":
            # Explicit α=0: structural term dropped entirely
            return sem
        if reward_mode == "median":
            return alpha * struct + (1.0 - alpha) * (sem - Rsem_median[i, 0])
        raise ValueError(f"Unknown reward_mode: {reward_mode}")

    n = len(S)
    rng = np.random.RandomState(seed)
    history = []
    for ep in range(episodes):
        beta = fixed_beta if fixed_beta is not None else \
            scheduler.get_coefficient(ep)
        agent.entropy_coef = beta
        agent.reset_buffer()
        order = rng.permutation(n)
        ep_rewards = []
        sel_entropy_accum = []
        for i in order:
            s = S[i]
            action, logp, value = agent.select_action(s, deterministic=False)
            r = float(compute_reward(i, int(action)))
            agent.store_transition(s, action, r, True, logp, value)
            ep_rewards.append(r)
            with torch.no_grad():
                probs = agent.policy(
                    torch.from_numpy(s).unsqueeze(0).float()
                ).squeeze(0)
                sel_entropy_accum.append(
                    -(probs * torch.log(probs + 1e-10)).sum().item()
                )
        metrics = agent.update(num_epochs=4)
        mean_r = float(np.mean(ep_rewards))
        sel_ent = float(np.mean(sel_entropy_accum))
        if fixed_beta is None:
            scheduler.update(mean_r, sel_ent)
        history.append({
            'ep': ep + 1, 'reward': mean_r, 'beta': beta,
            'sel_entropy': sel_ent, **metrics,
        })
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"  [ep {ep+1:3d}] R={mean_r:+7.3f}  beta={beta:.4f}  "
                  f"H={sel_ent:.3f}  "
                  f"policy_loss={metrics.get('policy_loss', 0):+.4f}")
    return agent, history


def evaluate(agent, S, Rsem, oracle, heuristic, grefer=0.4592):
    picked = []
    actions = []
    for i in range(len(S)):
        a, _, _ = agent.select_action(S[i], deterministic=True)
        picked.append(Rsem[i, a] / 100.0)
        actions.append(int(a))
    picked = np.array(picked)
    print(f"\n=== Test BERTScore-F1 (rescale_with_baseline=True, n={len(S)}) ===")
    print(f"  PPO variant (this run)     : {picked.mean():.4f} +- {picked.std():.4f}")
    print(f"  Heuristic (argmax R_struct): {heuristic.mean():.4f}")
    print(f"  Pool oracle                : {oracle.mean():.4f}")
    print(f"  G-Refer (reported)         : {grefer:.4f}")
    print(f"  delta vs G-Refer           : {picked.mean() - grefer:+.4f}")
    return picked, actions


def main():
    ap = argparse.ArgumentParser()
    default_tables = FINAL_ROOT / "results" / "paper_ppo_tables.npz"
    ap.add_argument("--tables", type=str, default=str(default_tables))
    ap.add_argument("--reward_mode", type=str, default="baseline",
                    choices=["baseline", "zscore", "alpha0", "median"])
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="Structural weight. Ignored when reward_mode='alpha0'.")
    ap.add_argument("--fixed_beta", type=str, default="none")
    ap.add_argument("--beta_init", type=float, default=0.1)
    ap.add_argument("--beta_final", type=float, default=0.01)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grefer", type=float, default=0.4592)
    ap.add_argument("--result_out", type=str, default=None)
    ap.add_argument("--history_out", type=str, default=None)
    args = ap.parse_args()

    fixed_beta = None if args.fixed_beta.lower() == "none" else \
        float(args.fixed_beta)

    z = np.load(args.tables)
    S_trn, Rsem_trn, Rstr_trn = z["S_trn"], z["Rsem_trn"], z["Rstr_trn"]
    S_tst, Rsem_tst, Rstr_tst = z["S_tst"], z["Rsem_tst"], z["Rstr_tst"]
    print(f"Loaded tables from {args.tables}")
    print(f"  S_trn={S_trn.shape}  S_tst={S_tst.shape}")
    print(f"  reward_mode={args.reward_mode}  alpha={args.alpha}  "
          f"fixed_beta={fixed_beta}  beta={args.beta_init}->{args.beta_final}  "
          f"episodes={args.episodes}  seed={args.seed}")

    oracle = Rsem_tst.max(axis=1) / 100.0
    heuristic_idx = Rstr_tst.argmax(axis=1)
    heuristic = Rsem_tst[np.arange(len(heuristic_idx)), heuristic_idx] / 100.0

    agent, history = train_with_reward(
        S_trn, Rsem_trn, Rstr_trn,
        alpha=args.alpha, reward_mode=args.reward_mode,
        episodes=args.episodes, seed=args.seed,
        initial_beta=args.beta_init, final_beta=args.beta_final,
        fixed_beta=fixed_beta,
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
            "tables": args.tables,
            "reward_mode": args.reward_mode,
            "alpha": float(args.alpha),
            "fixed_beta": fixed_beta,
            "beta_init": float(args.beta_init),
            "beta_final": float(args.beta_final),
            "episodes": int(args.episodes),
            "seed": int(args.seed),
            "n_test": int(len(S_tst)),
            "k_cands": int(Rsem_trn.shape[1]),
            "ppo_f1_mean": float(picked.mean()),
            "ppo_f1_std": float(picked.std()),
            "pool_oracle_f1": float(oracle.mean()),
            "heuristic_f1_mean": float(heuristic.mean()),
            "grefer_reported": float(args.grefer),
            "picks": actions,
        }
        Path(args.result_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.result_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved result -> {args.result_out}")


if __name__ == "__main__":
    main()
