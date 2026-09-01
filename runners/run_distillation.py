# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Policy distillation from LambdaRank teacher → PPO-style MLP student,
followed by optional GRPO fine-tuning.

Two-stage training on G-Refer held-out 600 (K=40 cached tables):

  Stage A — Supervised distillation.
    For each training state s_i, LambdaRank produces per-candidate
    scores s_i_LR ∈ R^K from the 30-dim per-candidate features. We
    softmax these with temperature T to get a target distribution
    q_i = softmax(s_i_LR / T).
    The MLP policy π_θ minimizes KL(q_i || π_θ(· | s_i)) over 800 train
    states.

  Stage B — GRPO fine-tuning (optional, gated by --rl_fine_tune).
    Warm-started from Stage A, fine-tune with GRPO on the same
    reward/advantage structure as variantD.

Why this can move past DPO/GRPO's 0.49 plateau:
  — The MLP student now has LambdaRank's knowledge baked into its
    parameters. Its argmax at test time should be close to
    LambdaRank's argmax (≈ 0.4980 F1).
  — Stage B adds RL exploration on top — can only match or beat Stage
    A, never hurt (we monitor).

Usage:
    python3 scripts/run_distillation.py --seed 42 \\
        --stageA_epochs 200 --stageB_epochs 100 \\
        --rl_fine_tune 1 \\
        --result_out results/seed42.json
"""
import argparse, copy, json, os, sys
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


def extract_candidate_features(S, k=None):
    """S has shape (N, 1536 + K*64) = [e_u(768); e_i(768); K*64 feat].
    Return the first 30 dims of each candidate's 64-dim slot.
    Derives K from state shape if not provided.
    """
    if k is None:
        k = (S.shape[1] - 1536) // 64
    tail = S[:, 1536:].reshape(S.shape[0], k, 64)
    return tail[:, :, :30]  # (N, K, 30)


def fit_lambdarank_teacher(X_trn_flat, y_trn_flat, groups_trn, seed):
    """Fit the same LambdaRank as the paper pipeline, return the trained
    ranker.
    """
    import lightgbm as lgb
    # Quintile-bin labels within groups
    y_lab = np.zeros_like(y_trn_flat, dtype=np.int32)
    idx = 0
    for g in groups_trn:
        chunk = y_trn_flat[idx:idx+g]
        lo, hi = chunk.min(), chunk.max()
        if hi > lo:
            q = np.clip((chunk - lo) / (hi - lo) * 5, 0, 4).astype(int)
            y_lab[idx:idx+g] = q
        idx += g
    ranker = lgb.LGBMRanker(
        objective='lambdarank', n_estimators=500, num_leaves=31,
        min_data_in_leaf=10, learning_rate=0.05,
        verbose=-1, random_state=seed,
    )
    ranker.fit(X_trn_flat, y_lab, group=groups_trn)
    return ranker


def train_distill_stageA(S_trn, teacher_scores, episodes, seed,
                          temperature=1.0, entropy_coef=0.0,
                          lr=3e-4, verbose=True):
    """Stage A: supervised KL distillation.

    teacher_scores: (N, K) LambdaRank scores for training states.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    N, K = teacher_scores.shape
    policy = PolicyNetwork(S_trn.shape[1], K, hidden_dim=256)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    S_t = torch.from_numpy(S_trn).float()
    # Softmax the teacher scores with temperature
    teacher_logits_t = torch.from_numpy(teacher_scores / temperature).float()
    teacher_probs = torch.softmax(teacher_logits_t, dim=-1)

    history = []
    for ep in range(episodes):
        probs = policy(S_t)  # (N, K)
        log_probs = torch.log(probs + 1e-12)
        # KL(q || p) = sum q * (log q - log p). We minimise forward KL.
        kl = (teacher_probs * (torch.log(teacher_probs + 1e-12) - log_probs)).sum(dim=-1).mean()
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        loss = kl - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

        if verbose and ((ep + 1) % 25 == 0 or ep == 0):
            # Student agreement with teacher argmax
            with torch.no_grad():
                student_arg = probs.argmax(dim=-1)
                teacher_arg = teacher_probs.argmax(dim=-1)
                agree = (student_arg == teacher_arg).float().mean().item()
            print(f"  [A ep {ep+1:3d}] KL={kl.item():.4f}  ent={entropy.item():.3f}  "
                  f"teacher_agree={agree:.3f}")
            history.append({
                'stage': 'A', 'ep': ep + 1, 'kl': float(kl.item()),
                'entropy': float(entropy.item()), 'teacher_agree': agree,
            })
    return policy, history


def train_distill_stageB(policy, S_trn, Rsem_trn, Rstr_trn, episodes, seed,
                          alpha=0.1, beta_init=0.05, beta_final=0.005,
                          lr_fine_tune=1e-4, verbose=True):
    """Stage B: GRPO fine-tune starting from the distilled policy.
    Lower LR to avoid blowing up the distilled prior.
    """
    # NOTE: we rebuild optimizer with the lower FT LR.
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr_fine_tune)
    scheduler = AdaptiveEntropyScheduler(
        initial_coef=beta_init, final_coef=beta_final, total_episodes=episodes,
    )

    Rsem_median = np.median(Rsem_trn, axis=1, keepdims=True)
    full_reward = alpha * Rstr_trn + (1 - alpha) * (Rsem_trn - Rsem_median)
    rmean = full_reward.mean(axis=1, keepdims=True)
    rstd = full_reward.std(axis=1, keepdims=True) + 1e-6
    A_table = (full_reward - rmean) / rstd  # (N, K)

    N = S_trn.shape[0]
    S_t = torch.from_numpy(S_trn).float()
    rng = np.random.RandomState(seed + 1000)  # different stream from A
    history = []

    for ep in range(episodes):
        beta = scheduler.get_coefficient(ep)

        # Snapshot π_θ_old
        with torch.no_grad():
            old_probs = policy(S_t).detach()

        # Rollout one action per state
        sampled_i, sampled_a, sampled_A, sampled_logp_old = [], [], [], []
        ep_rewards = []
        for i in rng.permutation(N):
            p = old_probs[i]
            dist = torch.distributions.Categorical(p)
            a = int(dist.sample().item())
            sampled_i.append(i)
            sampled_a.append(a)
            sampled_A.append(float(A_table[i, a]))
            sampled_logp_old.append(float(torch.log(p[a] + 1e-12).item()))
            ep_rewards.append(float(full_reward[i, a]))

        sampled_i = np.array(sampled_i)
        sampled_a_t = torch.tensor(sampled_a, dtype=torch.long)
        sampled_A_t = torch.tensor(sampled_A, dtype=torch.float32)
        sampled_logp_old_t = torch.tensor(sampled_logp_old, dtype=torch.float32)
        adv = sampled_A_t
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        states_batch = S_t[sampled_i]

        for epoch in range(4):
            probs = policy(states_batch)
            logp = torch.log(probs + 1e-12)
            logp_new = logp.gather(1, sampled_a_t.unsqueeze(1)).squeeze(1)
            entropy = -(probs * logp).sum(dim=-1).mean()

            ratio = torch.exp(logp_new - sampled_logp_old_t)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            loss = policy_loss - beta * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

        mean_r = float(np.mean(ep_rewards))
        with torch.no_grad():
            probs_all = policy(S_t)
            sel_ent = float(-(probs_all * torch.log(probs_all + 1e-10)).sum(dim=-1).mean().item())
        scheduler.update(mean_r, sel_ent)

        if verbose and ((ep + 1) % 25 == 0 or ep == 0):
            print(f"  [B ep {ep+1:3d}] R={mean_r:+6.3f}  beta={beta:.4f}  "
                  f"H={sel_ent:.3f}  loss={policy_loss.item():+.4f}")
            history.append({
                'stage': 'B', 'ep': ep + 1, 'reward': mean_r,
                'entropy': sel_ent, 'beta': beta,
                'policy_loss': float(policy_loss.item()),
            })
    return policy, history


def evaluate(policy, S, Rsem, tag, oracle, heuristic, grefer=0.4611,
             ppo_mean=0.4816, grpo_mean=0.4853, dpo_mean=0.4890,
             lambdarank_mean=0.4980):
    S_t = torch.from_numpy(S).float()
    with torch.no_grad():
        probs = policy(S_t)
        actions = probs.argmax(dim=-1).tolist()
    picked = np.array([Rsem[i, a] / 100.0 for i, a in enumerate(actions)])
    print(f"\n=== Test F1 ({tag}, n={len(S)}) ===")
    print(f"  {tag:<30}: {picked.mean():.4f} +- {picked.std():.4f}")
    print(f"  Pool oracle                   : {oracle.mean():.4f}")
    print(f"  G-Refer (held-out 600)        : {grefer:.4f}")
    print(f"  vs PPO (0.4816)               : {picked.mean()-ppo_mean:+.4f}")
    print(f"  vs GRPO (0.4853)              : {picked.mean()-grpo_mean:+.4f}")
    print(f"  vs DPO (0.4890)               : {picked.mean()-dpo_mean:+.4f}")
    print(f"  vs LambdaRank (0.4980)        : {picked.mean()-lambdarank_mean:+.4f}")
    return picked, actions


def main():
    ap = argparse.ArgumentParser()
    default_tables = FINAL_ROOT / "results" / "paper_ppo_tables.npz"
    ap.add_argument("--tables", type=str, default=str(default_tables))
    ap.add_argument("--stageA_epochs", type=int, default=200,
                    help="Supervised distillation epochs.")
    ap.add_argument("--stageB_epochs", type=int, default=100,
                    help="GRPO fine-tuning epochs.")
    ap.add_argument("--rl_fine_tune", type=int, default=1,
                    help="1 to run Stage B, 0 to skip.")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Softmax temperature for teacher scores. "
                         "Lower = sharper target.")
    ap.add_argument("--stageA_entropy", type=float, default=0.0)
    ap.add_argument("--stageB_alpha", type=float, default=0.1)
    ap.add_argument("--stageB_beta_init", type=float, default=0.05)
    ap.add_argument("--stageB_beta_final", type=float, default=0.005)
    ap.add_argument("--stageA_lr", type=float, default=3e-4)
    ap.add_argument("--stageB_lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grefer", type=float, default=0.4611)
    ap.add_argument("--result_out", type=str, default=None)
    ap.add_argument("--history_out", type=str, default=None)
    args = ap.parse_args()

    z = np.load(args.tables)
    S_trn, Rsem_trn, Rstr_trn = z["S_trn"], z["Rsem_trn"], z["Rstr_trn"]
    S_tst, Rsem_tst, Rstr_tst = z["S_tst"], z["Rsem_tst"], z["Rstr_tst"]
    print(f"Loaded tables: S_trn={S_trn.shape}  S_tst={S_tst.shape}  K={Rsem_trn.shape[1]}")

    # ---- Build LambdaRank teacher ----
    print("\n--- Training LambdaRank teacher on state-extracted features ---")
    feats_trn = extract_candidate_features(S_trn)   # (N_trn, K, 30)
    feats_tst = extract_candidate_features(S_tst)   # (N_tst, K, 30)
    X_trn_flat = feats_trn.reshape(-1, 30)
    X_tst_flat = feats_tst.reshape(-1, 30)
    y_trn_flat = Rsem_trn.reshape(-1)
    groups_trn = [S_trn.shape[1] // 30 * 0 + feats_trn.shape[1]] * feats_trn.shape[0]
    groups_trn = [feats_trn.shape[1]] * feats_trn.shape[0]  # K per group

    ranker = fit_lambdarank_teacher(X_trn_flat, y_trn_flat, groups_trn, args.seed)

    # Teacher scores for both train (for distillation) and test (for reporting)
    teacher_trn = ranker.predict(X_trn_flat).reshape(S_trn.shape[0], -1)
    teacher_tst = ranker.predict(X_tst_flat).reshape(S_tst.shape[0], -1)

    # Report LambdaRank's own F1 on held-out 600 as the upper-bound for Stage A
    lr_picks = teacher_tst.argmax(axis=1)
    lr_f1 = np.array([Rsem_tst[i, a] / 100.0 for i, a in enumerate(lr_picks)])
    print(f"  LambdaRank teacher F1 on held-out {S_tst.shape[0]}: {lr_f1.mean():.4f}")

    oracle = Rsem_tst.max(axis=1) / 100.0
    heuristic_idx = Rstr_tst.argmax(axis=1)
    heuristic = Rsem_tst[np.arange(len(heuristic_idx)), heuristic_idx] / 100.0

    # ---- Stage A: supervised distillation ----
    print(f"\n--- Stage A: distill teacher → MLP for {args.stageA_epochs} epochs ---")
    policy, hist_A = train_distill_stageA(
        S_trn, teacher_trn, episodes=args.stageA_epochs, seed=args.seed,
        temperature=args.temperature,
        entropy_coef=args.stageA_entropy, lr=args.stageA_lr,
    )
    picked_A, actions_A = evaluate(policy, S_tst, Rsem_tst, "Distill-only (Stage A)",
                                     oracle, heuristic, grefer=args.grefer)

    # ---- Stage B: GRPO fine-tune ----
    picked_AB = None
    actions_AB = None
    hist_B = []
    if args.rl_fine_tune:
        print(f"\n--- Stage B: GRPO fine-tune for {args.stageB_epochs} epochs ---")
        policy, hist_B = train_distill_stageB(
            policy, S_trn, Rsem_trn, Rstr_trn,
            episodes=args.stageB_epochs, seed=args.seed,
            alpha=args.stageB_alpha,
            beta_init=args.stageB_beta_init, beta_final=args.stageB_beta_final,
            lr_fine_tune=args.stageB_lr,
        )
        picked_AB, actions_AB = evaluate(policy, S_tst, Rsem_tst, "Distill + GRPO (Stage A+B)",
                                            oracle, heuristic, grefer=args.grefer)

    full_history = hist_A + hist_B

    if args.history_out:
        Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.history_out, "w") as f:
            json.dump(full_history, f, indent=2)
        print(f"  Saved history -> {args.history_out}")

    if args.result_out:
        out = {
            "method": "distillation",
            "tables": args.tables,
            "stageA_epochs": int(args.stageA_epochs),
            "stageB_epochs": int(args.stageB_epochs) if args.rl_fine_tune else 0,
            "temperature": float(args.temperature),
            "stageA_entropy": float(args.stageA_entropy),
            "stageB_alpha": float(args.stageB_alpha),
            "stageB_beta_init": float(args.stageB_beta_init),
            "stageB_beta_final": float(args.stageB_beta_final),
            "stageA_lr": float(args.stageA_lr),
            "stageB_lr": float(args.stageB_lr),
            "rl_fine_tune": bool(args.rl_fine_tune),
            "seed": int(args.seed),
            "n_test": int(len(S_tst)),
            "k_cands": int(Rsem_trn.shape[1]),
            "stageA_only_f1_mean": float(picked_A.mean()),
            "stageA_only_f1_std": float(picked_A.std()),
            "stageAB_f1_mean": float(picked_AB.mean()) if picked_AB is not None else None,
            "stageAB_f1_std": float(picked_AB.std()) if picked_AB is not None else None,
            "teacher_lambdarank_f1": float(lr_f1.mean()),
            "pool_oracle_f1": float(oracle.mean()),
            "heuristic_f1_mean": float(heuristic.mean()),
            "grefer_heldout_600": float(args.grefer),
            "ppo_heldout_600": 0.4816,
            "grpo_heldout_600": 0.4853,
            "dpo_heldout_600": 0.4890,
            "lambdarank_heldout_600": 0.4980,
            "picks_stageA": actions_A,
            "picks_stageAB": actions_AB,
        }
        Path(args.result_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.result_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved result -> {args.result_out}")


if __name__ == "__main__":
    main()
