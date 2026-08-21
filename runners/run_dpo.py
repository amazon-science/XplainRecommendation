"""
DPO (Direct Preference Optimization, Rafailov et al. 2023) on the same
explanation-selection bandit.

At every training sample i with K cached candidates and known
BERTScore-F1 labels R_sem[i, :], we sample a preference pair (a+, a-)
where F1(a+) > F1(a-), and minimize the DPO loss:

    L = -log σ( β · [ (log π_θ(a+|s) - log π_θ(a-|s))
                    - (log π_ref(a+|s) - log π_ref(a-|s)) ] )

Weighted by the F1 gap so large-gap pairs contribute more.

π_ref = a frozen snapshot of the policy at the start of training
(classical DPO uses the supervised-finetuned model as the reference;
here we start from a randomly-initialized policy and freeze a copy
at ep=0, then compare against it throughout — this is the "online
DPO / Online DPO-aligned" variant that trains from scratch).

Differences from PPO:
- No value network
- No reward-weighted advantage — the signal is pure pairwise preference
- KL regularization is implicit via the reference-model anchoring
- Clean pairwise learning-to-rank framing, still gradient-based RL

Usage (from approach_ppo_0.4864/experiments/variantE_dpo/):
    python3 scripts/run_dpo.py --seed 42 --episodes 500 \\
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


def build_pair_table(Rsem, n_pairs_per_sample=80, min_gap=2.0, seed=42):
    """For each training sample, select up to n_pairs_per_sample (a+, a-)
    pairs where F1 gap (in Rsem units, 0-100 scale) exceeds min_gap.

    Returns (sample_idx, a_plus, a_minus, gap) arrays.
    """
    rng = np.random.RandomState(seed)
    N, K = Rsem.shape
    s_all, ap_all, am_all, gap_all = [], [], [], []
    for i in range(N):
        order = np.argsort(-Rsem[i])  # descending
        # Consider only above-median candidates as potential positives to
        # avoid trivial losses.
        picked = 0
        # Random walk over plus/minus with rejection on gap
        for _ in range(n_pairs_per_sample * 4):  # 4x tries in case gaps small
            a_plus = int(rng.choice(order[:K//2]))      # top half
            a_minus = int(rng.choice(order[K//2:]))     # bottom half
            gap = float(Rsem[i, a_plus] - Rsem[i, a_minus])
            if gap >= min_gap:
                s_all.append(i)
                ap_all.append(a_plus)
                am_all.append(a_minus)
                gap_all.append(gap)
                picked += 1
                if picked >= n_pairs_per_sample:
                    break
    return (np.array(s_all, dtype=np.int64),
            np.array(ap_all, dtype=np.int64),
            np.array(am_all, dtype=np.int64),
            np.array(gap_all, dtype=np.float32))


def train_dpo(S, Rsem, Rstr, alpha, episodes, seed,
              dpo_beta, n_pairs_per_sample, min_gap,
              entropy_init=0.01, entropy_final=0.001,
              weight_by_gap=True,
              verbose=True):
    """
    Train a policy with DPO on pairwise preferences derived from Rsem.

    Args:
      dpo_beta: temperature in the DPO log-sigmoid loss. 0.1 is standard.
      n_pairs_per_sample: max pairs to generate per training state per call
      min_gap: minimum F1-gap (in Rsem units, 0-100) to accept a pair
      weight_by_gap: if True, pair loss weighted by (gap / mean_gap)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = S.shape[1]
    action_dim = Rsem.shape[1]

    policy = PolicyNetwork(state_dim, action_dim, hidden_dim=256)
    policy_ref = copy.deepcopy(policy)
    for p in policy_ref.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    print(f"Initialized DPO")
    print(f"  policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  dpo_beta={dpo_beta}  n_pairs/sample={n_pairs_per_sample}  "
          f"min_gap={min_gap}  weight_by_gap={weight_by_gap}")

    # Pre-compute a fixed pool of pairs once, using the true F1 labels.
    # (These labels are train-side only; no test leakage.)
    print("Building DPO preference pairs...")
    s_idx, a_plus, a_minus, gap = build_pair_table(
        Rsem, n_pairs_per_sample=n_pairs_per_sample,
        min_gap=min_gap, seed=seed,
    )
    print(f"  built {len(s_idx):,} preference pairs")
    gap_mean = float(gap.mean())

    S_t = torch.from_numpy(S).float()
    n_pairs = len(s_idx)
    rng = np.random.RandomState(seed)

    # DPO training: at each "episode" we sample a batch of pairs,
    # compute the DPO loss, and update. 500 episodes × 4 inner-epoch-like
    # updates to match PPO compute budget.
    batch_size = min(4096, n_pairs)
    history = []

    for ep in range(episodes):
        # Linear entropy schedule: DPO doesn't need exploration but a
        # small entropy bonus stabilizes on-policy-adjacent fine-tunes
        # of randomly-initialised policies.
        prog = ep / max(episodes - 1, 1)
        ent_coef = entropy_init * (1 - prog) + entropy_final * prog

        # Sample a batch of pair indices
        idx = rng.choice(n_pairs, size=batch_size, replace=False)
        sb = s_idx[idx]
        apb = a_plus[idx]
        amb = a_minus[idx]
        gapb = gap[idx]

        states = S_t[sb]
        actions_plus = torch.from_numpy(apb).long()
        actions_minus = torch.from_numpy(amb).long()

        # Current-policy log-probs
        probs = policy(states)
        logp = torch.log(probs + 1e-12)
        logp_plus = logp.gather(1, actions_plus.unsqueeze(1)).squeeze(1)
        logp_minus = logp.gather(1, actions_minus.unsqueeze(1)).squeeze(1)

        # Reference-policy log-probs
        with torch.no_grad():
            probs_ref = policy_ref(states)
            logp_ref = torch.log(probs_ref + 1e-12)
            logp_ref_plus = logp_ref.gather(1, actions_plus.unsqueeze(1)).squeeze(1)
            logp_ref_minus = logp_ref.gather(1, actions_minus.unsqueeze(1)).squeeze(1)

        # DPO log-ratio of ratios
        logits = dpo_beta * (
            (logp_plus - logp_minus) - (logp_ref_plus - logp_ref_minus)
        )
        per_pair_loss = -torch.nn.functional.logsigmoid(logits)

        if weight_by_gap:
            weights = torch.from_numpy(gapb / gap_mean).float()
            loss = (per_pair_loss * weights).mean()
        else:
            loss = per_pair_loss.mean()

        # Entropy bonus (on all candidates, to keep exploration alive on
        # unchosen actions — prevents policy collapse to 2 actions)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()
        total_loss = loss - ent_coef * entropy

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

        if verbose and ((ep + 1) % 50 == 0 or ep == 0):
            # Track: mean pair-margin on training pool
            with torch.no_grad():
                # Use a recent-progress proxy: fraction of pairs where
                # logits > 0 (policy prefers positive)
                pref_correct = float((logits > 0).float().mean().item())
            print(f"  [ep {ep+1:3d}] loss={loss.item():+.4f}  "
                  f"ent={entropy.item():.3f}  ent_c={ent_coef:.4f}  "
                  f"pref_acc={pref_correct:.3f}")
            history.append({
                'ep': ep + 1, 'loss': float(loss.item()),
                'entropy': float(entropy.item()), 'ent_coef': ent_coef,
                'pref_accuracy': pref_correct,
            })

    return policy, history


def evaluate(policy, S, Rsem, oracle, heuristic, grefer=0.4611):
    picked = []
    S_t = torch.from_numpy(S).float()
    with torch.no_grad():
        probs = policy(S_t)
        actions = probs.argmax(dim=-1).tolist()
    for i, a in enumerate(actions):
        picked.append(Rsem[i, a] / 100.0)
    picked = np.array(picked)
    print(f"\n=== Test BERTScore-F1 (n={len(S)}) ===")
    print(f"  DPO (this run)             : {picked.mean():.4f} +- {picked.std():.4f}")
    print(f"  Heuristic (argmax R_struct): {heuristic.mean():.4f}")
    print(f"  Pool oracle                : {oracle.mean():.4f}")
    print(f"  G-Refer (held-out 600)     : {grefer:.4f}")
    print(f"  delta vs G-Refer           : {picked.mean() - grefer:+.4f}")
    print(f"  delta vs PPO (0.4816)      : {picked.mean() - 0.4816:+.4f}")
    print(f"  delta vs GRPO (0.4853)     : {picked.mean() - 0.4853:+.4f}")
    return picked, actions


def main():
    ap = argparse.ArgumentParser()
    default_tables = FINAL_ROOT / "results" / "paper_ppo_tables.npz"
    ap.add_argument("--tables", type=str, default=str(default_tables))
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="(kept for API parity; unused in DPO)")
    ap.add_argument("--dpo_beta", type=float, default=0.1,
                    help="DPO temperature; 0.1 is standard.")
    ap.add_argument("--n_pairs_per_sample", type=int, default=80)
    ap.add_argument("--min_gap", type=float, default=2.0,
                    help="Minimum F1-gap (in Rsem 0-100 units) to include a pair.")
    ap.add_argument("--no_weight_by_gap", action="store_true")
    ap.add_argument("--entropy_init", type=float, default=0.01)
    ap.add_argument("--entropy_final", type=float, default=0.001)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grefer", type=float, default=0.4611)
    ap.add_argument("--result_out", type=str, default=None)
    ap.add_argument("--history_out", type=str, default=None)
    args = ap.parse_args()

    z = np.load(args.tables)
    S_trn, Rsem_trn, Rstr_trn = z["S_trn"], z["Rsem_trn"], z["Rstr_trn"]
    S_tst, Rsem_tst, Rstr_tst = z["S_tst"], z["Rsem_tst"], z["Rstr_tst"]
    print(f"Loaded tables: S_trn={S_trn.shape}  S_tst={S_tst.shape}  K={Rsem_trn.shape[1]}")
    print(f"  episodes={args.episodes}  seed={args.seed}")

    oracle = Rsem_tst.max(axis=1) / 100.0
    heuristic_idx = Rstr_tst.argmax(axis=1)
    heuristic = Rsem_tst[np.arange(len(heuristic_idx)), heuristic_idx] / 100.0

    policy, history = train_dpo(
        S_trn, Rsem_trn, Rstr_trn,
        alpha=args.alpha, episodes=args.episodes, seed=args.seed,
        dpo_beta=args.dpo_beta,
        n_pairs_per_sample=args.n_pairs_per_sample,
        min_gap=args.min_gap,
        entropy_init=args.entropy_init,
        entropy_final=args.entropy_final,
        weight_by_gap=(not args.no_weight_by_gap),
    )
    picked, actions = evaluate(policy, S_tst, Rsem_tst, oracle, heuristic, grefer=args.grefer)

    if args.history_out:
        Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.history_out, "w") as f:
            json.dump(history, f, indent=2)
        print(f"  Saved history -> {args.history_out}")

    if args.result_out:
        out = {
            "method": "dpo",
            "tables": args.tables,
            "dpo_beta": float(args.dpo_beta),
            "n_pairs_per_sample": int(args.n_pairs_per_sample),
            "min_gap": float(args.min_gap),
            "weight_by_gap": (not args.no_weight_by_gap),
            "entropy_init": float(args.entropy_init),
            "entropy_final": float(args.entropy_final),
            "episodes": int(args.episodes),
            "seed": int(args.seed),
            "n_test": int(len(S_tst)),
            "k_cands": int(Rsem_trn.shape[1]),
            "dpo_f1_mean": float(picked.mean()),
            "dpo_f1_std": float(picked.std()),
            "pool_oracle_f1": float(oracle.mean()),
            "heuristic_f1_mean": float(heuristic.mean()),
            "grefer_heldout_600": float(args.grefer),
            "ppo_heldout_600_5seed_mean": 0.4816,
            "grpo_heldout_600_5seed_mean": 0.4853,
            "picks": actions,
        }
        Path(args.result_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.result_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved result -> {args.result_out}")


if __name__ == "__main__":
    main()
