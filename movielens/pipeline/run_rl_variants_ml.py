# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
5-seed PPO / GRPO / DPO / Distillation sweep on MovieLens tables.

Mirrors pipeline/run_rl_variants.py but points at
MovieLens-specific paths:
  - Tables:  movielens/results/paper_ppo_tables.npz  (K=18)
  - Output:  movielens/results/experiments/<variant>/seed<N>.json

Runners (runners/) are unchanged — they read action_dim from
the table shape, so K=18 works without modification.

Usage:
    python3 movielens/pipeline/run_rl_variants_ml.py
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from _paths import FINAL_ROOT, DATA_ROOT  # noqa: E402

ML_ROOT = FINAL_ROOT / "movielens"
TABLES = ML_ROOT / "results" / "paper_ppo_tables.npz"
EXP = ML_ROOT / "results" / "experiments"
RUNNERS = FINAL_ROOT / "runners"

# Compare against the ensemble reranker baseline instead of G-Refer
ENSEMBLE_BASELINE = 0.3281

VARIANTS = {
    "ppo":          RUNNERS / "run_ppo.py",
    "grpo":         RUNNERS / "run_grpo.py",
    "dpo":          RUNNERS / "run_dpo.py",
    "distillation": RUNNERS / "run_distillation.py",
}


def run_one(variant, script, seed, episodes):
    out_dir = EXP / variant
    log_dir = EXP / variant / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"seed{seed}.json"
    history_path = out_dir / f"seed{seed}_history.json"
    log_path = log_dir / f"seed{seed}.log"

    cmd = [
        sys.executable, "-u", str(script),
        "--tables", str(TABLES),
        "--seed", str(seed),
        "--grefer", str(ENSEMBLE_BASELINE),
        "--result_out", str(result_path),
    ]
    if variant == "distillation":
        cmd += ["--stageA_epochs", "200", "--stageB_epochs", str(episodes)]
    else:
        cmd += ["--episodes", str(episodes)]
    if variant == "ppo":
        cmd += ["--reward_mode", "baseline", "--alpha", "0.1",
                "--history_out", str(history_path)]

    print(f"\n[{variant} seed={seed}] → {log_path}")
    start = time.time()
    with open(log_path, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=str(DATA_ROOT))
    mins = (time.time() - start) / 60
    if rc != 0:
        print(f"  FAILED (rc={rc}) after {mins:.1f} min — see {log_path}")
        return None
    if not result_path.exists():
        print(f"  WARNING: no result_out produced")
        return None
    r = json.load(open(result_path))
    f1 = (r.get("ppo_f1_mean") or r.get("grpo_f1_mean") or
          r.get("dpo_f1_mean") or r.get("stageAB_f1_mean") or
          r.get("stageA_only_f1_mean") or r.get("test_f1_mean"))
    print(f"  done ({mins:.1f} min)  test F1 = {f1}")
    return f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS.keys()))
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--episodes", type=int, default=500)
    args = ap.parse_args()

    if not TABLES.exists():
        print(f"error: tables missing at {TABLES}", file=sys.stderr)
        sys.exit(2)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    import numpy as np
    summary = {}
    for v in variants:
        if v not in VARIANTS:
            continue
        f1s = []
        for s in seeds:
            f1 = run_one(v, VARIANTS[v], s, args.episodes)
            if f1 is not None:
                f1s.append(f1)
        summary[v] = f1s
        if f1s:
            arr = np.array(f1s)
            print(f"\n[{v}] {len(arr)} seeds: mean={arr.mean():.4f} "
                  f"std={arr.std():.4f}")

    summary_path = EXP / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
