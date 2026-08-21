"""
Run 5-seed PPO / GRPO / DPO / Distillation on the paper_ppo_tables.npz
produced by the data-prep pipeline. Results land in
experiments/<variant>/results/seed<N>.json.

Usage:
    python3 pipeline/run_rl_variants.py              # all 4 × 5 seeds
    python3 pipeline/run_rl_variants.py --variants ppo,grpo
    python3 pipeline/run_rl_variants.py --seeds 42,43
    python3 pipeline/run_rl_variants.py --episodes 300
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import FINAL_ROOT, DATA_ROOT  # noqa: E402

TABLES = FINAL_ROOT / "results" / "paper_ppo_tables.npz"
EXP = FINAL_ROOT / "results" / "experiments"
RUNNERS = FINAL_ROOT / "runners"

GREFER_F1 = 0.4592

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
        "--grefer", str(GREFER_F1),
        "--result_out", str(result_path),
    ]
    if variant == "distillation":
        # distillation has its own stageA/stageB epoch args
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

    summary = {}
    for v in variants:
        if v not in VARIANTS:
            print(f"warning: skipping unknown variant '{v}'", file=sys.stderr)
            continue
        f1s = []
        for s in seeds:
            f1 = run_one(v, VARIANTS[v], s, args.episodes)
            if f1 is not None:
                f1s.append(f1)
        summary[v] = f1s
        if f1s:
            import numpy as np
            arr = np.array(f1s)
            print(f"\n[{v}] {len(arr)} seeds: mean={arr.mean():.4f} "
                  f"std={arr.std():.4f}  seeds={seeds}  f1s={f1s}")

    summary_path = EXP / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
