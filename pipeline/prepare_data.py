# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
One-shot orchestrator: turn XRec's published Google-Local data into the
training and inference tables this project consumes.

Runs the data-prep pipeline described in README.md.
Each stage caches its output; re-running skips work that's already done.

    Stage 1  verify      split integrity              (~1 min,     $0)
    Stage 2  sample      deterministic 5k train draw  (~30 sec,    $0)
    Stage 3  pool        candidate generation         (~3 hours,   ~$11 Bedrock)
    Stage 4  featurize   features + BERTScore labels  (~4 hours,   ~$0.40)
    Stage 5  tables      LambdaRank + PPO state/rwd   (~2 min,     ~$0.20)

Usage:
    python3 pipeline/prepare_data.py            # run everything
    python3 pipeline/prepare_data.py --stages verify,sample
    python3 pipeline/prepare_data.py --force-stage pool
    python3 pipeline/prepare_data.py --max_workers 4

Each stage is a thin wrapper over the standalone script of the same
name — if you prefer to run them one at a time, the standalone scripts
in this folder do the exact same thing.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import FINAL_ROOT, DATA_ROOT  # noqa: E402

SCRIPTS = FINAL_ROOT / "pipeline"
RESULTS = FINAL_ROOT / "results"

STAGES = [
    # (id, script, output_marker_path, approx_wallclock_min, approx_cost_usd)
    ("verify",    "verify_splits.py",                 None,                                      1,    0.0),
    ("sample",    "sample_training.py",               RESULTS / "trn_5k.pkl",                    1,    0.0),
    ("pool",      "build_pool.py",                    RESULTS / "pool" / "exp_pool_clean_xrec_tst_n3000_s6_b4_usamazonnova-l_anthropicclaud_42.pkl",
                                                                                                 180,  11.0),
    ("featurize", "featurize_and_label.py",           RESULTS / "features" / "features.pkl",    240,  0.4),
    ("tables",    "fit_lambdarank_and_build_tables.py", RESULTS / "paper_ppo_tables.npz",         2,    0.2),
]


def run_stage(stage_id, script_name, extra_args, log_path):
    cmd = [sys.executable, "-u", str(SCRIPTS / script_name), *extra_args]
    print(f"\n{'='*70}\n[{stage_id}] launching: {' '.join(cmd)}\n  log → {log_path}\n{'='*70}")
    start = time.time()
    with open(log_path, "w") as logf:
        rc = subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(DATA_ROOT))
    elapsed = time.time() - start
    if rc != 0:
        print(f"[{stage_id}] FAILED (rc={rc}) after {elapsed/60:.1f} min — see {log_path}")
        sys.exit(rc)
    print(f"[{stage_id}] ok ({elapsed/60:.1f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stages",
        default=",".join(s[0] for s in STAGES),
        help="comma-separated stage ids to run in order (default: all)",
    )
    ap.add_argument(
        "--force-stage",
        action="append",
        default=[],
        help="re-run this stage even if its output already exists "
             "(pass multiple times for multiple stages)",
    )
    ap.add_argument("--max_workers", type=int, default=4,
                    help="parallel workers for Bedrock calls (pool + featurize)")
    ap.add_argument("--log-dir", default=str(FINAL_ROOT / "logs"),
                    help="directory for stage logs")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    wanted = [s.strip() for s in args.stages.split(",") if s.strip()]
    known = {s[0] for s in STAGES}
    unknown = set(wanted) - known
    if unknown:
        print(f"error: unknown stage(s): {unknown}. Known: {sorted(known)}", file=sys.stderr)
        sys.exit(2)

    stage_map = {s[0]: s for s in STAGES}
    forced = set(args.force_stage)

    total_est_min = sum(stage_map[s][3] for s in wanted)
    total_est_cost = sum(stage_map[s][4] for s in wanted)
    print(f"Plan: {wanted}")
    print(f"Estimated wall-clock: ~{total_est_min/60:.1f} hours (~{total_est_min} min)")
    print(f"Estimated Bedrock cost: ~${total_est_cost:.2f}")

    for stage_id in wanted:
        _, script, marker, _, _ = stage_map[stage_id]

        if marker is not None and marker.exists() and stage_id not in forced:
            print(f"[{stage_id}] skipping — output already exists: {marker}")
            continue

        extra = []
        if stage_id in ("pool", "featurize"):
            extra = ["--max_workers", str(args.max_workers)]

        log_path = log_dir / f"prepare_data_{stage_id}.log"
        run_stage(stage_id, script, extra, log_path)

        if marker is not None and not marker.exists():
            print(f"[{stage_id}] WARNING: expected output {marker} not produced", file=sys.stderr)

    print("\nData prep complete.")
    print("Next: train a picker — see README.md §6 (RL variants).")


if __name__ == "__main__":
    main()
