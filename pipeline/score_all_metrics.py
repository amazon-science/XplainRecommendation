"""
Score every method's picked explanations with all three metrics:
BERTScore-F1, BARTScore, USR.

Writes rolled-up results/experiments/summary_all_metrics.json.

For LambdaRank: uses results/lambdarank_result.json (has picked_texts + references).
For PPO/GRPO/DPO/Distillation: loads `picks` (action index 0..K-1) from each
  seedN.json, reproduces the K=40 rng.choice mapping from seed=42 that built
  paper_ppo_tables.npz, and looks up the candidate text in features.pkl.

Run as:
    python3 pipeline/score_all_metrics.py
"""
import glob
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import FINAL_ROOT, DATA_ROOT, setup_sys_path  # noqa: E402
setup_sys_path()

import numpy as np  # noqa: E402

# G-Refer's BARTScore lives outside 
sys.path.insert(0, str(DATA_ROOT / "G-Refer" / "evaluation"))

from scripts.rag_bandit_pipeline import bert_f1_batch  # noqa: E402

K_CANDS = 40
TABLE_SEED = 42  # must match the seed used in fit_lambdarank_and_build_tables.py


def usr_metric(sequences):
    unique = []
    for seq in sequences:
        if any(len(u) == len(seq) and all(a == b for a, b in zip(u, seq))
               for u in unique):
            continue
        unique.append(seq)
    return len(unique) / max(len(sequences), 1), len(unique)


def bartscore_list(preds, refs, batch_size=4, device="cpu", scorer=None):
    from tqdm import tqdm
    if scorer is None:
        from bart_score import BARTScorer
        scorer = BARTScorer(device=device, checkpoint="facebook/bart-large-cnn")
    out = []
    for i in tqdm(range(0, len(preds), batch_size), desc="BARTScore"):
        out.extend(scorer.score(preds[i:i + batch_size],
                                 refs[i:i + batch_size], batch_size=batch_size))
    return np.array(out), scorer


def strip_marker(s):
    if not isinstance(s, str):
        return ""
    if "### " in s:
        return s.split("### ", 1)[-1].strip()
    return s.strip()


def score_three(preds, refs, label, scorer=None):
    f1 = bert_f1_batch(preds, refs)
    usr, n_uniq = usr_metric([p.split() for p in preds])
    bart, scorer = bartscore_list(preds, refs, device="cpu", scorer=scorer)
    out = {
        "label": label,
        "n_samples": len(preds),
        "bertscore_f1_mean": float(f1.mean()),
        "bertscore_f1_std": float(f1.std()),
        "bartscore_mean": float(bart.mean()),
        "bartscore_std": float(bart.std()),
        "usr": usr,
        "n_unique": n_uniq,
    }
    print(f"  {label:40s}  F1={out['bertscore_f1_mean']:.4f}  "
          f"BART={out['bartscore_mean']:.4f}  "
          f"USR={usr:.4f} ({n_uniq}/{len(preds)})")
    return out, scorer


def build_pick_to_text_lookup(features_pkl_path, seed=TABLE_SEED, k=K_CANDS):
    """Reproduce the K=40 rng.choice sequence from fit_lambdarank_and_build_tables
    so we can map (sample_idx, action_in_K) -> (candidate_text, reference_text).

    Returns: cand_lookup[sid] -> list of k texts, refs[sid] -> reference text.
    """
    feat = pickle.load(open(features_pkl_path, "rb"))
    meta_tst = feat["meta_tst"]
    by_sample_tst = feat["by_sample_tst"]

    sample_ids = sorted(by_sample_tst.keys())
    rng = np.random.RandomState(seed)
    cand_texts = {}       # sid -> list of k candidate texts
    references = {}       # sid -> reference text
    for sid in sample_ids:
        group_idxs = list(by_sample_tst[sid])
        if len(group_idxs) >= k:
            chosen = rng.choice(group_idxs, size=k, replace=False)
        else:
            chosen = rng.choice(group_idxs, size=k, replace=True)
        chosen = np.array(chosen, dtype=np.int64)
        # meta row: (sample_idx, model, style, j, prediction, reference)
        cand_texts[sid] = [strip_marker(meta_tst[i][4]) for i in chosen]
        references[sid] = strip_marker(meta_tst[group_idxs[0]][5])
    return cand_texts, references, sample_ids


def texts_for_picks(picks, cand_texts, references, sample_ids):
    """picks: list of length N_test, each entry is an action index 0..K-1.
    Returns preds[], refs[] (parallel arrays)."""
    assert len(picks) == len(sample_ids), f"{len(picks)} vs {len(sample_ids)}"
    preds = []
    refs = []
    for sid, a in zip(sample_ids, picks):
        preds.append(cand_texts[sid][int(a)])
        refs.append(references[sid])
    return preds, refs


def main():
    results_dir = FINAL_ROOT / "results"
    summary = {}
    scorer = None

    # Build pick-to-text lookup once
    # features.pkl is large; try local first, then fall back to the source
    features_pkl = FINAL_ROOT / "results" / "features" / "features.pkl"
    if not features_pkl.exists():
        features_pkl = DATA_ROOT / "clean_xrec_data" / "results" / "features" / "features.pkl"
    print(f"Loading features from {features_pkl}...")
    cand_texts, references, sample_ids = build_pick_to_text_lookup(features_pkl)
    print(f"  built lookup: {len(sample_ids)} samples, K={K_CANDS}")

    # --- LambdaRank ---
    lr_path = results_dir / "lambdarank_result.json"
    if lr_path.exists():
        print("\n=== LambdaRank ===")
        lr = json.load(open(lr_path))
        preds = [strip_marker(p) for p in lr["picked_texts"]]
        refs = [strip_marker(r) for r in lr["references"]]
        summary["lambdarank"], scorer = score_three(preds, refs, "lambdarank", scorer)

    # --- RL variants (picks live inside seedN.json) ---
    def per_seed_entries(variant, picks_key):
        entries = []
        for seed_path in sorted(
                (results_dir / "experiments" / variant).glob("seed*.json")):
            if "history" in seed_path.name:
                continue
            seed_data = json.load(open(seed_path))
            picks = seed_data.get(picks_key)
            if picks is None:
                print(f"  [skip] {seed_path.name} — '{picks_key}' missing")
                continue
            preds, refs = texts_for_picks(picks, cand_texts, references, sample_ids)
            entries.append((seed_path.stem, preds, refs))
        return entries

    def rollup(entries, variant_label):
        nonlocal scorer
        per_seed = []
        for stem, preds, refs in entries:
            label = f"{variant_label} {stem}"
            out, scorer = score_three(preds, refs, label, scorer)
            per_seed.append(out)
        if not per_seed:
            return None
        f1s = np.array([s["bertscore_f1_mean"] for s in per_seed])
        barts = np.array([s["bartscore_mean"] for s in per_seed])
        usrs = np.array([s["usr"] for s in per_seed])
        return {
            "n_seeds": len(per_seed),
            "bertscore_f1_mean": float(f1s.mean()),
            "bertscore_f1_std": float(f1s.std()),
            "bartscore_mean": float(barts.mean()),
            "bartscore_std": float(barts.std()),
            "usr_mean": float(usrs.mean()),
            "usr_std": float(usrs.std()),
            "per_seed": per_seed,
        }

    for variant, picks_key in [("ppo", "picks"), ("grpo", "picks"),
                                ("dpo", "picks")]:
        print(f"\n=== {variant.upper()} ===")
        entries = per_seed_entries(variant, picks_key)
        r = rollup(entries, variant)
        if r:
            summary[variant + "_5seed"] = r

    print("\n=== DISTILLATION (Stage A only) ===")
    entries = per_seed_entries("distillation", "picks_stageA")
    r = rollup(entries, "distillation_stageA")
    if r:
        summary["distillation_stageA_5seed"] = r

    print("\n=== DISTILLATION (Stage A+B) ===")
    entries = per_seed_entries("distillation", "picks_stageAB")
    r = rollup(entries, "distillation_stageAB")
    if r:
        summary["distillation_stageAB_5seed"] = r

    # --- Baselines (already triple-metric in baselines.json) ---
    baselines_path = results_dir / "baselines.json"
    if baselines_path.exists():
        summary["baselines"] = json.load(open(baselines_path))

    out_path = results_dir / "experiments" / "summary_all_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
