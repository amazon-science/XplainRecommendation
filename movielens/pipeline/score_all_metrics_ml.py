"""
Score MovieLens results with BERTScore-F1 + BARTScore + USR.

Reads:
  - movielens/results/lambdarank_result.json (has picked_texts + references)
  - movielens/results/experiments/<variant>/seed*.json (has picks)
  - movielens/results/features.pkl (for picks -> text mapping)

Writes movielens/results/experiments/summary_all_metrics.json.
"""
import glob
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from _paths import FINAL_ROOT, DATA_ROOT, setup_sys_path  # noqa: E402
setup_sys_path()

import numpy as np  # noqa: E402

sys.path.insert(0, str(DATA_ROOT / "G-Refer" / "evaluation"))
from scripts.rag_bandit_pipeline import bert_f1_batch  # noqa: E402

ML_ROOT = FINAL_ROOT / "movielens"
RESULTS = ML_ROOT / "results"

K_CANDS = 18
TABLE_SEED = 42


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
    """Reproduce the exact K-subsample sequence that build_tables.py used.
    That function sampled train first, then test, with a shared
    np.random.RandomState(seed). We must replay in the same order so the
    test rng state matches."""
    feat = pickle.load(open(features_pkl_path, "rb"))
    meta_tst = feat["meta_tst"]
    by_sample_trn = feat["by_sample_trn"]
    by_sample_tst = feat["by_sample_tst"]

    rng = np.random.RandomState(seed)

    # 1. Consume the train RNG state exactly as build_tables does.
    for sid in sorted(by_sample_trn.keys()):
        idxs = list(by_sample_trn[sid])
        if len(idxs) >= k:
            _ = rng.choice(idxs, size=k, replace=False)
        else:
            _ = rng.choice(idxs, size=k - len(idxs), replace=True)

    # 2. Now replay the test RNG — this matches Rsem_tst's column order.
    sample_ids = sorted(by_sample_tst.keys())
    cand_texts = {}
    references = {}
    for sid in sample_ids:
        group_idxs = list(by_sample_tst[sid])
        if len(group_idxs) >= k:
            chosen = rng.choice(group_idxs, size=k, replace=False)
        else:
            pad = rng.choice(group_idxs, size=k - len(group_idxs), replace=True)
            chosen = np.concatenate([group_idxs, pad])
        chosen = np.array(chosen, dtype=np.int64)
        cand_texts[sid] = [strip_marker(meta_tst[i][4]) for i in chosen]
        references[sid] = strip_marker(meta_tst[group_idxs[0]][5])
    return cand_texts, references, sample_ids


def texts_for_picks(picks, cand_texts, references, sample_ids):
    assert len(picks) == len(sample_ids), f"{len(picks)} vs {len(sample_ids)}"
    preds = [cand_texts[sid][int(a)] for sid, a in zip(sample_ids, picks)]
    refs = [references[sid] for sid in sample_ids]
    return preds, refs


def main():
    features_pkl = RESULTS / "features.pkl"
    print(f"Loading features from {features_pkl}...")
    cand_texts, references, sample_ids = build_pick_to_text_lookup(features_pkl)
    print(f"  built lookup: {len(sample_ids)} samples, K={K_CANDS}")

    summary = {}
    scorer = None

    # --- LambdaRank ---
    lr_path = RESULTS / "lambdarank_result.json"
    if lr_path.exists():
        print("\n=== LambdaRank ===")
        lr = json.load(open(lr_path))
        preds = [strip_marker(p) for p in lr["picked_texts"]]
        refs = [strip_marker(r) for r in lr["references"]]
        summary["lambdarank"], scorer = score_three(preds, refs, "lambdarank", scorer)

    def per_seed_entries(variant, picks_key):
        entries = []
        for seed_path in sorted(
                (RESULTS / "experiments" / variant).glob("seed*.json")):
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
            out, scorer = score_three(preds, refs, f"{variant_label} {stem}", scorer)
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
    r = rollup(per_seed_entries("distillation", "picks_stageA"),
               "distillation_stageA")
    if r:
        summary["distillation_stageA_5seed"] = r

    print("\n=== DISTILLATION (Stage A+B) ===")
    r = rollup(per_seed_entries("distillation", "picks_stageAB"),
               "distillation_stageAB")
    if r:
        summary["distillation_stageAB_5seed"] = r

    out_path = RESULTS / "experiments" / "summary_all_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
