# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Reranker v3: pairwise learning-to-rank (LambdaRank via LightGBM).

Same candidate pool + features as reranker_v2, but:
  - Labels are true BERTScore F1 per candidate (same as v2).
  - Loss is pairwise rank-based (LambdaRank) instead of pointwise MSE.
  - Groups = per-sample candidate pools — ranking is only relative within
    a sample, which is exactly what we want for argmax-per-sample rerank.

Caches the generated candidate pool to disk so repeat runs are fast.
"""

import argparse, json, os, re, sys, pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.reranker_v2 import (
    load_reviews, extract_user_profile, extract_biz_profile, filter_reviews,
    build_prompt_A, build_prompt_B, generate_pool, featurize, feature_names,
    _pred_features,
)
from src.bedrock_llm import BedrockLLM
from scripts.rag_bandit_pipeline import (
    load_grefer_samples, bert_f1_batch, embed_texts_titan,
)


def cache_key(split, n, n_single, n_synth, models, seed):
    mkey = "_".join(m.split("/")[-1].split(":")[0].replace(".", "")[:14]
                    for m in models)
    return f"pool_{split}_n{n}_s{n_single}_b{n_synth}_{mkey}_{seed}.pkl"


def build_or_load_pool(split_name, samples, models, n_single, n_synth,
                       max_workers, iid_reviews, ref_centroid, cache_path,
                       embed_llm, clients, args, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_dir / cache_key(split_name, len(samples), n_single, n_synth,
                                 models, args.seed)
    if key.exists():
        print(f"  Loading cached pool: {key.name}")
        with open(key, "rb") as f:
            return pickle.load(f)
    pack = generate_pool(
        samples, models, n_single, n_synth, max_workers,
        iid_reviews, ref_centroid, cache_path, embed_llm, clients,
    )
    with open(key, "wb") as f:
        pickle.dump(pack, f)
    return pack


def fit_lambdarank(X_trn, y_trn, group_trn, seed, n_estimators=500,
                    num_leaves=31, learning_rate=0.05):
    import lightgbm as lgb
    # LightGBM needs integer-labeled relevance for lambdarank. Quantize y into
    # 5 bins within each sample so pairs respect *relative* F1 within the group.
    # Global quantization would let pairs from different samples dominate.
    y_rel = np.zeros(len(y_trn), dtype=int)
    idx = 0
    for g in group_trn:
        block = y_trn[idx:idx + g]
        # Rank within the group: best -> 4, next -> 3, ..., worst -> 0 (cap at 4)
        order = np.argsort(-block)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))
        # Map to 0-4 (higher = better). Group of size 15 -> top 3 = 4, next 3 = 3, ...
        bin_size = max(1, len(block) // 5)
        lvl = np.minimum(4, 4 - (ranks // bin_size))
        lvl = np.maximum(lvl, 0)
        y_rel[idx:idx + g] = lvl
        idx += g
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        min_child_samples=10,
        random_state=seed,
        verbosity=-1,
    )
    ranker.fit(X_trn, y_rel, group=group_trn)
    return ranker


def _groups_from_meta(meta):
    """Returns (group_sizes_in_order, sample_idx_to_mask_range).

    meta is ordered by featurize() as (sample_0, sample_1, ...). Return the
    number of candidates per sample in that same order."""
    from collections import OrderedDict
    per_sample = OrderedDict()
    for k, m in enumerate(meta):
        per_sample.setdefault(m[0], []).append(k)
    # Verify meta is contiguous by sample_i
    flat = []
    for ks in per_sample.values():
        flat.extend(ks)
    assert flat == list(range(len(meta))), "meta must be grouped contiguously by sample"
    return [len(ks) for ks in per_sample.values()], per_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--n_single", type=int, default=4)
    ap.add_argument("--n_synth", type=int, default=8)
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", type=str, nargs="+", default=[
        "us.amazon.nova-lite-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    ])
    args = ap.parse_args()

    trn, tst = load_grefer_samples()
    rng = np.random.RandomState(args.seed)
    trn_sel = rng.choice(len(trn), size=args.n_train, replace=False)
    trn_eval = [trn[i] for i in trn_sel]
    tst_sel = rng.choice(len(tst), size=args.n_test, replace=False)
    tst_eval = [tst[i] for i in tst_sel]

    iid_reviews = load_reviews()
    cache_path = Path("results/rag_bandit/titan_cache.json")
    pool_cache = Path("results/rag_bandit/pool_cache")

    clients = {m: BedrockLLM(model_id=m, max_tokens=300, temperature=0.0)
               for m in args.models}
    embed_llm = clients[args.models[0]]

    print("Building reference-style centroid from ALL train refs...")
    all_trn_refs = [t.reference.replace("### ", "").strip() for t in trn]
    trn_ref_embs = embed_texts_titan(
        embed_llm, all_trn_refs, cache_path=cache_path,
        max_workers=args.max_workers,
    )
    ref_centroid = trn_ref_embs.mean(axis=0)
    ref_centroid /= (np.linalg.norm(ref_centroid) + 1e-9)
    trn_ref_embs_norm = trn_ref_embs / (
        np.linalg.norm(trn_ref_embs, axis=1, keepdims=True) + 1e-9)
    sims_to_centroid = trn_ref_embs_norm @ ref_centroid
    exemplar_ref = all_trn_refs[int(np.argmax(sims_to_centroid))]

    print(f"\n=== STAGE 1: Train pool (n={args.n_train}) ===")
    trn_pack = build_or_load_pool(
        "train", trn_eval, args.models, args.n_single, args.n_synth,
        args.max_workers, iid_reviews, ref_centroid, cache_path,
        embed_llm, clients, args, pool_cache,
    )
    print(f"\n=== STAGE 2: Test pool (n={args.n_test}) ===")
    tst_pack = build_or_load_pool(
        "test", tst_eval, args.models, args.n_single, args.n_synth,
        args.max_workers, iid_reviews, ref_centroid, cache_path,
        embed_llm, clients, args, pool_cache,
    )

    print(f"\n=== STAGE 3: Featurize ===")
    X_trn, meta_trn = featurize(trn_pack, embed_llm, ref_centroid, exemplar_ref,
                                 trn_ref_embs_norm, cache_path, args.max_workers)
    X_tst, meta_tst = featurize(tst_pack, embed_llm, ref_centroid, exemplar_ref,
                                 trn_ref_embs_norm, cache_path, args.max_workers)
    print(f"  X_train: {X_trn.shape}  X_test: {X_tst.shape}")

    print(f"\n=== STAGE 4: True-F1 labels ===")
    y_trn = bert_f1_batch([m[4] for m in meta_trn], [m[5] for m in meta_trn])
    y_tst = bert_f1_batch([m[4] for m in meta_tst], [m[5] for m in meta_tst])
    print(f"  train mean={y_trn.mean():.4f}  test mean={y_tst.mean():.4f}")

    grp_trn, by_sample_trn = _groups_from_meta(meta_trn)
    grp_tst, by_sample_tst = _groups_from_meta(meta_tst)
    oracle_trn = np.array([max(y_trn[ks]) for ks in by_sample_trn.values()])
    oracle_tst = np.array([max(y_tst[ks]) for ks in by_sample_tst.values()])
    print(f"  train oracle ceiling: {oracle_trn.mean():.4f}")
    print(f"  test  oracle ceiling: {oracle_tst.mean():.4f}")

    print(f"\n=== STAGE 5a: Fit pointwise GBR (baseline) ===")
    from sklearn.ensemble import GradientBoostingRegressor
    gbr = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                     learning_rate=0.05, random_state=args.seed)
    gbr.fit(X_trn, y_trn)
    pt_pred = gbr.predict(X_tst)

    print(f"\n=== STAGE 5b: Fit LightGBM LambdaRank (pairwise) ===")
    ranker = fit_lambdarank(X_trn, y_trn, grp_trn, seed=args.seed)
    rk_pred = ranker.predict(X_tst)

    feat_names = feature_names(args.models)
    print("\n  LambdaRank feature importances (top 12):")
    imps = ranker.feature_importances_
    order = np.argsort(-imps)
    for k in order[:12]:
        print(f"    {feat_names[k]:<34} {imps[k]}")

    print(f"\n=== STAGE 6: Evaluate ===")
    proxy_texts = [
        f"{tst_pack['biz_profiles'][m[0]]} {tst_pack['user_profiles'][m[0]]} {exemplar_ref}".strip()
        for m in meta_tst
    ]
    print("  Computing proxy_f1...")
    proxy_f1 = bert_f1_batch([m[4] for m in meta_tst], proxy_texts)

    picks = {"pointwise": [], "pairwise": [], "proxy": [], "oracle": []}
    from collections import Counter
    pairwise_picks = Counter()
    pointwise_picks = Counter()
    for i, ks in by_sample_tst.items():
        ks = list(ks)
        kp = ks[int(np.argmax(pt_pred[ks]))]
        kr = ks[int(np.argmax(rk_pred[ks]))]
        kx = ks[int(np.argmax(proxy_f1[ks]))]
        ko = ks[int(np.argmax(y_tst[ks]))]
        picks["pointwise"].append(y_tst[kp])
        picks["pairwise"].append(y_tst[kr])
        picks["proxy"].append(y_tst[kx])
        picks["oracle"].append(y_tst[ko])
        pairwise_picks[(meta_tst[kr][1], meta_tst[kr][2])] += 1
        pointwise_picks[(meta_tst[kp][1], meta_tst[kp][2])] += 1

    def summarize(name, vals):
        arr = np.array(vals)
        print(f"  {name:<10}: {arr.mean():.4f} ± {arr.std():.4f}  "
              f"%>=0.50 {100*(arr>=0.5).mean():.1f}%  "
              f"%>=0.55 {100*(arr>=0.55).mean():.1f}%  "
              f"%>=0.60 {100*(arr>=0.6).mean():.1f}%")

    n = len(picks["pairwise"])
    print(f"\n  === Test F1 (n={n}) ===")
    summarize("PAIRWISE",  picks["pairwise"])
    summarize("POINTWISE", picks["pointwise"])
    summarize("PROXY",     picks["proxy"])
    summarize("ORACLE",    picks["oracle"])

    print(f"\n  Pairwise picks (model, style):  {dict(pairwise_picks)}")
    print(f"  Pointwise picks (model, style): {dict(pointwise_picks)}")

    # Wins of pairwise over pointwise
    pw = np.array(picks["pairwise"])
    pt = np.array(picks["pointwise"])
    wins = int((pw > pt).sum())
    ties = int((pw == pt).sum())
    print(f"\n  Pairwise beats pointwise on {wins}/{n} ({100*wins/n:.0f}%), "
          f"ties {ties}")

    # Bucket oracle for reference
    per_bucket_oracle = {}
    for b in set((m[1], m[2]) for m in meta_tst):
        ks = [k for k, m in enumerate(meta_tst) if (m[1], m[2]) == b]
        if ks:
            per_bucket_oracle[b] = float(np.mean(y_tst[ks]))
    print(f"\n  Mean F1 per (model, style) bucket:")
    for b, v in sorted(per_bucket_oracle.items(), key=lambda x: -x[1]):
        print(f"    {b[0]} [{b[1]}]: {v:.4f}")

    out_dir = Path("results/rag_bandit")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"reranker_pairwise_{args.n_train}tr_{args.n_test}te.json"
    with open(out_file, "w") as f:
        json.dump({
            "summary": {
                "n_train": args.n_train, "n_test": args.n_test,
                "models": args.models,
                "pairwise_mean_f1": float(pw.mean()),
                "pointwise_mean_f1": float(pt.mean()),
                "proxy_mean_f1": float(np.array(picks["proxy"]).mean()),
                "oracle_mean_f1": float(np.array(picks["oracle"]).mean()),
                "pairwise_vs_pointwise_wins": wins,
                "pairwise_picks": {f"{k[0]}__{k[1]}": v
                                   for k, v in pairwise_picks.items()},
                "per_bucket_oracle": {f"{k[0]}__{k[1]}": v
                                      for k, v in per_bucket_oracle.items()},
                "lambdarank_feature_importances": {
                    n: int(v) for n, v in zip(feat_names, imps)
                },
            },
            "per_sample": [
                {"i": i, "pairwise": float(pw[k]), "pointwise": float(pt[k]),
                 "proxy": float(picks["proxy"][k]),
                 "oracle": float(picks["oracle"][k])}
                for k, i in enumerate(by_sample_tst.keys())
            ],
        }, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
