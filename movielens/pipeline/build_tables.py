# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
MovieLens data-prep pipeline (single stage).

Loads the cached 600 train / 300 test pools at
    results/rag_bandit/pool_cache_ml/poolA_tr600_te300_*.pkl
    results/rag_bandit/pool_cache_ml/poolB_tr600_te300_*.pkl

and produces:
    movielens/results/paper_ppo_tables.npz   (S_*, Rsem_*, Rstr_*)
    movielens/results/lambdarank_model.pkl
    movielens/results/lambdarank_result.json (picked_texts + references)
    movielens/results/features.pkl           (for post-hoc text lookup)

K_CANDS = 18 (MovieLens pool has exactly 18 candidates per sample).

The cached pools were built with Claude Haiku 4.5 + Nova Lite. Titan
embeddings are cached per-pipeline at
    movielens/results/titan_cache.json
(isolated from the 700 MB global cache to avoid OOM).
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from _paths import FINAL_ROOT, DATA_ROOT, setup_sys_path  # noqa: E402
setup_sys_path()
os.chdir(DATA_ROOT)

import numpy as np  # noqa: E402
import lightgbm as lgb  # noqa: E402

# MovieLens helpers live in scripts/ — imported via setup_sys_path
from scripts.reranker_movielens import (  # noqa: E402
    load_movielens_samples, MLSample, _OCC,
    knn_refs_for_samples as ml_knn_refs,
    featurize as ml_featurize,
)
from scripts.rag_bandit_pipeline import bert_f1_batch, embed_texts_titan  # noqa: E402
from scripts.reranker_pairwise import fit_lambdarank, _groups_from_meta  # noqa: E402
from src.bedrock_llm import BedrockLLM  # noqa: E402

ML_ROOT = FINAL_ROOT / "movielens"
RESULTS = ML_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

K_CANDS = 18
CAND_FEAT_DIM = 64
USER_ITEM_EMB_DIM = 768

# Feature indices for structural-reward proxy (same as paper_ppo_on_pool_movielens)
IDX_COS_USER = 0
IDX_COS_BIZ = 1
IDX_KNN_MEAN = 17
IDX_KNN_STD = 19

W_REACH = 30.0
W_NODE_SIM = 8.0
W_NODE_DIV = 6.0


def build_groups(meta):
    from collections import defaultdict
    groups = defaultdict(list)
    for k, m in enumerate(meta):
        groups[m[0]].append(k)
    return {i: np.array(groups[i]) for i in sorted(groups.keys())}


def subsample_k_per_group(groups_dict, k, rng):
    picked = {}
    for sid, idxs in groups_dict.items():
        if len(idxs) >= k:
            picked[sid] = rng.choice(idxs, size=k, replace=False)
        else:
            pad = rng.choice(idxs, size=k - len(idxs), replace=True)
            picked[sid] = np.concatenate([idxs, pad])
    return picked


def pad_features(X_row, target=CAND_FEAT_DIM):
    if X_row.shape[0] >= target:
        return X_row[:target]
    return np.concatenate([X_row, np.zeros(target - X_row.shape[0])])


def build_state_and_reward_tables(X_feat, y_f1, meta, groups_k,
                                    user_embs, item_embs, sample_ids_ordered):
    n_samples = len(sample_ids_ordered)
    state_dim = 2 * USER_ITEM_EMB_DIM + K_CANDS * CAND_FEAT_DIM
    states = np.zeros((n_samples, state_dim), dtype=np.float32)
    sem_reward = np.zeros((n_samples, K_CANDS), dtype=np.float32)
    str_reward = np.zeros((n_samples, K_CANDS), dtype=np.float32)

    for row, sid in enumerate(sample_ids_ordered):
        idxs = groups_k[sid]
        cand_feats = np.stack([pad_features(X_feat[i]) for i in idxs], axis=0)
        states[row] = np.concatenate(
            [user_embs[sid], item_embs[sid], cand_feats.reshape(-1)]
        )
        sem_reward[row] = y_f1[idxs] * 100.0

        feats = X_feat[idxs]
        reach = (feats[:, IDX_KNN_MEAN] > 0.3).astype(np.float32) \
            if feats.shape[1] > IDX_KNN_MEAN else np.ones(K_CANDS, dtype=np.float32)
        node_sim = 0.5 * (feats[:, IDX_COS_USER] + feats[:, IDX_COS_BIZ])
        node_div = feats[:, IDX_KNN_STD] if feats.shape[1] > IDX_KNN_STD \
            else np.zeros(K_CANDS, dtype=np.float32)
        raw = W_REACH * reach + W_NODE_SIM * node_sim + W_NODE_DIV * node_div
        raw = raw + np.random.RandomState(int(sid)).randn(K_CANDS) * 0.01
        if raw.max() > raw.min():
            raw = (raw - raw.min()) / (raw.max() - raw.min()) * 100.0
        else:
            raw = np.zeros_like(raw)
        str_reward[row] = raw

    return states, sem_reward, str_reward


def to_768(arr):
    if arr.shape[1] == 768:
        return arr.astype(np.float32)
    if arr.shape[1] > 768:
        return arr[:, :768].astype(np.float32)
    pad = np.zeros((arr.shape[0], 768 - arr.shape[1]), dtype=np.float32)
    return np.concatenate([arr, pad], axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=600)
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--k_neighbors", type=int, default=5)
    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool_cache_dir", type=str,
                    default="results/rag_bandit/pool_cache_ml")
    ap.add_argument("--models", type=str, nargs="+", default=[
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "us.amazon.nova-lite-v1:0",
    ])
    args = ap.parse_args()

    # Isolated Titan cache to avoid 700 MB OOM seen on the big shared cache
    cache_path = RESULTS / "titan_cache.json"

    print(f"n_train={args.n_train}  n_test={args.n_test}  K={K_CANDS}")

    # ---- Samples ----
    all_samples = load_movielens_samples(seed=args.seed)
    trn_eval = all_samples[: args.n_train]
    tst_eval = all_samples[args.n_train: args.n_train + args.n_test]
    print(f"  train={len(trn_eval)}  test={len(tst_eval)}")

    # ---- Cached pools ----
    mtag = "_".join(m.split(":")[0].split(".")[-1][:10] for m in args.models)
    trn_cache = Path(args.pool_cache_dir) / \
        f"poolA_tr{args.n_train}_te{args.n_test}_{mtag}.pkl"
    tst_cache = Path(args.pool_cache_dir) / \
        f"poolB_tr{args.n_train}_te{args.n_test}_{mtag}.pkl"
    if not trn_cache.exists() or not tst_cache.exists():
        raise FileNotFoundError(f"Missing cache: {trn_cache} / {tst_cache}")
    trn_pack = pickle.load(open(trn_cache, "rb"))
    tst_pack = pickle.load(open(tst_cache, "rb"))
    assert len(trn_pack["samples"]) == len(trn_eval)
    assert len(tst_pack["samples"]) == len(tst_eval)
    print(f"  loaded pools: trn preds={len(trn_pack['preds'])}  "
          f"tst preds={len(tst_pack['preds'])}")

    # ---- Embeddings + centroid/exemplar ----
    clients = {m: BedrockLLM(model_id=m, max_tokens=300, temperature=0.0)
               for m in args.models}
    embed_llm = clients[args.models[0]]

    print("\nEmbedding prompts + refs...")
    prompt_texts = [s.prompt for s in all_samples]
    ref_texts = [s.reference for s in all_samples]
    prompt_embs = embed_texts_titan(embed_llm, prompt_texts,
                                     cache_path=cache_path,
                                     max_workers=args.max_workers)
    ref_embs = embed_texts_titan(embed_llm, ref_texts,
                                  cache_path=cache_path,
                                  max_workers=args.max_workers)
    _norm = lambda M: M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    prompt_embs_n = _norm(prompt_embs)
    ref_embs_n = _norm(ref_embs)
    trn_prompt_embs_n = prompt_embs_n[: args.n_train]
    trn_ref_embs_n = ref_embs_n[: args.n_train]
    tst_prompt_embs_n = prompt_embs_n[args.n_train: args.n_train + args.n_test]
    centroid = trn_ref_embs_n.mean(axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-9)
    exemplar_ref = trn_eval[int(np.argmax(trn_ref_embs_n @ centroid))].reference

    # ---- Cross-encoder + kNN ----
    print("\nkNN + cross-encoder...")
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",
                                   max_length=512)
    trn_knn = ml_knn_refs(trn_eval, trn_eval, trn_prompt_embs_n,
                           trn_prompt_embs_n, args.k_neighbors)
    tst_knn = ml_knn_refs(tst_eval, trn_eval, tst_prompt_embs_n,
                           trn_prompt_embs_n, args.k_neighbors,
                           exclude_self=False)

    # ---- Featurize ----
    print("\nFeaturize train...")
    X_trn, meta_trn = ml_featurize(
        trn_pack, embed_llm, centroid, exemplar_ref, trn_ref_embs_n,
        cache_path, args.max_workers, trn_knn, cross_encoder, args.models,
    )
    print(f"  X_trn={X_trn.shape}")
    print("Featurize test...")
    X_tst, meta_tst = ml_featurize(
        tst_pack, embed_llm, centroid, exemplar_ref, trn_ref_embs_n,
        cache_path, args.max_workers, tst_knn, cross_encoder, args.models,
    )
    print(f"  X_tst={X_tst.shape}")

    # ---- True F1 labels ----
    print("\nTrue-F1 labels (BERTScore rescaled)...")
    y_trn = bert_f1_batch([m[4] for m in meta_trn], [m[5] for m in meta_trn])
    y_tst = bert_f1_batch([m[4] for m in meta_tst], [m[5] for m in meta_tst])
    print(f"  train F1 mean: {y_trn.mean():.4f}   test F1 mean: {y_tst.mean():.4f}")

    # ---- User/item 768-dim embeddings ----
    print("\nUser/item 768-dim embeddings...")
    user_texts_trn = [f"{s.age_group} {s.gender}, {_OCC.get(s.occupation, '')}"
                       for s in trn_eval]
    biz_texts_trn = [f"{s.title} {' '.join(s.genres)}" for s in trn_eval]
    user_texts_tst = [f"{s.age_group} {s.gender}, {_OCC.get(s.occupation, '')}"
                       for s in tst_eval]
    biz_texts_tst = [f"{s.title} {' '.join(s.genres)}" for s in tst_eval]
    u_trn = to_768(embed_texts_titan(embed_llm, user_texts_trn,
                                       cache_path=cache_path,
                                       max_workers=args.max_workers))
    b_trn = to_768(embed_texts_titan(embed_llm, biz_texts_trn,
                                       cache_path=cache_path,
                                       max_workers=args.max_workers))
    u_tst = to_768(embed_texts_titan(embed_llm, user_texts_tst,
                                       cache_path=cache_path,
                                       max_workers=args.max_workers))
    b_tst = to_768(embed_texts_titan(embed_llm, biz_texts_tst,
                                       cache_path=cache_path,
                                       max_workers=args.max_workers))

    # ---- Build groups + K-subsampled tables ----
    grp_trn = build_groups(meta_trn)
    grp_tst = build_groups(meta_tst)
    rng_sub = np.random.RandomState(args.seed)
    grp_trn_k = subsample_k_per_group(grp_trn, K_CANDS, rng_sub)
    grp_tst_k = subsample_k_per_group(grp_tst, K_CANDS, rng_sub)

    trn_sample_ids = sorted(grp_trn_k.keys())
    tst_sample_ids = sorted(grp_tst_k.keys())

    S_trn, Rsem_trn, Rstr_trn = build_state_and_reward_tables(
        X_trn, y_trn, meta_trn, grp_trn_k, u_trn, b_trn, trn_sample_ids,
    )
    S_tst, Rsem_tst, Rstr_tst = build_state_and_reward_tables(
        X_tst, y_tst, meta_tst, grp_tst_k, u_tst, b_tst, tst_sample_ids,
    )
    print(f"\n  S_trn={S_trn.shape}  S_tst={S_tst.shape}")
    print(f"  train oracle: {(Rsem_trn.max(axis=1)/100).mean():.4f}")
    print(f"  test  oracle: {(Rsem_tst.max(axis=1)/100).mean():.4f}")

    tables_path = RESULTS / "paper_ppo_tables.npz"
    np.savez_compressed(
        tables_path,
        S_trn=S_trn, Rsem_trn=Rsem_trn, Rstr_trn=Rstr_trn,
        S_tst=S_tst, Rsem_tst=Rsem_tst, Rstr_tst=Rstr_tst,
    )
    print(f"  saved → {tables_path}")

    # ---- Fit LambdaRank on full (non-subsampled) groups ----
    print("\nFit LambdaRank (full groups)...")
    grp_trn_list = [len(grp_trn[i]) for i in sorted(grp_trn.keys())]
    grp_tst_list = [len(grp_tst[i]) for i in sorted(grp_tst.keys())]

    y_trn_lab = np.zeros_like(y_trn, dtype=np.int32)
    idx = 0
    for g in grp_trn_list:
        chunk = y_trn[idx: idx + g]
        if len(chunk) > 1:
            lo, hi = chunk.min(), chunk.max()
            if hi > lo:
                q = np.clip((chunk - lo) / (hi - lo) * 5, 0, 4).astype(int)
                y_trn_lab[idx: idx + g] = q
        idx += g

    ranker = fit_lambdarank(X_trn, y_trn_lab, grp_trn_list,
                             seed=args.seed, n_estimators=500)
    s_tst = ranker.predict(X_tst)

    picks_idx = []
    for sid in sorted(grp_tst.keys()):
        rows = list(grp_tst[sid])
        best = int(np.argmax(s_tst[rows]))
        picks_idx.append(rows[best])
    picks_idx = np.array(picks_idx)
    lr_f1 = y_tst[picks_idx]
    print(f"  LambdaRank test F1 = {lr_f1.mean():.4f} ± {lr_f1.std():.4f}")

    # Save ranker + result with picked_texts (needed for BART/USR scoring later)
    with open(RESULTS / "lambdarank_model.pkl", "wb") as f:
        pickle.dump(ranker, f)
    lr_out = {
        "test_f1_mean": float(lr_f1.mean()),
        "test_f1_std": float(lr_f1.std()),
        "n_test": int(len(picks_idx)),
        "picked_texts": [meta_tst[p][4] for p in picks_idx],
        "references": [meta_tst[p][5] for p in picks_idx],
    }
    with open(RESULTS / "lambdarank_result.json", "w") as f:
        json.dump(lr_out, f, indent=2)
    print(f"  saved LambdaRank artefacts")

    # Save features.pkl so score_all_metrics_ml.py can do picks → text lookup
    feat_pkl = RESULTS / "features.pkl"
    out = {
        "X_trn": X_trn, "X_tst": X_tst,
        "y_trn": y_trn, "y_tst": y_tst,
        "meta_trn": meta_trn, "meta_tst": meta_tst,
        "grp_trn": grp_trn_list, "grp_tst": grp_tst_list,
        "by_sample_trn": {i: list(grp_trn[i]) for i in grp_trn},
        "by_sample_tst": {i: list(grp_tst[i]) for i in grp_tst},
        "trn_samples": trn_eval, "tst_samples": tst_eval,
        "models": args.models,
        "oracle_trn_mean": float((Rsem_trn.max(axis=1) / 100).mean()),
        "oracle_tst_mean": float((Rsem_tst.max(axis=1) / 100).mean()),
    }
    with open(feat_pkl, "wb") as f:
        pickle.dump(out, f)
    print(f"  saved features.pkl for pick-to-text mapping")

    print("\n✅ MovieLens tables ready")
    print(f"   LambdaRank test F1 = {lr_f1.mean():.4f} ± {lr_f1.std():.4f}")
    print(f"   Pool oracle ceiling = {(Rsem_tst.max(axis=1)/100).mean():.4f}")


if __name__ == "__main__":
    main()
