# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Stage 3: fit LambdaRank and build PPO state tables.

Reads features.pkl produced by featurize_and_label.py. Produces:
  - lambdarank_model.pkl            (LightGBM ranker)
  - lambdarank_scores.npz           (per-candidate scores for train and test)
  - lambdarank_result.json          (test F1 + picks)
  - paper_ppo_tables.npz            (S_trn, Rsem_trn, Rstr_trn, S_tst, Rsem_tst, Rstr_tst)
    (the PPO state+reward tables, usable directly by run_grpo.py / run_dpo.py /
     run_distillation.py)
"""
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import FINAL_ROOT, DATA_ROOT, setup_sys_path  # noqa: E402
setup_sys_path()
os.chdir(DATA_ROOT)

import numpy as np  # noqa: E402
import lightgbm as lgb  # noqa: E402

from scripts.rag_bandit_pipeline import embed_texts_titan  # noqa: E402
from scripts.reranker_pairwise import fit_lambdarank  # noqa: E402
from src.bedrock_llm import BedrockLLM  # noqa: E402

FEAT_DIR = FINAL_ROOT / "results" / "features"
OUT_DIR = FINAL_ROOT / "results"


K_CANDS = 40
USER_ITEM_EMB_DIM = 768
CAND_FEAT_DIM = 64


def build_ppo_tables(X_feat, y_f1, meta, by_sample, user_embs, item_embs,
                      W_REACH=30, W_NODE_SIM=8, W_NODE_DIV=6, seed=42):
    """Mirror scripts/paper_ppo_on_pool.py:build_state_and_reward_tables.

    Subsample K=40 candidates per sample with seed so state shape is fixed.
    """
    sample_ids = sorted(by_sample.keys())
    n_samples = len(sample_ids)
    state_dim = 2 * USER_ITEM_EMB_DIM + K_CANDS * CAND_FEAT_DIM
    states = np.zeros((n_samples, state_dim), dtype=np.float32)
    sem_reward = np.zeros((n_samples, K_CANDS), dtype=np.float32)
    str_reward = np.zeros((n_samples, K_CANDS), dtype=np.float32)

    rng = np.random.RandomState(seed)

    def pad_to_64(feat_vec):
        out = np.zeros(CAND_FEAT_DIM, dtype=np.float32)
        out[:len(feat_vec)] = feat_vec
        return out

    for row, sid in enumerate(sample_ids):
        group_idxs = list(by_sample[sid])
        # Subsample K=40 if group is larger; if smaller, pad by sampling with replacement.
        if len(group_idxs) >= K_CANDS:
            chosen = rng.choice(group_idxs, size=K_CANDS, replace=False)
        else:
            chosen = rng.choice(group_idxs, size=K_CANDS, replace=True)
        chosen = np.array(chosen, dtype=np.int64)

        cand_feats = np.stack([pad_to_64(X_feat[i]) for i in chosen], axis=0)
        # Find this sample's (uid, iid)
        m0 = meta[group_idxs[0]]
        # meta row is (sample_idx, model_id, style, j, pred, reference)
        # sid IS the sample_idx from meta's first element
        s_emb = np.concatenate([user_embs[sid], item_embs[sid]]).astype(np.float32)
        states[row] = np.concatenate([s_emb, cand_feats.reshape(-1)])
        sem_reward[row] = y_f1[chosen] * 100.0

        # Structural reward proxy
        feats = X_feat[chosen]
        reach = (feats[:, 15] > 0.3).astype(np.float32) if feats.shape[1] > 15 else np.ones(K_CANDS, dtype=np.float32)
        node_sim = 0.5 * (feats[:, 0] + feats[:, 1])
        node_div = feats[:, 17] if feats.shape[1] > 17 else np.zeros(K_CANDS, dtype=np.float32)
        raw = W_REACH * reach + W_NODE_SIM * node_sim + W_NODE_DIV * node_div
        noise = np.random.RandomState(int(sid)).randn(K_CANDS) * 0.01
        raw = raw + noise
        if raw.max() > raw.min():
            raw = (raw - raw.min()) / (raw.max() - raw.min()) * 100.0
        else:
            raw = np.zeros_like(raw)
        str_reward[row] = raw
    return states, sem_reward, str_reward, sample_ids


def main():
    print(f"Loading features from {FEAT_DIR / 'features.pkl'}...")
    feat = pickle.load(open(FEAT_DIR / "features.pkl", "rb"))
    X_trn, X_tst = feat["X_trn"], feat["X_tst"]
    y_trn, y_tst = feat["y_trn"], feat["y_tst"]
    meta_trn, meta_tst = feat["meta_trn"], feat["meta_tst"]
    grp_trn, grp_tst = feat["grp_trn"], feat["grp_tst"]
    by_sample_trn, by_sample_tst = feat["by_sample_trn"], feat["by_sample_tst"]
    trn_samples = feat["trn_samples"]
    tst_samples = feat["tst_samples"]
    print(f"  X_trn={X_trn.shape}  X_tst={X_tst.shape}")
    print(f"  groups: trn={len(grp_trn)}  tst={len(grp_tst)}")

    # ---- LambdaRank ----
    print("\nQuantile-binning labels within groups...")
    y_trn_lab = np.zeros_like(y_trn, dtype=np.int32)
    idx = 0
    for g in grp_trn:
        chunk = y_trn[idx:idx + g]
        if len(chunk) > 1:
            lo, hi = chunk.min(), chunk.max()
            if hi > lo:
                q = np.clip((chunk - lo) / (hi - lo) * 5, 0, 4).astype(int)
                y_trn_lab[idx:idx + g] = q
        idx += g

    print("Fitting LambdaRank...")
    ranker = fit_lambdarank(X_trn, y_trn_lab, grp_trn, seed=42, n_estimators=500)
    s_tst = ranker.predict(X_tst)

    picks = []
    idx = 0
    for i, sid in enumerate(sorted(by_sample_tst.keys())):
        rows = list(by_sample_tst[sid])
        best_local = int(np.argmax(s_tst[rows]))
        picks.append(rows[best_local])
    lr_f1 = np.array([y_tst[p] for p in picks])
    print(f"  LambdaRank test F1 = {lr_f1.mean():.4f}")

    # Save ranker + scores
    with open(OUT_DIR / "lambdarank_model.pkl", "wb") as f:
        pickle.dump(ranker, f)
    np.savez_compressed(OUT_DIR / "lambdarank_scores.npz",
                        s_trn=ranker.predict(X_trn),
                        s_tst=s_tst)
    lr_result = {
        "test_f1_mean": float(lr_f1.mean()),
        "test_f1_std": float(lr_f1.std()),
        "n_test": int(len(picks)),
        "picked_texts": [meta_tst[p][4] for p in picks],
        "references": [meta_tst[p][5] for p in picks],
    }
    with open(OUT_DIR / "lambdarank_result.json", "w") as f:
        json.dump(lr_result, f, indent=2)
    print(f"  saved lambdarank_model.pkl + lambdarank_result.json")

    # ---- PPO tables ----
    print("\nComputing user/item embeddings for state vector...")
    cache_path = FINAL_ROOT / "results" / "titan_cache.json"
    embed_llm = BedrockLLM(
        model_id="us.amazon.nova-lite-v1:0",
        max_tokens=300, temperature=0.0,
    )
    # Titan user/item text = user_summary / item_summary
    # Extract them from samples' synthesized prompts
    import re
    def extract_user_summary(p):
        m = re.search(r"User profile:\s*(.+?)(?:\n|$)", p, re.S)
        return m.group(1).strip() if m else ""
    def extract_biz_summary(p):
        m = re.search(r"Business profile:\s*(.+?)(?=User profile:)", p, re.S)
        return m.group(1).strip() if m else ""

    user_txts_trn = [extract_user_summary(s.prompt) for s in trn_samples]
    biz_txts_trn = [extract_biz_summary(s.prompt) for s in trn_samples]
    user_txts_tst = [extract_user_summary(s.prompt) for s in tst_samples]
    biz_txts_tst = [extract_biz_summary(s.prompt) for s in tst_samples]

    u_trn = embed_texts_titan(embed_llm, user_txts_trn, cache_path=cache_path, max_workers=8)
    b_trn = embed_texts_titan(embed_llm, biz_txts_trn, cache_path=cache_path, max_workers=8)
    u_tst = embed_texts_titan(embed_llm, user_txts_tst, cache_path=cache_path, max_workers=8)
    b_tst = embed_texts_titan(embed_llm, biz_txts_tst, cache_path=cache_path, max_workers=8)

    def to_768(arr):
        if arr.shape[1] == 768:
            return arr
        if arr.shape[1] > 768:
            return arr[:, :768].astype(np.float32)
        pad = np.zeros((arr.shape[0], 768 - arr.shape[1]), dtype=np.float32)
        return np.concatenate([arr, pad], axis=1).astype(np.float32)
    u_trn, b_trn = to_768(u_trn), to_768(b_trn)
    u_tst, b_tst = to_768(u_tst), to_768(b_tst)

    print("Building PPO state/reward tables (K=40 subsample per sample)...")
    S_trn, Rsem_trn, Rstr_trn, _ = build_ppo_tables(
        X_trn, y_trn, meta_trn, by_sample_trn, u_trn, b_trn,
    )
    S_tst, Rsem_tst, Rstr_tst, _ = build_ppo_tables(
        X_tst, y_tst, meta_tst, by_sample_tst, u_tst, b_tst,
    )
    print(f"  S_trn={S_trn.shape}  S_tst={S_tst.shape}")
    print(f"  train mean F1: {y_trn.mean():.4f}, test mean F1: {y_tst.mean():.4f}")

    tables_path = OUT_DIR / "paper_ppo_tables.npz"
    np.savez_compressed(
        tables_path,
        S_trn=S_trn, Rsem_trn=Rsem_trn, Rstr_trn=Rstr_trn,
        S_tst=S_tst, Rsem_tst=Rsem_tst, Rstr_tst=Rstr_tst,
    )
    print(f"  saved → {tables_path}")

    print("\n✅ LambdaRank + PPO tables ready")
    print(f"   LambdaRank test F1 = {lr_f1.mean():.4f} ± {lr_f1.std():.4f}")


if __name__ == "__main__":
    main()
