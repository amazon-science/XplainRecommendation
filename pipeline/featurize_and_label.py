"""
Stage 2: featurize candidates + compute true BERTScore-F1 labels.

For the 5,000 train + 3,000 test pairs:
  1. Generate big-pool styles (C, D, E) via reranker_bigpool
  2. Generate style F (length-tuned) via reranker_styleF
  3. Featurize all candidates with the 30-dim feature vector
  4. Compute true F1 labels against ground-truth refs (train AND test)
  5. Save: X_trn, X_tst, y_trn, y_tst, meta_trn, meta_tst, grp_trn, grp_tst

This is the single most compute-heavy stage. On 8-core CPU:
  - big-pool + styleF generation: ~1 hour (if Bedrock rate-limit-friendly)
  - featurization (cross-encoder + Titan embedding): ~30 min
  - true-F1 labelling (BERTScore over all ~110k candidates): ~1 hour

Outputs saved to clean_xrec_data/results/features/
"""
import argparse
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

from scripts.rag_bandit_pipeline import Sample, bert_f1_batch, embed_texts_titan  # noqa: E402
from scripts.reranker_v2 import load_reviews  # noqa: E402
from scripts.reranker_final import build_or_load_pool_exp, knn_refs_for_samples  # noqa: E402
from scripts.reranker_styleF import featurize_with_F, gen_styleF  # noqa: E402
from scripts.reranker_bigpool import gen_new_styles, nearest_train_samples  # noqa: E402
from src.bedrock_llm import BedrockLLM  # noqa: E402
from scripts.reranker_pairwise import _groups_from_meta  # noqa: E402

RESULTS = FINAL_ROOT / "results"
POOL_DIR = RESULTS / "pool"
FEAT_DIR = RESULTS / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)


def build_sample_from_xrec_row(row, split):
    uid, iid = int(row["uid"]), int(row["iid"])
    title = str(row["title"]).strip()
    user_summary = str(row["user_summary"]).strip()
    biz_summary = str(row["item_summary"]).strip()
    reference = str(row["explanation"]).strip()
    prompt = (
        f"Given the business title, business profile, and user profile, "
        f"please explain why the user would enjoy this business within 50 words. "
        f"Business title: {title}. "
        f"Business profile: {biz_summary} "
        f"User profile: {user_summary}\n### \n### Explanation:"
    )
    return Sample(uid=uid, iid=iid, prompt=prompt,
                  reference=reference, grefer_output="", split=split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--k_neighbors", type=int, default=5)
    args = ap.parse_args()

    # Load the same samples the pool was built on
    import pandas as pd
    trn_5k = pickle.load(open(RESULTS / "trn_5k.pkl", "rb"))
    xrec_tst = pickle.load(open(DATA_ROOT / "XRec" / "data" / "google" / "tst.pkl", "rb"))

    trn_samples = [build_sample_from_xrec_row(r, "trn") for _, r in trn_5k.iterrows()]
    tst_samples = [build_sample_from_xrec_row(r, "tst") for _, r in xrec_tst.iterrows()]

    # Load cached A/B pool — filename key depends on model set (h3 vs h45)
    models = ["us.amazon.nova-lite-v1:0", "us.anthropic.claude-haiku-4-5-20251001-v1:0"]
    mkey = "_".join(m.split("/")[-1].split(":")[0].replace(".", "")[:14] for m in models)
    trn_pack_path = POOL_DIR / f"exp_pool_clean_xrec_trn_n5000_s6_b4_{mkey}_42.pkl"
    tst_pack_path = POOL_DIR / f"exp_pool_clean_xrec_tst_n3000_s6_b4_{mkey}_42.pkl"
    assert trn_pack_path.exists(), f"missing {trn_pack_path}"
    assert tst_pack_path.exists(), f"missing {tst_pack_path}"
    trn_pack = pickle.load(open(trn_pack_path, "rb"))
    tst_pack = pickle.load(open(tst_pack_path, "rb"))
    print(f"Loaded A/B pool: trn={len(trn_pack['preds'])}  tst={len(tst_pack['preds'])}")

    # Titan embedding infrastructure
    cache_path = RESULTS / "titan_cache.json"
    models = ["us.amazon.nova-lite-v1:0", "us.anthropic.claude-haiku-4-5-20251001-v1:0"]
    clients = {m: BedrockLLM(model_id=m, max_tokens=300, temperature=0.0) for m in models}
    embed_llm = clients[models[0]]

    # Ref centroid + exemplar — computed from 5k train refs
    print("\nBuilding reference centroid + exemplar...")
    trn_refs = [s.reference for s in trn_samples]
    trn_ref_embs = embed_texts_titan(embed_llm, trn_refs,
                                       cache_path=cache_path, max_workers=args.max_workers)
    ref_centroid = trn_ref_embs.mean(axis=0)
    ref_centroid /= (np.linalg.norm(ref_centroid) + 1e-9)
    trn_ref_embs_norm = trn_ref_embs / (
        np.linalg.norm(trn_ref_embs, axis=1, keepdims=True) + 1e-9)
    exemplar_ref = trn_refs[int(np.argmax(trn_ref_embs_norm @ ref_centroid))]
    print(f"  exemplar ref (first 80c): {exemplar_ref[:80]}")

    # iid_reviews needed for style D's review context
    iid_reviews = load_reviews()

    # nearest-train-neighbors for style C (few-shot)
    print("\nComputing nearest train neighbors for style C...")
    trn_neighbors_trn = nearest_train_samples(
        trn_samples, trn_samples, embed_llm, cache_path,
        args.max_workers, k=2,
    )
    trn_neighbors_tst = nearest_train_samples(
        tst_samples, trn_samples, embed_llm, cache_path,
        args.max_workers, k=2,
    )

    # --- Big-pool styles (C, D, E) — per-model-set suffix so h3 / h45 coexist ---
    model_tag = "h45" if any("haiku-4-5" in m for m in models) else "h3"
    bigpool_path = POOL_DIR / f"big_new_styles_clean_xrec_tr5000_te3000_t2_{model_tag}.pkl"
    if bigpool_path.exists():
        with open(bigpool_path, "rb") as f:
            bigpool = pickle.load(f)
        print(f"\nLoaded cached big-pool (C/D/E): trn={len(bigpool['trn'])} tst={len(bigpool['tst'])}")
    else:
        print("\nGenerating big-pool (C/D/E) — ~30 min...")
        big_trn = gen_new_styles(
            trn_samples, models, trn_pack, clients, args.max_workers,
            trn_neighbors_trn, exemplar_ref, iid_reviews, n_temp=2,
        )
        big_tst = gen_new_styles(
            tst_samples, models, tst_pack, clients, args.max_workers,
            trn_neighbors_tst, exemplar_ref, iid_reviews, n_temp=2,
        )
        bigpool = {"trn": big_trn, "tst": big_tst}
        with open(bigpool_path, "wb") as f:
            pickle.dump(bigpool, f)
        print(f"  Saved → {bigpool_path.name}")

    # --- Style F ---
    fpath = POOL_DIR / f"styleF_clean_xrec_tr5000_te3000_{model_tag}.pkl"
    if fpath.exists():
        with open(fpath, "rb") as f:
            fpack = pickle.load(f)
        print(f"Loaded cached style F: trn={len(fpack['trn'])} tst={len(fpack['tst'])}")
    else:
        print("\nGenerating style F (length-tuned, 25-33w) — ~15 min...")
        f_trn = gen_styleF(trn_samples, models, trn_pack, clients, args.max_workers)
        f_tst = gen_styleF(tst_samples, models, tst_pack, clients, args.max_workers)
        fpack = {"trn": f_trn, "tst": f_tst}
        with open(fpath, "wb") as f:
            pickle.dump(fpack, f)
        print(f"  Saved → {fpath.name}")

    trn_all = dict(trn_pack["preds"])
    trn_all.update(bigpool["trn"]); trn_all.update(fpack["trn"])
    tst_all = dict(tst_pack["preds"])
    tst_all.update(bigpool["tst"]); tst_all.update(fpack["tst"])
    print(f"\nTotal candidate count: trn={len(trn_all)}  tst={len(tst_all)}")

    # --- Cross-encoder + kNN refs ---
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    print("\nComputing kNN refs for each sample (train refs only)...")
    trn_knn = knn_refs_for_samples(trn_samples, trn_samples, embed_llm, cache_path,
                                    args.max_workers, args.k_neighbors)
    tst_knn = knn_refs_for_samples(tst_samples, trn_samples, embed_llm, cache_path,
                                    args.max_workers, args.k_neighbors)

    # --- Featurize ---
    print("\nFeaturizing train...")
    X_trn, meta_trn = featurize_with_F(
        trn_pack, trn_all, embed_llm, ref_centroid, exemplar_ref,
        trn_ref_embs_norm, cache_path, args.max_workers,
        trn_knn, cross_encoder, models,
    )
    print(f"  X_train: {X_trn.shape}")

    print("Featurizing test...")
    X_tst, meta_tst = featurize_with_F(
        tst_pack, tst_all, embed_llm, ref_centroid, exemplar_ref,
        trn_ref_embs_norm, cache_path, args.max_workers,
        tst_knn, cross_encoder, models,
    )
    print(f"  X_test: {X_tst.shape}")

    # --- True F1 labels ---
    print("\nComputing true BERTScore-F1 labels (~1 hour)...")
    y_trn = bert_f1_batch([m[4] for m in meta_trn], [m[5] for m in meta_trn])
    print(f"  train F1 mean: {y_trn.mean():.4f}")
    y_tst = bert_f1_batch([m[4] for m in meta_tst], [m[5] for m in meta_tst])
    print(f"  test  F1 mean: {y_tst.mean():.4f}")

    # --- Groups ---
    grp_trn, by_sample_trn = _groups_from_meta(meta_trn)
    grp_tst, by_sample_tst = _groups_from_meta(meta_tst)

    oracle_trn = np.array([max(y_trn[list(ks)]) for ks in by_sample_trn.values()])
    oracle_tst = np.array([max(y_tst[list(ks)]) for ks in by_sample_tst.values()])
    print(f"\n  train oracle: {oracle_trn.mean():.4f}")
    print(f"  test  oracle: {oracle_tst.mean():.4f}")

    # --- Save ---
    out = {
        "X_trn": X_trn, "X_tst": X_tst,
        "y_trn": y_trn, "y_tst": y_tst,
        "meta_trn": meta_trn, "meta_tst": meta_tst,
        "grp_trn": grp_trn, "grp_tst": grp_tst,
        "by_sample_trn": by_sample_trn, "by_sample_tst": by_sample_tst,
        "trn_samples": trn_samples, "tst_samples": tst_samples,
        "models": models,
        "ref_centroid": ref_centroid,
        "exemplar_ref": exemplar_ref,
        "oracle_trn_mean": float(oracle_trn.mean()),
        "oracle_tst_mean": float(oracle_tst.mean()),
    }
    out_path = FEAT_DIR / "features.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"\n✅ Saved features + labels → {out_path}")


if __name__ == "__main__":
    main()
