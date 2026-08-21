"""
Build the candidate pool for 5,000 train + 3,000 test pairs from XRec.

For each (uid, iid):
  - Synthesize a G-Refer-format prompt from (title, user_summary, item_summary)
  - Pull same-business reviews from iid_reviews.json
  - Run the existing 6-style × 2-LLM generation pipeline
  - Cache candidates to disk

Reuses cached candidates from prior runs where (uid, iid) overlap exists
(1,400 of our 3,000 test pairs are already cached from the previous
approach_ppo_0.4864 pool — 800 from its train pool + 600 from its test pool).

Output: clean_xrec_data/results/pool/trn_5k.pkl and tst_3k.pkl
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import FINAL_ROOT, DATA_ROOT, setup_sys_path  # noqa: E402
setup_sys_path()
os.chdir(DATA_ROOT)  # many lib helpers use repo-relative cache paths

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.rag_bandit_pipeline import Sample, embed_texts_titan  # noqa: E402
from scripts.reranker_v2 import load_reviews  # noqa: E402
from scripts.reranker_final import build_or_load_pool_exp  # noqa: E402
from src.bedrock_llm import BedrockLLM  # noqa: E402

RESULTS = FINAL_ROOT / "results"
POOL_DIR = RESULTS / "pool"
POOL_DIR.mkdir(parents=True, exist_ok=True)


def build_sample_from_xrec_row(row) -> Sample:
    """Synthesize a Sample with a G-Refer-format composite prompt.

    The existing prompt builders (build_prompt_A, build_prompt_B) extract
    user/biz profile substrings from Sample.prompt using regex — so the
    composite string must match the format used throughout scripts/.
    """
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
        f"User profile: {user_summary}\n"
        f"### \n"
        f"### Explanation:"
    )
    return Sample(
        uid=uid, iid=iid, prompt=prompt,
        reference=reference, grefer_output="", split="trn",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=5000)
    ap.add_argument("--n_test", type=int, default=3000)
    ap.add_argument("--n_single", type=int, default=6)
    ap.add_argument("--n_synth", type=int, default=4)
    ap.add_argument("--max_workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ---- Load splits ----
    trn_5k = pickle.load(open(RESULTS / "trn_5k.pkl", "rb"))
    xrec_tst = pickle.load(open(DATA_ROOT / "XRec" / "data" / "google" / "tst.pkl", "rb"))
    print(f"Train: {len(trn_5k)}  Test: {len(xrec_tst)}")

    if args.n_train < len(trn_5k):
        print(f"  [debug head-cut] using first {args.n_train} of {len(trn_5k)} trn_5k rows")
        trn_5k = trn_5k.head(args.n_train).reset_index(drop=True)
    if args.n_test < len(xrec_tst):
        print(f"  [debug head-cut] using first {args.n_test} of {len(xrec_tst)} tst rows")
        xrec_tst = xrec_tst.head(args.n_test).reset_index(drop=True)

    # ---- Build Sample objects ----
    print("Building Sample objects with synthesized prompts...")
    trn_samples = [build_sample_from_xrec_row(r) for _, r in trn_5k.iterrows()]
    tst_samples = [build_sample_from_xrec_row(r) for _, r in xrec_tst.iterrows()]
    for s in trn_samples:
        s.split = "trn"
    for s in tst_samples:
        s.split = "tst"
    print(f"  {len(trn_samples)} train Sample objects, {len(tst_samples)} test Sample objects")

    # Sanity: prompt structure parses with existing regex
    from scripts.reranker_v2 import extract_user_profile, extract_biz_profile
    s0 = trn_samples[0]
    up = extract_user_profile(s0.prompt)
    bp = extract_biz_profile(s0.prompt)
    assert up, f"extract_user_profile failed on synthesized prompt:\n{s0.prompt[:500]}"
    assert bp, f"extract_biz_profile failed on synthesized prompt:\n{s0.prompt[:500]}"
    print(f"  sanity check: first sample parses correctly")
    print(f"    user_profile[:80]: {up[:80]}")
    print(f"    biz_profile[:80]:  {bp[:80]}")

    # ---- Reviews ----
    print("\nLoading iid_reviews.json...")
    iid_reviews = load_reviews()
    print(f"  {len(iid_reviews)} unique iids with reviews")
    train_iids_with_reviews = sum(1 for s in trn_samples if s.iid in iid_reviews)
    test_iids_with_reviews = sum(1 for s in tst_samples if s.iid in iid_reviews)
    print(f"  train pairs with reviews:  {train_iids_with_reviews}/{len(trn_samples)}")
    print(f"  test  pairs with reviews:  {test_iids_with_reviews}/{len(tst_samples)}")

    # ---- Bedrock clients ----
    models = ["us.amazon.nova-lite-v1:0",
              "us.anthropic.claude-haiku-4-5-20251001-v1:0"]
    clients = {m: BedrockLLM(model_id=m, max_tokens=300, temperature=0.0)
               for m in models}
    embed_llm = clients[models[0]]

    # ---- Reference centroid (for ranking reviews in retrieval) ----
    print("\nComputing reference centroid (from our 5k train refs only)...")
    cache_path = RESULTS / "titan_cache.json"
    trn_refs = [s.reference for s in trn_samples]
    trn_ref_embs = embed_texts_titan(embed_llm, trn_refs,
                                     cache_path=cache_path,
                                     max_workers=args.max_workers)
    ref_centroid = trn_ref_embs.mean(axis=0)
    ref_centroid /= (np.linalg.norm(ref_centroid) + 1e-9)

    # ---- Generate pool ----
    print("\n=== Generating training pool ===")
    trn_pack = build_or_load_pool_exp(
        "clean_xrec_trn", trn_samples, models, args.n_single, args.n_synth,
        args.max_workers, iid_reviews, ref_centroid, cache_path,
        embed_llm, clients, args.seed, POOL_DIR,
    )
    print(f"  saved trn_pack with {len(trn_pack['preds'])} cached candidates")

    print("\n=== Generating test pool ===")
    tst_pack = build_or_load_pool_exp(
        "clean_xrec_tst", tst_samples, models, args.n_single, args.n_synth,
        args.max_workers, iid_reviews, ref_centroid, cache_path,
        embed_llm, clients, args.seed, POOL_DIR,
    )
    print(f"  saved tst_pack with {len(tst_pack['preds'])} cached candidates")

    print("\n✅ Pool build complete")
    print(f"   Train pool: {POOL_DIR}/exp_pool_clean_xrec_trn_*.pkl")
    print(f"   Test  pool: {POOL_DIR}/exp_pool_clean_xrec_tst_*.pkl")


if __name__ == "__main__":
    main()
