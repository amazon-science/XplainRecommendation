"""
Deterministically sample 5,000 training pairs from XRec's trn.pkl, FILTERED
to items that have same-business reviews available in our local cache
(`data/google_local/iid_reviews.json`).

Why filter?  Our explanation-generation pipeline grounds candidates in
same-business reviews (styles A, B, D, F). Training pairs without review
coverage would silently fall back to generic LLM output with no retrieval
signal, contaminating the training distribution.

The filter is deterministic: the Google Local Reviews cache was built from
the same McAuley Lab corpus XRec/G-Refer use, and covers 24,650 of the
94,663 training pairs (26%). We keep only those 24,650 and then sample
5,000 with `random_state=42`.

Test set is unaffected — 2,958 of 3,000 test pairs (99%) have reviews.
The 42 uncovered test pairs will drop naturally during pool generation
and appear as empty rows at eval time.

Run as:
    python3 pipeline/sample_training.py
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import FINAL_ROOT, DATA_ROOT, setup_sys_path  # noqa: E402
setup_sys_path()

import pandas as pd  # noqa: E402

XREC = DATA_ROOT / "XRec" / "data" / "google"
REVIEWS = DATA_ROOT / "data" / "google_local" / "iid_reviews.json"
OUT = FINAL_ROOT / "results" / "trn_5k.pkl"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading iid_reviews.json cache...")
    iid_revs = json.load(open(REVIEWS))
    iids_with_reviews = {int(k) for k in iid_revs.keys()}
    print(f"  {len(iids_with_reviews)} unique iids have reviews in the local cache")

    print("\nLoading XRec trn.pkl...")
    trn = pickle.load(open(XREC / "trn.pkl", "rb"))
    print(f"  full trn: {len(trn)} rows")

    mask = trn["iid"].astype(int).isin(iids_with_reviews)
    trn_covered = trn[mask].reset_index(drop=True)
    print(f"\nFiltering to review-covered pairs...")
    print(f"  trn pairs with local review coverage: {len(trn_covered)} / {len(trn)} "
          f"({100*len(trn_covered)/len(trn):.1f}%)")

    print("\nSampling 5,000 rows with random_state=42 (deterministic)...")
    trn_5k = trn_covered.sample(n=5000, random_state=42).reset_index(drop=True)
    print(f"  sampled: {len(trn_5k)} rows")
    print(f"  columns: {list(trn_5k.columns)}")

    with open(OUT, "wb") as f:
        pickle.dump(trn_5k, f)
    print(f"  saved → {OUT}")

    # Summary stats for the README
    n_users = trn_5k["uid"].nunique()
    n_items = trn_5k["iid"].nunique()
    print(f"\n  unique users: {n_users}   unique items: {n_items}")
    print(f"  avg explanation length (words): "
          f"{trn_5k['explanation'].str.split().str.len().mean():.1f}")

    # Verify every sampled row has reviews
    sampled_covered = trn_5k["iid"].astype(int).isin(iids_with_reviews).sum()
    print(f"  sampled pairs with reviews: {sampled_covered}/{len(trn_5k)} (should be 5000)")

    # Deterministic spot-check
    r = trn_5k.iloc[0]
    print(f"  first row: uid={r['uid']}  iid={r['iid']}  title={r['title'][:40]}...")


if __name__ == "__main__":
    main()
