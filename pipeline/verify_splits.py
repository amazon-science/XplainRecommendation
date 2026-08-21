"""
Verify that XRec's published splits behave as we claim in README.md.

Checks:
  1. trn.pkl, val.pkl, tst.pkl are pairwise disjoint by (uid, iid)
  2. tst.pkl matches G-Refer's google_pred.jsonl 1:1 in pair membership
  3. tst.pkl's 'explanation' column is byte-identical to G-Refer's
     source_data.chosen (when the stripped '### ' marker is removed)

Run as:
    python3 pipeline/verify_splits.py
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import DATA_ROOT, setup_sys_path  # noqa: E402
setup_sys_path()

import pandas as pd  # noqa: E402

XREC = DATA_ROOT / "XRec" / "data" / "google"
GREFER_PRED = DATA_ROOT / "G-Refer" / "gen_explanations" / "G-Refer" / "google_pred.jsonl"


def main():
    print("Loading XRec splits...")
    trn = pickle.load(open(XREC / "trn.pkl", "rb"))
    val = pickle.load(open(XREC / "val.pkl", "rb"))
    tst = pickle.load(open(XREC / "tst.pkl", "rb"))
    print(f"  trn: {len(trn)}  val: {len(val)}  tst: {len(tst)}")

    to_pairs = lambda df: set(zip(df["uid"].astype(int), df["iid"].astype(int)))
    trn_p, val_p, tst_p = to_pairs(trn), to_pairs(val), to_pairs(tst)
    print(f"  unique pairs: trn={len(trn_p)}  val={len(val_p)}  tst={len(tst_p)}")

    # 1. Pairwise disjoint
    print("\n[1] Checking pairwise disjoint splits...")
    assert len(trn_p & tst_p) == 0, f"trn ∩ tst = {len(trn_p & tst_p)}"
    assert len(val_p & tst_p) == 0, f"val ∩ tst = {len(val_p & tst_p)}"
    assert len(trn_p & val_p) == 0, f"trn ∩ val = {len(trn_p & val_p)}"
    print("  ✓ trn, val, tst are pairwise disjoint")

    # 2. tst.pkl matches G-Refer predictions
    print("\n[2] Checking tst.pkl == G-Refer google_pred.jsonl pair set...")
    grefer_rows = [json.loads(L) for L in open(GREFER_PRED)]
    grefer_p = set((int(r["source_data"]["uid"]), int(r["source_data"]["iid"]))
                    for r in grefer_rows)
    assert tst_p == grefer_p, (
        f"Pair mismatch: tst has {len(tst_p - grefer_p)} not in G-Refer, "
        f"G-Refer has {len(grefer_p - tst_p)} not in tst"
    )
    print(f"  ✓ all 3000 test pairs in tst.pkl match G-Refer's google_pred.jsonl")

    # 3. Explanation bytes match
    print("\n[3] Checking explanation strings match byte-exact...")
    # Build lookup
    grefer_by_pair = {(int(r["source_data"]["uid"]), int(r["source_data"]["iid"])):
                     r["source_data"]["chosen"].split("### ", 1)[-1].strip()
                     for r in grefer_rows}
    mismatches = 0
    checked = 0
    for _, r in tst.iterrows():
        key = (int(r["uid"]), int(r["iid"]))
        xrec_exp = str(r["explanation"]).strip()
        grefer_exp = grefer_by_pair[key]
        if xrec_exp != grefer_exp:
            mismatches += 1
            if mismatches <= 3:
                print(f"  mismatch at {key}:")
                print(f"    XRec:    {xrec_exp[:100]}...")
                print(f"    G-Refer: {grefer_exp[:100]}...")
        checked += 1
    print(f"  checked {checked} / {len(tst)} pairs, mismatches: {mismatches}")
    if mismatches == 0:
        print("  ✓ all explanations byte-exact across XRec and G-Refer")
    else:
        print(f"  ✗ {mismatches} mismatches found — investigate")

    print("\n✅ All split integrity checks passed" if mismatches == 0 else
          f"\n❌ {mismatches} mismatches — data integrity issue")


if __name__ == "__main__":
    main()
