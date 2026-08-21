"""
Compute G-Refer + XRec baselines on the exact same 3,000 XRec test pairs
our method evaluates on.

Outputs: clean_xrec_data/results/baselines.json
"""
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import FINAL_ROOT, DATA_ROOT, setup_sys_path  # noqa: E402
setup_sys_path()
sys.path.insert(0, str(DATA_ROOT / "G-Refer" / "evaluation"))
os.chdir(DATA_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.rag_bandit_pipeline import bert_f1_batch  # noqa: E402

OUT = FINAL_ROOT / "results" / "baselines.json"


def usr_metric(sequences):
    unique = []
    for seq in sequences:
        if any(len(u) == len(seq) and all(a == b for a, b in zip(u, seq))
               for u in unique):
            continue
        unique.append(seq)
    return len(unique) / max(len(sequences), 1), len(unique)


def bartscore_list(preds, refs, batch_size=4, device="cpu"):
    from bart_score import BARTScorer
    from tqdm import tqdm
    s = BARTScorer(device=device, checkpoint="facebook/bart-large-cnn")
    out = []
    for i in tqdm(range(0, len(preds), batch_size), desc="BARTScore"):
        out.extend(s.score(preds[i:i + batch_size],
                            refs[i:i + batch_size], batch_size=batch_size))
    return np.array(out)


def strip_marker(text):
    if not isinstance(text, str):
        return ""
    if "### " in text:
        return text.split("### ", 1)[-1].strip()
    return text.strip()


def main():
    # Load the reference 3,000 test pairs from XRec tst.pkl
    xrec_tst = pickle.load(open(DATA_ROOT / "XRec" / "data" / "google" / "tst.pkl", "rb"))
    refs = [strip_marker(str(e)) for e in xrec_tst["explanation"]]
    print(f"Test set: {len(xrec_tst)} pairs from XRec tst.pkl")

    # --- XRec predictions ---
    print("\n=== XRec ===")
    xrec_pred_list = pickle.load(open(DATA_ROOT / "XRec" / "data" / "google" / "tst_pred.pkl", "rb"))
    xrec_preds = [strip_marker(str(p)) for p in xrec_pred_list]
    assert len(xrec_preds) == len(refs)

    f1_xrec = bert_f1_batch(xrec_preds, refs)
    usr_xrec, n_uniq_xrec = usr_metric([p.split() for p in xrec_preds])
    print(f"  BERTScore F1 = {f1_xrec.mean():.4f}  USR = {usr_xrec:.4f} ({n_uniq_xrec}/{len(refs)} unique)")
    bart_xrec = bartscore_list(xrec_preds, refs, device="cpu")
    print(f"  BARTScore = {bart_xrec.mean():.4f}")

    # --- G-Refer predictions ---
    print("\n=== G-Refer ===")
    grefer_rows = {}
    for L in open(DATA_ROOT / "G-Refer" / "gen_explanations" / "G-Refer" / "google_pred.jsonl"):
        d = json.loads(L)
        key = (int(d["source_data"]["uid"]), int(d["source_data"]["iid"]))
        grefer_rows[key] = strip_marker(d["output_str"])

    grefer_preds = []
    matched = 0
    for _, r in xrec_tst.iterrows():
        key = (int(r["uid"]), int(r["iid"]))
        if key in grefer_rows:
            grefer_preds.append(grefer_rows[key])
            matched += 1
        else:
            grefer_preds.append("")
    print(f"  matched {matched}/{len(refs)} XRec test pairs in G-Refer output")

    # only score the matched ones
    paired = [(p, r) for p, r in zip(grefer_preds, refs) if p]
    if paired:
        p_list, r_list = zip(*paired)
        f1_g = bert_f1_batch(list(p_list), list(r_list))
        usr_g, n_uniq_g = usr_metric([p.split() for p in p_list])
        print(f"  BERTScore F1 = {f1_g.mean():.4f} (n={len(paired)})")
        print(f"  USR = {usr_g:.4f} ({n_uniq_g}/{len(paired)} unique)")
        bart_g = bartscore_list(list(p_list), list(r_list), device="cpu")
        print(f"  BARTScore = {bart_g.mean():.4f}")
    else:
        f1_g = np.array([]); usr_g = 0.0; n_uniq_g = 0; bart_g = np.array([])

    # --- Save ---
    out = {
        "n_test": len(refs),
        "xrec": {
            "bertscore_f1_mean": float(f1_xrec.mean()),
            "bertscore_f1_std": float(f1_xrec.std()),
            "bartscore_mean": float(bart_xrec.mean()),
            "bartscore_std": float(bart_xrec.std()),
            "usr": usr_xrec,
            "n_unique": n_uniq_xrec,
            "n_samples": len(refs),
        },
        "grefer": {
            "bertscore_f1_mean": float(f1_g.mean()) if len(f1_g) else None,
            "bertscore_f1_std": float(f1_g.std()) if len(f1_g) else None,
            "bartscore_mean": float(bart_g.mean()) if len(bart_g) else None,
            "bartscore_std": float(bart_g.std()) if len(bart_g) else None,
            "usr": usr_g,
            "n_unique": n_uniq_g,
            "n_samples": len(paired),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUT}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
