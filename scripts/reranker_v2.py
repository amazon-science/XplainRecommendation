# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Reranker v2: larger candidate pool + richer features.

Candidate pool per sample:
  - 3 models × 4 single-review paraphrases (style A) = 12
  - 3 models × 1 multi-review synthesis (style B)    =  3
  Total 15 per sample.

Models: Nova Lite, Haiku 3, Claude 3.5 Haiku (all template-friendly).

Features per candidate (beyond v1):
  - prompt_type_A, prompt_type_B (one-hot)
  - model one-hot (3 dims)
  - generic_positive_count (count of house-style words)
  - has_number (0/1)
  - has_named_entity_hint (0/1, heuristic: contains capitalized word not
    at sentence start that looks like a proper noun)
  - sentence_count (should be 1)
  - length chars / words
  - cos_user / cos_biz / cos_ref_centroid / cos_exemplar (from embeddings)
  - top5_nearest_trn_ref
  - rev_sim_user / rev_sim_ref (source review scoring)
  - rating of source reviews (mean if style B)
  - avg length of source reviews
"""

import argparse, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import DATA_ROOT
from src.bedrock_llm import BedrockLLM
from scripts.rag_bandit_pipeline import (
    load_grefer_samples, invoke_bedrock,
    bert_f1_batch, embed_texts_titan, _short_context, SYSTEM_PROMPT,
)

REVIEW_INDEX = DATA_ROOT / "data" / "google_local" / "iid_reviews.json"

GENERIC_POSITIVE = [
    "delicious", "friendly", "great", "convenient", "atmosphere",
    "service", "prices", "options", "experience", "fresh", "quality",
    "tasty", "cozy", "welcoming", "clean", "comfortable",
    "selection", "variety", "authentic", "affordable", "reasonable",
]

GREFER_SYSTEM_B = (
    "You are writing explanations in the G-Refer house style. "
    "Format rules — follow exactly:\n"
    "- EXACTLY ONE sentence\n"
    "- 25-35 words\n"
    "- Starts with 'The user would enjoy the business because of its' OR "
    "'The user would enjoy the business for its' OR "
    "'The user would enjoy the business because it'\n"
    "- Do NOT name specific dishes unless widely mentioned in the reviews\n"
    "- Do NOT mention the business name\n"
    "- List 3-4 reasons connected with commas and 'and'\n"
    "- Use generic-positive vocabulary: delicious, friendly, great, "
    "convenient, atmosphere, service, prices, options, experience\n"
)


def load_reviews():
    with open(REVIEW_INDEX) as f:
        return {int(k): v for k, v in json.load(f).items()}


def extract_user_profile(p):
    m = re.search(r"User profile:\s*([^\n#]+)", p)
    return m.group(1).strip() if m else ""


def extract_biz_profile(p):
    m = re.search(r"Business profile:\s*([^\n]+?)(?=\s*User profile:|$)", p)
    return m.group(1).strip() if m else ""


def filter_reviews(revs, min_words=10, max_words=60):
    return [r for r in revs if min_words <= len(r["text"].split()) <= max_words]


def build_prompt_A(s, review):
    """Single-review paraphrase — same as v3."""
    ctx = _short_context(s.prompt)
    return (
        SYSTEM_PROMPT + "\n\n" +
        "Use the real Google Maps review below as grounding. Paraphrase its "
        "key points into the house style, matching the user's taste. "
        "Start with 'The user would enjoy the business because'. "
        "ONE sentence, 20-30 words.\n\n" +
        f"{ctx}\n\n" +
        f"Real review ({review['rating']}★): \"{review['text']}\"\n\n" +
        "Explanation:"
    )


def build_prompt_B(s, reviews):
    """Multi-review synthesis — same as v4."""
    ctx = _short_context(s.prompt)
    rev_block = "\n".join(
        f"- ({r['rating']}★) {r['text']}"
        for r in reviews
    )
    return (
        GREFER_SYSTEM_B + "\n\n" +
        "Context:\n" +
        f"{ctx}\n\n" +
        f"Real user reviews of this business (select the recurring themes):\n{rev_block}\n\n" +
        "Task: synthesize the top 3-4 reasons that THIS user (given their "
        "profile) would enjoy this business, drawing from the reviews. "
        "Output ONE sentence in the exact house style above.\n\n" +
        "Explanation:"
    )


def generate_pool(samples, models, n_single, n_synth_reviews, max_workers,
                  iid_reviews, ref_centroid, cache_path, embed_llm, clients):
    """Generate candidates: (samples × models × n_single) A + (samples × models × 1) B."""
    user_profiles = [extract_user_profile(s.prompt) for s in samples]
    user_embs = embed_texts_titan(embed_llm, user_profiles, cache_path=cache_path,
                                   max_workers=max_workers)

    all_texts, text_idx = [], {}
    sample_cands = {}
    for i, s in enumerate(samples):
        revs = filter_reviews(iid_reviews.get(s.iid, {}).get("reviews", []))[:80]
        sample_cands[i] = revs
        for r in revs:
            if r["text"] not in text_idx:
                text_idx[r["text"]] = len(all_texts)
                all_texts.append(r["text"])
    print(f"  Embedding {len(all_texts)} reviews...")
    rev_embs = embed_texts_titan(embed_llm, all_texts, cache_path=cache_path,
                                  max_workers=max_workers)
    rev_embs_norm = rev_embs / (
        np.linalg.norm(rev_embs, axis=1, keepdims=True) + 1e-9)

    # Rank reviews per sample
    ranked = {}
    for i, s in enumerate(samples):
        cands = sample_cands[i]
        if not cands: ranked[i] = []; continue
        idxs = [text_idx[r["text"]] for r in cands]
        ce = rev_embs_norm[idxs]
        u = user_embs[i]; u = u / (np.linalg.norm(u) + 1e-9)
        sims_user = ce @ u
        sims_ref = ce @ ref_centroid
        ratings = np.array([(r.get("rating") or 3) for r in cands])
        rating_bonus = (ratings - 3) * 0.02
        combined = 0.5 * sims_user + 0.4 * sims_ref + rating_bonus
        order = np.argsort(-combined)
        n_take = max(n_single, n_synth_reviews)
        top = [(cands[int(k)], float(sims_user[int(k)]), float(sims_ref[int(k)]))
               for k in order[:n_take]]
        ranked[i] = top

    # Build jobs
    jobs = []
    for model_id in models:
        for i in range(len(samples)):
            top = ranked[i]
            if not top: continue
            # Style A: single-review paraphrase × n_single
            for j, (r, _, _) in enumerate(top[:n_single]):
                jobs.append(("A", model_id, i, j, r, top[:n_single]))
            # Style B: one synthesis using top-n_synth_reviews
            jobs.append(("B", model_id, i, -1, None, top[:n_synth_reviews]))

    print(f"  Generating {len(jobs)} candidates across "
          f"{len(models)} models × {len(samples)} samples...")

    def gen(job):
        style, model_id, i, j, r, rev_ctx = job
        s = samples[i]
        if style == "A":
            msg = build_prompt_A(s, r)
        else:
            msg = build_prompt_B(s, [x[0] for x in rev_ctx])
        out = invoke_bedrock(clients[model_id], msg, temperature=0.0)
        return style, model_id, i, j, out or ""

    preds = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(gen, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="  gen"):
            style, m, i, j, out = fut.result()
            preds[(style, m, i, j)] = out

    return {
        "samples": samples,
        "user_profiles": user_profiles,
        "biz_profiles": [extract_biz_profile(s.prompt) for s in samples],
        "ranked": ranked,
        "preds": preds,
        "models": list(models),
        "n_single": n_single,
        "n_synth": n_synth_reviews,
    }


def _count_numbers(s):
    return len(re.findall(r"\b\d+\b", s))


def _count_generic_positive(s):
    low = s.lower()
    return sum(1 for w in GENERIC_POSITIVE if w in low)


def _sentence_count(s):
    # Split on . ! ? followed by space or end
    parts = re.split(r"[.!?]+\s*", s.strip())
    return len([p for p in parts if p.strip()])


def _named_entity_hint(s):
    # Crude: capitalized tokens after first word that aren't common
    toks = s.split()
    if len(toks) <= 1:
        return 0.0
    for tok in toks[1:]:
        if re.match(r"^[A-Z][a-z]+$", tok) and tok.lower() not in {
            "the", "user", "business", "it", "its", "and", "or",
        }:
            return 1.0
    return 0.0


def featurize(pack, embed_llm, ref_centroid, exemplar_ref,
              trn_ref_embs_norm, cache_path, max_workers):
    samples = pack["samples"]
    preds = pack["preds"]
    ranked = pack["ranked"]
    models = pack["models"]
    user_profiles = pack["user_profiles"]
    biz_profiles = pack["biz_profiles"]
    n_single = pack["n_single"]

    # Collect unique texts for embedding
    pred_texts = [v for v in preds.values() if v.strip()]
    to_embed = list(set(pred_texts + biz_profiles + user_profiles + [exemplar_ref]))
    print(f"  Embedding {len(to_embed)} unique texts for features...")
    embs = embed_texts_titan(embed_llm, to_embed, cache_path=cache_path,
                              max_workers=max_workers)
    emb_idx = {t: k for k, t in enumerate(to_embed)}
    embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)

    def embn(t):
        return embs_n[emb_idx[t]] if t.strip() in emb_idx else None

    exemplar_vec = embs_n[emb_idx[exemplar_ref]]

    X, meta = [], []
    for i, s in enumerate(samples):
        u_vec = embn(user_profiles[i])
        b_vec = embn(biz_profiles[i])
        top = ranked[i]
        if not top: continue

        for m in models:
            # Style A
            for j, (rev, rev_sim_u, rev_sim_r) in enumerate(top[:n_single]):
                pred = preds.get(("A", m, i, j), "")
                if not pred.strip(): continue
                p_vec = embn(pred)
                rating = float(rev.get("rating") or 3)
                src_len = len(rev["text"].split())
                feat = _pred_features(pred, p_vec, u_vec, b_vec, ref_centroid,
                                       exemplar_vec, trn_ref_embs_norm,
                                       rev_sim_u, rev_sim_r, rating, src_len,
                                       style="A", models=models, model_id=m)
                X.append(feat)
                meta.append((i, m, "A", j, pred, s.reference))

            # Style B (single synthesis)
            pred = preds.get(("B", m, i, -1), "")
            if not pred.strip(): continue
            p_vec = embn(pred)
            # Use top-n_synth avg rating and mean sim as source features
            used = top[:pack["n_synth"]]
            rev_sim_u = float(np.mean([t[1] for t in used])) if used else 0.0
            rev_sim_r = float(np.mean([t[2] for t in used])) if used else 0.0
            ratings = [t[0].get("rating") or 3 for t in used]
            src_len = float(np.mean([len(t[0]["text"].split()) for t in used])) if used else 0.0
            feat = _pred_features(pred, p_vec, u_vec, b_vec, ref_centroid,
                                   exemplar_vec, trn_ref_embs_norm,
                                   rev_sim_u, rev_sim_r, float(np.mean(ratings)),
                                   src_len, style="B", models=models, model_id=m)
            X.append(feat)
            meta.append((i, m, "B", -1, pred, s.reference))
    return np.array(X), meta


def _pred_features(pred, p_vec, u_vec, b_vec, ref_centroid, exemplar_vec,
                    trn_ref_embs_norm, rev_sim_user, rev_sim_ref, rating,
                    src_len, style, models, model_id):
    cos_user = float(p_vec @ u_vec) if u_vec is not None else 0.0
    cos_biz = float(p_vec @ b_vec) if b_vec is not None else 0.0
    cos_ref = float(p_vec @ ref_centroid)
    cos_exe = float(p_vec @ exemplar_vec)
    sims_nearest = trn_ref_embs_norm @ p_vec
    top5 = float(np.sort(sims_nearest)[-5:].mean())
    n_words = len(pred.split())
    n_chars = len(pred)
    starts_ok = 1.0 if pred.lstrip().startswith(
        "The user would enjoy the business") else 0.0
    n_nums = float(_count_numbers(pred))
    n_pos = float(_count_generic_positive(pred))
    n_sents = float(_sentence_count(pred))
    has_ne = float(_named_entity_hint(pred))
    style_A = 1.0 if style == "A" else 0.0
    style_B = 1.0 if style == "B" else 0.0
    model_oh = [1.0 if model_id == mm else 0.0 for mm in models]
    return [
        cos_user, cos_biz, cos_ref, cos_exe, top5,
        float(n_words), float(n_chars), starts_ok,
        n_nums, n_pos, n_sents, has_ne,
        rating, src_len, rev_sim_user, rev_sim_ref,
        style_A, style_B,
    ] + model_oh


def feature_names(models):
    return [
        "cos_user", "cos_biz", "cos_ref_centroid", "cos_exemplar", "top5_nearest_trn_ref",
        "n_words", "n_chars", "starts_ok",
        "n_nums", "n_generic_positive", "sent_count", "has_named_entity",
        "rating", "src_len", "rev_sim_user", "rev_sim_ref",
        "style_A", "style_B",
    ] + [f"model={m}" for m in models]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--n_single", type=int, default=4,
                    help="Single-review candidates per model")
    ap.add_argument("--n_synth", type=int, default=8,
                    help="Reviews injected into style-B synthesis")
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

    print(f"\n=== STAGE 1: Generate train pool (n={args.n_train}) ===")
    trn_pack = generate_pool(
        trn_eval, args.models, args.n_single, args.n_synth,
        args.max_workers, iid_reviews, ref_centroid, cache_path,
        embed_llm, clients,
    )
    print(f"\n=== STAGE 2: Generate test pool (n={args.n_test}) ===")
    tst_pack = generate_pool(
        tst_eval, args.models, args.n_single, args.n_synth,
        args.max_workers, iid_reviews, ref_centroid, cache_path,
        embed_llm, clients,
    )

    print(f"\n=== STAGE 3: Featurize ===")
    print("Train:")
    X_trn, meta_trn = featurize(trn_pack, embed_llm, ref_centroid, exemplar_ref,
                                trn_ref_embs_norm, cache_path, args.max_workers)
    print("Test:")
    X_tst, meta_tst = featurize(tst_pack, embed_llm, ref_centroid, exemplar_ref,
                                trn_ref_embs_norm, cache_path, args.max_workers)
    print(f"  X_train: {X_trn.shape}  X_test: {X_tst.shape}")

    print(f"\n=== STAGE 4: Compute true-F1 labels ===")
    y_trn = bert_f1_batch([m[4] for m in meta_trn], [m[5] for m in meta_trn])
    y_tst = bert_f1_batch([m[4] for m in meta_tst], [m[5] for m in meta_tst])
    print(f"  train label mean={y_trn.mean():.4f}  test label mean={y_tst.mean():.4f}")

    # Oracle ceiling (per-sample best)
    by_sample_trn = {}
    for k, m in enumerate(meta_trn):
        by_sample_trn.setdefault(m[0], []).append(k)
    oracle_trn = np.array([max(y_trn[ks]) for ks in by_sample_trn.values()])
    print(f"  train oracle ceiling: {oracle_trn.mean():.4f}")

    by_sample_tst = {}
    for k, m in enumerate(meta_tst):
        by_sample_tst.setdefault(m[0], []).append(k)
    oracle_tst = np.array([max(y_tst[ks]) for ks in by_sample_tst.values()])
    print(f"  test  oracle ceiling: {oracle_tst.mean():.4f}")

    print(f"\n=== STAGE 5: Fit reranker ===")
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_scores = []
    for f, (itr, ite) in enumerate(kf.split(X_trn)):
        gbr = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                         learning_rate=0.05,
                                         random_state=args.seed)
        gbr.fit(X_trn[itr], y_trn[itr])
        r = np.corrcoef(gbr.predict(X_trn[ite]), y_trn[ite])[0, 1]
        cv_scores.append(r)
    print(f"  5-fold CV Pearson r: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    gbr = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                     learning_rate=0.05,
                                     random_state=args.seed)
    gbr.fit(X_trn, y_trn)
    y_tst_pred = gbr.predict(X_tst)

    feat_names = feature_names(args.models)
    print("\n  Feature importances:")
    order = np.argsort(-gbr.feature_importances_)
    for k in order[:12]:
        print(f"    {feat_names[k]:<34} {gbr.feature_importances_[k]:.4f}")

    print(f"\n=== STAGE 6: Evaluate on test ===")
    picks_learned, picks_oracle = [], []
    proxy_texts = [
        f"{tst_pack['biz_profiles'][m[0]]} {tst_pack['user_profiles'][m[0]]} {exemplar_ref}".strip()
        for m in meta_tst
    ]
    print("  Computing proxy_f1...")
    proxy_f1 = bert_f1_batch([m[4] for m in meta_tst], proxy_texts)

    picks_learned, picks_proxy, picks_oracle = [], [], []
    from collections import Counter
    pick_breakdown = Counter()
    for i, ks in by_sample_tst.items():
        kl = ks[int(np.argmax(y_tst_pred[ks]))]
        kp = ks[int(np.argmax(proxy_f1[ks]))]
        ko = ks[int(np.argmax(y_tst[ks]))]
        picks_learned.append(y_tst[kl])
        picks_proxy.append(y_tst[kp])
        picks_oracle.append(y_tst[ko])
        pick_breakdown[(meta_tst[kl][1], meta_tst[kl][2])] += 1

    pl = np.array(picks_learned); pp = np.array(picks_proxy); po = np.array(picks_oracle)
    print(f"\n  === Test F1 (n={len(pl)} samples) ===")
    print(f"  LEARNED  : {pl.mean():.4f} ± {pl.std():.4f}  "
          f"%>=0.50 {100*(pl>=0.5).mean():.1f}%  "
          f"%>=0.55 {100*(pl>=0.55).mean():.1f}%  "
          f"%>=0.60 {100*(pl>=0.6).mean():.1f}%")
    print(f"  PROXY    : {pp.mean():.4f} ± {pp.std():.4f}  "
          f"%>=0.50 {100*(pp>=0.5).mean():.1f}%  "
          f"%>=0.55 {100*(pp>=0.55).mean():.1f}%  "
          f"%>=0.60 {100*(pp>=0.6).mean():.1f}%")
    print(f"  ORACLE   : {po.mean():.4f} ± {po.std():.4f}  "
          f"%>=0.50 {100*(po>=0.5).mean():.1f}%  "
          f"%>=0.55 {100*(po>=0.55).mean():.1f}%  "
          f"%>=0.60 {100*(po>=0.6).mean():.1f}%")
    print(f"\n  Learned picks by (model, style): {dict(pick_breakdown)}")

    # Per-bucket oracle (helpful diagnostic)
    per_bucket_oracle = {}
    for bucket in set((m[1], m[2]) for m in meta_tst):
        ks = [k for k, m in enumerate(meta_tst) if (m[1], m[2]) == bucket]
        if ks:
            per_bucket_oracle[bucket] = float(np.mean(y_tst[ks]))
    print(f"\n  Mean F1 per (model, style) bucket:")
    for bucket, v in sorted(per_bucket_oracle.items(), key=lambda x: -x[1]):
        print(f"    {bucket[0]} [{bucket[1]}]: {v:.4f}")

    out_dir = Path("results/rag_bandit")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"reranker_v2_{args.n_train}tr_{args.n_test}te.json"
    with open(out_file, "w") as f:
        json.dump({
            "summary": {
                "n_train": args.n_train, "n_test": args.n_test,
                "n_single": args.n_single, "n_synth": args.n_synth,
                "models": args.models,
                "learned_mean_f1": float(pl.mean()),
                "proxy_mean_f1": float(pp.mean()),
                "oracle_mean_f1": float(po.mean()),
                "cv_pearson_r": float(np.mean(cv_scores)),
                "per_bucket_oracle": {f"{k[0]}__{k[1]}": v
                                      for k, v in per_bucket_oracle.items()},
                "learned_pick_breakdown": {f"{k[0]}__{k[1]}": v
                                           for k, v in pick_breakdown.items()},
                "feature_importances": {
                    n: float(v) for n, v in zip(feat_names, gbr.feature_importances_)
                },
            },
        }, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
