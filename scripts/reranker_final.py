"""Reranker final: expanded pool + full-train + cross-encoder + ensemble.

Combines every no-fine-tuning lever:
  1. Expanded pool: 8 single-review candidates per model (vs 4 before),
     MMR-diversified so candidates 5-8 aren't just near-duplicates of 1-4.
  2. Full 2400 train samples for LambdaRank supervision.
  3. Ensemble: LambdaRank + XGBoostRanker + pointwise GBR, averaged by rank.
  4. Cross-encoder feature using sentence-transformers cross-encoder:
     score(pred, user_profile + biz_profile) and score(pred, kNN_refs).

Caches pool + features so reruns are cheap.

Usage:
  python3 scripts/reranker_final.py --n_train 2400 --n_test 200 \
      --n_single 8 --n_synth 8 --max_workers 8
"""

import argparse, json, os, pickle, sys, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.reranker_v2 import (
    load_reviews, extract_user_profile, extract_biz_profile, filter_reviews,
    build_prompt_A, build_prompt_B,
)
from scripts.reranker_pairwise import fit_lambdarank, _groups_from_meta
from src.bedrock_llm import BedrockLLM
from scripts.rag_bandit_pipeline import (
    load_grefer_samples, invoke_bedrock,
    bert_f1_batch, embed_texts_titan,
)


GENERIC_POSITIVE = [
    "delicious", "friendly", "great", "convenient", "atmosphere",
    "service", "prices", "options", "experience", "fresh", "quality",
    "tasty", "cozy", "welcoming", "clean", "comfortable",
    "selection", "variety", "authentic", "affordable", "reasonable",
]


def mmr_select(review_embs_norm, combined_scores, k, lam=0.5):
    """MMR: pick k reviews that are both high-scoring and mutually diverse.

    combined_scores: (N,) array of base scores
    Returns: list of indices into the input arrays.
    """
    n = review_embs_norm.shape[0]
    if n == 0: return []
    k = min(k, n)
    selected = []
    remaining = list(range(n))
    # First pick: highest score
    first = int(np.argmax(combined_scores))
    selected.append(first)
    remaining.remove(first)
    while len(selected) < k and remaining:
        # For each remaining, diversity = 1 - max similarity to any selected
        sel_embs = review_embs_norm[selected]
        rem_embs = review_embs_norm[remaining]
        sims_to_sel = rem_embs @ sel_embs.T   # (n_rem, n_sel)
        max_sim = sims_to_sel.max(axis=1)
        # MMR score
        mmr_scores = (lam * combined_scores[remaining]
                       - (1 - lam) * max_sim)
        best = int(np.argmax(mmr_scores))
        selected.append(remaining[best])
        remaining.pop(best)
    return selected


def invoke_with_retry(llm, msg, temperature, max_tokens=300, retries=3):
    delay = 0.5
    for i in range(retries):
        out = invoke_bedrock(llm, msg, temperature=temperature, max_tokens=max_tokens)
        if out: return out
        time.sleep(delay); delay = min(delay * 2, 4.0)
    return ""


def generate_pool_expanded(samples, models, n_single, n_synth, max_workers,
                            iid_reviews, ref_centroid, cache_path, embed_llm,
                            clients):
    """MMR-diversified top-n_single reviews per sample + 1 synthesis per model."""
    user_profiles = [extract_user_profile(s.prompt) for s in samples]
    user_embs = embed_texts_titan(embed_llm, user_profiles,
                                    cache_path=cache_path, max_workers=max_workers)

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
    rev_embs_norm = rev_embs / (np.linalg.norm(rev_embs, axis=1, keepdims=True) + 1e-9)

    # Rank reviews with MMR diversification
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
        # MMR: pick n_single diverse high-scoring; also keep top n_synth for style-B context
        n_take = max(n_single, n_synth)
        picks = mmr_select(ce, combined, n_take, lam=0.7)
        ranked[i] = [(cands[p], float(sims_user[p]), float(sims_ref[p]))
                     for p in picks]

    # Build jobs
    jobs = []
    for model_id in models:
        for i in range(len(samples)):
            top = ranked[i]
            if not top: continue
            for j, (r, _, _) in enumerate(top[:n_single]):
                jobs.append(("A", model_id, i, j, r, None))
            jobs.append(("B", model_id, i, -1, None, top[:n_synth]))

    print(f"  Generating {len(jobs)} candidates...")

    def gen(job):
        style, model_id, i, j, r, rev_ctx = job
        s = samples[i]
        if style == "A":
            msg = build_prompt_A(s, r)
        else:
            msg = build_prompt_B(s, [x[0] for x in rev_ctx])
        out = invoke_with_retry(clients[model_id], msg, temperature=0.0)
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
        "n_synth": n_synth,
    }


def build_or_load_pool_exp(split, samples, models, n_single, n_synth,
                             max_workers, iid_reviews, ref_centroid,
                             cache_path, embed_llm, clients, seed, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    mkey = "_".join(m.split("/")[-1].split(":")[0].replace(".", "")[:14]
                    for m in models)
    key = cache_dir / f"exp_pool_{split}_n{len(samples)}_s{n_single}_b{n_synth}_{mkey}_{seed}.pkl"
    if key.exists():
        print(f"  Loading cached: {key.name}")
        with open(key, "rb") as f:
            return pickle.load(f)
    pack = generate_pool_expanded(
        samples, models, n_single, n_synth, max_workers,
        iid_reviews, ref_centroid, cache_path, embed_llm, clients,
    )
    with open(key, "wb") as f: pickle.dump(pack, f)
    return pack


# ---- features ----

def _count_numbers(s): return len(re.findall(r"\b\d+\b", s))
def _count_pos(s):
    low = s.lower(); return sum(1 for w in GENERIC_POSITIVE if w in low)
def _sent_count(s):
    parts = re.split(r"[.!?]+\s*", s.strip())
    return len([p for p in parts if p.strip()])
def _ne_hint(s):
    toks = s.split()
    if len(toks) <= 1: return 0.0
    for tok in toks[1:]:
        if re.match(r"^[A-Z][a-z]+$", tok) and tok.lower() not in {
            "the", "user", "business", "it", "its", "and", "or"}:
            return 1.0
    return 0.0


def _knn_bert_f1(preds_refs_pairs):
    preds = [p for p, _ in preds_refs_pairs]
    refs = [r for _, r in preds_refs_pairs]
    return bert_f1_batch(preds, refs)


def featurize_final(pack, embed_llm, ref_centroid, exemplar_ref,
                     trn_ref_embs_norm, cache_path, max_workers,
                     knn_refs_per_sample, cross_encoder):
    """Full feature set: base + kNN BERTScore + cross-encoder."""
    samples = pack["samples"]
    preds = pack["preds"]
    ranked = pack["ranked"]
    models = pack["models"]
    user_profiles = pack["user_profiles"]
    biz_profiles = pack["biz_profiles"]
    n_single = pack["n_single"]

    pred_texts = [v for v in preds.values() if v.strip()]
    to_embed = list(set(pred_texts + biz_profiles + user_profiles + [exemplar_ref]))
    print(f"  Embedding {len(to_embed)} unique texts...")
    embs = embed_texts_titan(embed_llm, to_embed, cache_path=cache_path,
                              max_workers=max_workers)
    emb_idx = {t: k for k, t in enumerate(to_embed)}
    embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)

    def embn(t):
        return embs_n[emb_idx[t]] if t.strip() in emb_idx else None

    exemplar_vec = embs_n[emb_idx[exemplar_ref]]

    # Candidate list in deterministic order
    candidate_keys = []
    for i in range(len(samples)):
        top = ranked[i]
        if not top: continue
        for m in models:
            for j in range(min(n_single, len(top))):
                k = ("A", m, i, j)
                if preds.get(k, "").strip():
                    candidate_keys.append(k)
            k = ("B", m, i, -1)
            if preds.get(k, "").strip():
                candidate_keys.append(k)

    # ---- kNN BERTScore features ----
    print(f"  kNN-BERTScore: preparing pairs...")
    bert_pairs, pair_back = [], {}
    for (style, m, i, j) in candidate_keys:
        pred = preds[(style, m, i, j)]
        for rk, ref_text in enumerate(knn_refs_per_sample[i]):
            pair_back[(style, m, i, j, rk)] = len(bert_pairs)
            bert_pairs.append((pred, ref_text))

    print(f"  Scoring {len(bert_pairs)} kNN pairs...")
    knn_f1 = bert_f1_batch([p for p, _ in bert_pairs],
                             [r for _, r in bert_pairs])
    k_per = len(knn_refs_per_sample[0]) if knn_refs_per_sample else 5
    knn_stats = {}
    for key in candidate_keys:
        (style, m, i, j) = key
        scores = [knn_f1[pair_back[(style, m, i, j, rk)]]
                  for rk in range(k_per) if (style, m, i, j, rk) in pair_back]
        if scores:
            arr = np.array(scores)
            knn_stats[key] = (float(arr.mean()), float(arr.max()), float(arr.std()))
        else:
            knn_stats[key] = (0.0, 0.0, 0.0)

    # ---- cross-encoder features ----
    # CE scores candidate vs (a) "user_profile + biz_profile" and (b) "knn ref texts"
    print(f"  Cross-encoder scoring...")
    ce_pairs_up, ce_pairs_knn = [], []
    up_targets = [f"{user_profiles[i]} {biz_profiles[i]}" for i in range(len(samples))]
    ce_up_back = {}
    ce_knn_back = {}
    for key in candidate_keys:
        (style, m, i, j) = key
        pred = preds[key]
        ce_up_back[key] = len(ce_pairs_up)
        ce_pairs_up.append((pred, up_targets[i]))
        # One CE score vs concatenation of the kNN refs
        knn_concat = " ".join(knn_refs_per_sample[i][:3])  # top 3 refs
        ce_knn_back[key] = len(ce_pairs_knn)
        ce_pairs_knn.append((pred, knn_concat))

    ce_up_scores = cross_encoder.predict(ce_pairs_up, show_progress_bar=True,
                                          batch_size=64)
    ce_knn_scores = cross_encoder.predict(ce_pairs_knn, show_progress_bar=True,
                                            batch_size=64)

    # ---- assemble X, meta ----
    X, meta = [], []
    for i, s in enumerate(samples):
        u_vec = embn(user_profiles[i])
        b_vec = embn(biz_profiles[i])
        top = ranked[i]
        if not top: continue

        for m in models:
            for j in range(min(n_single, len(top))):
                rev, rev_sim_u, rev_sim_r = top[j]
                key = ("A", m, i, j)
                pred = preds.get(key, "")
                if not pred.strip(): continue
                p_vec = embn(pred)
                feat = _pred_feats_final(
                    pred, p_vec, u_vec, b_vec, ref_centroid, exemplar_vec,
                    trn_ref_embs_norm, rev_sim_u, rev_sim_r,
                    float(rev.get("rating") or 3),
                    float(len(rev["text"].split())), style="A",
                    models=models, model_id=m,
                    knn=knn_stats[key],
                    ce_up=float(ce_up_scores[ce_up_back[key]]),
                    ce_knn=float(ce_knn_scores[ce_knn_back[key]]),
                    cand_rank=j,
                )
                X.append(feat)
                meta.append((i, m, "A", j, pred, s.reference))

            key = ("B", m, i, -1)
            pred = preds.get(key, "")
            if pred.strip():
                p_vec = embn(pred)
                used = top[:pack["n_synth"]]
                rev_sim_u = float(np.mean([t[1] for t in used])) if used else 0.0
                rev_sim_r = float(np.mean([t[2] for t in used])) if used else 0.0
                ratings = [t[0].get("rating") or 3 for t in used]
                src_len = float(np.mean([len(t[0]["text"].split()) for t in used])) if used else 0.0
                feat = _pred_feats_final(
                    pred, p_vec, u_vec, b_vec, ref_centroid, exemplar_vec,
                    trn_ref_embs_norm, rev_sim_u, rev_sim_r,
                    float(np.mean(ratings)), src_len, style="B",
                    models=models, model_id=m, knn=knn_stats[key],
                    ce_up=float(ce_up_scores[ce_up_back[key]]),
                    ce_knn=float(ce_knn_scores[ce_knn_back[key]]),
                    cand_rank=0,
                )
                X.append(feat)
                meta.append((i, m, "B", -1, pred, s.reference))
    return np.array(X), meta


def _pred_feats_final(pred, p_vec, u_vec, b_vec, ref_centroid, exemplar_vec,
                       trn_ref_embs_norm, rev_sim_u, rev_sim_r, rating,
                       src_len, style, models, model_id, knn, ce_up, ce_knn,
                       cand_rank):
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
    style_A = 1.0 if style == "A" else 0.0
    style_B = 1.0 if style == "B" else 0.0
    model_oh = [1.0 if model_id == mm else 0.0 for mm in models]
    return [
        cos_user, cos_biz, cos_ref, cos_exe, top5,
        float(n_words), float(n_chars), starts_ok,
        float(_count_numbers(pred)), float(_count_pos(pred)),
        float(_sent_count(pred)), float(_ne_hint(pred)),
        rating, src_len, rev_sim_u, rev_sim_r,
        style_A, style_B, float(cand_rank),
        knn[0], knn[1], knn[2],
        ce_up, ce_knn,
    ] + model_oh


def feat_names(models):
    return [
        "cos_user", "cos_biz", "cos_ref_centroid", "cos_exemplar",
        "top5_nearest_trn_ref", "n_words", "n_chars", "starts_ok",
        "n_nums", "n_generic_positive", "sent_count", "has_named_entity",
        "rating", "src_len", "rev_sim_user", "rev_sim_ref",
        "style_A", "style_B", "cand_rank",
        "knn_bert_mean", "knn_bert_max", "knn_bert_std",
        "ce_user_biz", "ce_knn_refs",
    ] + [f"model={m}" for m in models]


def knn_refs_for_samples(samples, train_samples, embed_llm, cache_path,
                          max_workers, k, exclude_self=True):
    def combined_key(s):
        u = extract_user_profile(s.prompt); b = extract_biz_profile(s.prompt)
        return f"User: {u} | Business: {b}"
    qry_texts = [combined_key(s) for s in samples]
    trn_texts = [combined_key(s) for s in train_samples]
    trn_ref_texts = [t.reference.replace("### ", "").strip() for t in train_samples]
    trn_keys = [(t.uid, t.iid) for t in train_samples]
    qry_keys = [(s.uid, s.iid) for s in samples]
    q = embed_texts_titan(embed_llm, qry_texts, cache_path=cache_path,
                          max_workers=max_workers)
    t = embed_texts_titan(embed_llm, trn_texts, cache_path=cache_path,
                          max_workers=max_workers)
    q_n = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    t_n = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    sims = q_n @ t_n.T
    out = []
    for i in range(len(samples)):
        order = np.argsort(-sims[i])
        picks = []
        for j in order:
            if exclude_self and trn_keys[int(j)] == qry_keys[i]: continue
            picks.append(trn_ref_texts[int(j)])
            if len(picks) >= k: break
        out.append(picks)
    return out


def ensemble_rank_scores(X_trn, y_trn, grp_trn, X_tst, seed):
    """Fit 3 rankers, return per-sample rank-averaged score for each test row."""
    from sklearn.ensemble import GradientBoostingRegressor
    import xgboost as xgb

    # 1. LambdaRank (LightGBM)
    lgbm = fit_lambdarank(X_trn, y_trn, grp_trn, seed=seed, n_estimators=500)
    s_lgbm = lgbm.predict(X_tst)

    # 2. XGBoost ranker
    # Quantize labels within each train group
    y_rel = np.zeros(len(y_trn), dtype=int)
    idx = 0
    for g in grp_trn:
        block = y_trn[idx:idx + g]
        order = np.argsort(-block)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))
        bin_size = max(1, len(block) // 5)
        lvl = np.maximum(0, np.minimum(4, 4 - (ranks // bin_size)))
        y_rel[idx:idx + g] = lvl
        idx += g
    xgr = xgb.XGBRanker(
        objective="rank:pairwise", n_estimators=500, max_depth=6,
        learning_rate=0.05, random_state=seed, verbosity=0,
    )
    xgr.fit(X_trn, y_rel, group=grp_trn)
    s_xgb = xgr.predict(X_tst)

    # 3. Pointwise GBR
    gbr = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                     learning_rate=0.05, random_state=seed)
    gbr.fit(X_trn, y_trn)
    s_gbr = gbr.predict(X_tst)

    return s_lgbm, s_xgb, s_gbr, lgbm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=2400)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--n_single", type=int, default=8)
    ap.add_argument("--n_synth", type=int, default=8)
    ap.add_argument("--k_neighbors", type=int, default=5)
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", type=str, nargs="+", default=[
        "us.amazon.nova-lite-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
    ])
    args = ap.parse_args()

    trn, tst = load_grefer_samples()
    rng = np.random.RandomState(args.seed)
    if args.n_train >= len(trn):
        trn_eval = trn
    else:
        sel = rng.choice(len(trn), size=args.n_train, replace=False)
        trn_eval = [trn[i] for i in sel]
    # Same tst_sel as before (consume the same seed state)
    _consume = rng.choice(len(trn), size=300, replace=False) if args.n_train >= len(trn) else None
    # To stay aligned with earlier scripts that did n_train then n_test,
    # re-seed from args.seed path: consume `sel for trn` then pick tst.
    rng2 = np.random.RandomState(args.seed)
    _ = rng2.choice(len(trn), size=min(args.n_train, len(trn)), replace=False)
    tst_sel = rng2.choice(len(tst), size=args.n_test, replace=False)
    tst_eval = [tst[i] for i in tst_sel]
    print(f"  trn_eval={len(trn_eval)}  tst_eval={len(tst_eval)}")

    iid_reviews = load_reviews()
    cache_path = Path("results/rag_bandit/titan_cache.json")
    pool_cache = Path("results/rag_bandit/pool_cache")

    clients = {m: BedrockLLM(model_id=m, max_tokens=300, temperature=0.0)
               for m in args.models}
    embed_llm = clients[args.models[0]]

    print("Building reference-style centroid...")
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

    print(f"\n=== Stage 1: Generate expanded pools ===")
    trn_pack = build_or_load_pool_exp(
        "train", trn_eval, args.models, args.n_single, args.n_synth,
        args.max_workers, iid_reviews, ref_centroid, cache_path,
        embed_llm, clients, args.seed, pool_cache,
    )
    tst_pack = build_or_load_pool_exp(
        "test", tst_eval, args.models, args.n_single, args.n_synth,
        args.max_workers, iid_reviews, ref_centroid, cache_path,
        embed_llm, clients, args.seed, pool_cache,
    )

    print(f"\n=== Stage 2: Load cross-encoder ===")
    from sentence_transformers import CrossEncoder
    # ms-marco-MiniLM-L-6-v2 is fast (~20M params) and well-calibrated for
    # relevance scoring. Scores are logits ~ [-10, 10].
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",
                                   max_length=512)

    print(f"\n=== Stage 3: kNN neighbors (k={args.k_neighbors}) ===")
    trn_knn = knn_refs_for_samples(trn_eval, trn, embed_llm, cache_path,
                                    args.max_workers, args.k_neighbors)
    tst_knn = knn_refs_for_samples(tst_eval, trn, embed_llm, cache_path,
                                    args.max_workers, args.k_neighbors)

    print(f"\n=== Stage 4: Featurize ===")
    print("Train:")
    X_trn, meta_trn = featurize_final(
        trn_pack, embed_llm, ref_centroid, exemplar_ref,
        trn_ref_embs_norm, cache_path, args.max_workers,
        trn_knn, cross_encoder,
    )
    print("Test:")
    X_tst, meta_tst = featurize_final(
        tst_pack, embed_llm, ref_centroid, exemplar_ref,
        trn_ref_embs_norm, cache_path, args.max_workers,
        tst_knn, cross_encoder,
    )
    print(f"  X_train: {X_trn.shape}  X_test: {X_tst.shape}")

    print(f"\n=== Stage 5: True-F1 labels ===")
    y_trn = bert_f1_batch([m[4] for m in meta_trn], [m[5] for m in meta_trn])
    y_tst = bert_f1_batch([m[4] for m in meta_tst], [m[5] for m in meta_tst])
    grp_trn, by_sample_trn = _groups_from_meta(meta_trn)
    grp_tst, by_sample_tst = _groups_from_meta(meta_tst)
    oracle_trn = np.array([max(y_trn[ks]) for ks in by_sample_trn.values()])
    oracle_tst = np.array([max(y_tst[ks]) for ks in by_sample_tst.values()])
    print(f"  train mean={y_trn.mean():.4f}  test mean={y_tst.mean():.4f}")
    print(f"  train oracle: {oracle_trn.mean():.4f}")
    print(f"  test  oracle: {oracle_tst.mean():.4f}")

    print(f"\n=== Stage 6: Fit ensemble of rankers ===")
    s_lgbm, s_xgb, s_gbr, lgbm = ensemble_rank_scores(
        X_trn, y_trn, grp_trn, X_tst, args.seed)

    # Average of within-group rank percentiles
    def within_group_percentile(scores, grp_sizes):
        out = np.zeros_like(scores)
        idx = 0
        for g in grp_sizes:
            block = scores[idx:idx + g]
            order = np.argsort(block)
            pct = np.zeros_like(block)
            pct[order] = np.linspace(0, 1, g)
            out[idx:idx + g] = pct
            idx += g
        return out

    pct_lgbm = within_group_percentile(s_lgbm, grp_tst)
    pct_xgb = within_group_percentile(s_xgb, grp_tst)
    pct_gbr = within_group_percentile(s_gbr, grp_tst)
    s_ensemble = (pct_lgbm + pct_xgb + pct_gbr) / 3

    feat_ns = feat_names(args.models)
    print("\n  LambdaRank top-15 features:")
    imps = lgbm.feature_importances_
    for k in np.argsort(-imps)[:15]:
        print(f"    {feat_ns[k]:<34} {int(imps[k])}")

    print(f"\n=== Stage 7: Evaluate ===")
    proxy_texts = [
        f"{tst_pack['biz_profiles'][m[0]]} {tst_pack['user_profiles'][m[0]]} {exemplar_ref}".strip()
        for m in meta_tst
    ]
    proxy_f1 = bert_f1_batch([m[4] for m in meta_tst], proxy_texts)

    picks = {"ensemble": [], "lgbm": [], "xgb": [], "gbr": [], "proxy": [],
             "oracle": []}
    from collections import Counter
    pick_breakdown = Counter()
    for i, ks in by_sample_tst.items():
        ks = list(ks)
        k_ens = ks[int(np.argmax(s_ensemble[ks]))]
        k_lg  = ks[int(np.argmax(s_lgbm[ks]))]
        k_xg  = ks[int(np.argmax(s_xgb[ks]))]
        k_gb  = ks[int(np.argmax(s_gbr[ks]))]
        k_px  = ks[int(np.argmax(proxy_f1[ks]))]
        k_or  = ks[int(np.argmax(y_tst[ks]))]
        picks["ensemble"].append(y_tst[k_ens])
        picks["lgbm"].append(y_tst[k_lg])
        picks["xgb"].append(y_tst[k_xg])
        picks["gbr"].append(y_tst[k_gb])
        picks["proxy"].append(y_tst[k_px])
        picks["oracle"].append(y_tst[k_or])
        pick_breakdown[(meta_tst[k_ens][1], meta_tst[k_ens][2])] += 1

    def summ(name, vals):
        arr = np.array(vals)
        print(f"  {name:<10}: {arr.mean():.4f} ± {arr.std():.4f}  "
              f"%>=0.50 {100*(arr>=0.5).mean():.1f}%  "
              f"%>=0.55 {100*(arr>=0.55).mean():.1f}%  "
              f"%>=0.60 {100*(arr>=0.6).mean():.1f}%")

    n = len(picks["ensemble"])
    print(f"\n  === Test F1 (n={n}) ===")
    summ("ENSEMBLE",  picks["ensemble"])
    summ("LambdaRank", picks["lgbm"])
    summ("XGBRanker",  picks["xgb"])
    summ("PointwiseGBR", picks["gbr"])
    summ("PROXY",     picks["proxy"])
    summ("ORACLE",    picks["oracle"])

    print(f"\n  Ensemble picks (model, style): {dict(pick_breakdown)}")

    out_dir = Path("results/rag_bandit")
    out_file = out_dir / f"reranker_final_n{args.n_train}tr_n{args.n_test}te.json"
    pw = np.array(picks["ensemble"])
    with open(out_file, "w") as f:
        json.dump({
            "summary": {
                "n_train": args.n_train, "n_test": args.n_test,
                "n_single": args.n_single,
                "ensemble_f1": float(pw.mean()),
                "lgbm_f1": float(np.array(picks["lgbm"]).mean()),
                "xgb_f1": float(np.array(picks["xgb"]).mean()),
                "gbr_f1": float(np.array(picks["gbr"]).mean()),
                "proxy_f1": float(np.array(picks["proxy"]).mean()),
                "oracle_f1": float(np.array(picks["oracle"]).mean()),
                "train_oracle": float(oracle_trn.mean()),
                "pick_breakdown": {f"{k[0]}__{k[1]}": v
                                    for k, v in pick_breakdown.items()},
                "feature_importances": {n_: int(v) for n_, v in
                                          zip(feat_ns, imps)},
            },
        }, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
