# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Reranker: expand the candidate pool with new generation styles to raise
the oracle ceiling (which currently caps us at 0.5355).

Builds on reranker_final.py by ADDING three new candidate styles to the
existing cached pool:

  Style C: few-shot exemplar priming. Prime the LLM with k=2 nearest
           (train prompt, train ref) pairs so it imitates house style from
           worked examples rather than from rule lists. One cand/model.

  Style D: high-temperature (t=0.7) multi-review synthesis for lexical
           diversity. 2 samples per model → the argmax-over-pool often
           lives in the tail of the sample distribution.

  Style E: exemplar-anchored paraphrase. Show the model the exemplar ref +
           this user/biz profile + top-3 reviews; ask it to "write the same
           style of explanation for THIS user". One cand/model.

Re-uses existing pool_cache + titan_cache + cross-encoder; only pays for
the incremental LLM calls.

Usage:
  python3 scripts/reranker_bigpool.py --n_train 800 --n_test 200 \
      --n_single 6 --n_synth 4 --n_temp 2 --max_workers 10
"""

import argparse, json, os, pickle, sys, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.reranker_v2 import (
    load_reviews, extract_user_profile, extract_biz_profile, filter_reviews,
    build_prompt_A, build_prompt_B, GREFER_SYSTEM_B,
)
from scripts.reranker_pairwise import fit_lambdarank, _groups_from_meta
from scripts.reranker_final import (
    build_or_load_pool_exp, knn_refs_for_samples,
    invoke_with_retry, mmr_select,
    _count_numbers, _count_pos, _sent_count, _ne_hint,
    ensemble_rank_scores,
)
from src.bedrock_llm import BedrockLLM
from scripts.rag_bandit_pipeline import (
    load_grefer_samples, invoke_bedrock,
    bert_f1_batch, embed_texts_titan, _short_context, SYSTEM_PROMPT,
)


def build_prompt_C(sample, neighbors, reviews):
    """Few-shot exemplar priming. neighbors: list of (prompt_ctx, ref_text)."""
    ctx = _short_context(sample.prompt)
    ex_block = "\n\n".join(
        f"Example {i+1} context: {ncx}\n"
        f"Example {i+1} explanation: {nref}"
        for i, (ncx, nref) in enumerate(neighbors)
    )
    rev_block = "\n".join(f"- ({r['rating']}\u2605) {r['text']}"
                           for r in reviews[:3])
    return (
        "You are writing a recommendation explanation in a specific house "
        "style. First, study these examples of the exact style:\n\n"
        f"{ex_block}\n\n---\n\n"
        "Now write an explanation for THIS user-business pair, matching the "
        "style of the examples above exactly (same sentence structure, same "
        "generic-attribute vocabulary, same length ~20-30 words, ONE "
        "sentence):\n\n"
        f"Target context: {ctx}\n\n"
        f"Real Google Maps reviews of this business (use as grounding):\n"
        f"{rev_block}\n\n"
        "Explanation:"
    )


def build_prompt_D(sample, reviews):
    """Same as style B but meant to be sampled at higher temperature."""
    ctx = _short_context(sample.prompt)
    rev_block = "\n".join(f"- ({r['rating']}\u2605) {r['text']}"
                           for r in reviews)
    return (
        GREFER_SYSTEM_B + "\n\n"
        "Context:\n" + ctx + "\n\n"
        "Real user reviews of this business (select the recurring themes):\n"
        f"{rev_block}\n\n"
        "Task: synthesize the top 3-4 reasons that THIS user (given their "
        "profile) would enjoy this business. Output ONE sentence in the "
        "house style. Use natural, varied phrasing.\n\n"
        "Explanation:"
    )


def build_prompt_E(sample, exemplar_ref, reviews):
    """Paraphrase exemplar_ref using this user/biz context."""
    ctx = _short_context(sample.prompt)
    rev_block = "\n".join(f"- ({r['rating']}\u2605) {r['text']}"
                           for r in reviews[:3])
    return (
        "Here is an exemplar explanation in the exact house style to match:\n"
        f"  \"{exemplar_ref}\"\n\n"
        "Rewrite this exemplar to fit THIS user-business pair. Keep the "
        "same sentence structure (\"The user would enjoy the business "
        "because...\"), same length (20-30 words), same generic-attribute "
        "vocabulary. Only the reasons should change to match what THIS "
        "business is good at, based on the real reviews below.\n\n"
        f"Target context: {ctx}\n\n"
        f"Real reviews:\n{rev_block}\n\n"
        "Rewritten explanation:"
    )


def gen_new_styles(samples, models, pack, clients, max_workers,
                    trn_neighbors, exemplar_ref, iid_reviews, n_temp=2):
    """Generate style-C (few-shot), D (t=0.7 × n_temp), E (exemplar paraphrase)
    candidates on top of the existing pack. Returns updated preds dict + new keys."""
    ranked = pack["ranked"]
    new_preds = {}
    jobs = []
    for model_id in models:
        for i, s in enumerate(samples):
            top = ranked[i]
            if not top: continue
            revs = [t[0] for t in top[:5]]

            # C: few-shot with nearest 2 train neighbors
            if trn_neighbors[i]:
                jobs.append(("C", model_id, i, 0, revs,
                             trn_neighbors[i], None, 0.0))

            # D: t=0.7 synthesis × n_temp samples
            for t in range(n_temp):
                jobs.append(("D", model_id, i, t, revs, None, None, 0.7))

            # E: exemplar paraphrase
            jobs.append(("E", model_id, i, 0, revs, None, exemplar_ref, 0.0))

    print(f"  Generating {len(jobs)} new-style candidates...")

    def gen(job):
        style, model_id, i, j, revs, neighbors, exemplar, temp = job
        s = samples[i]
        if style == "C":
            # neighbors: list of Sample objects; extract (ctx, ref)
            nb_pairs = [(_short_context(n.prompt),
                          n.reference.replace("### ", "").strip())
                         for n in neighbors]
            msg = build_prompt_C(s, nb_pairs, revs)
        elif style == "D":
            msg = build_prompt_D(s, revs)
        else:  # E
            msg = build_prompt_E(s, exemplar, revs)
        out = invoke_with_retry(clients[model_id], msg, temperature=temp)
        return style, model_id, i, j, out or ""

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(gen, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="  newgen"):
            style, m, i, j, out = fut.result()
            new_preds[(style, m, i, j)] = out

    return new_preds


def nearest_train_samples(query_samples, train_samples, embed_llm, cache_path,
                           max_workers, k=2):
    """For each query sample, return k nearest train Samples (for few-shot)."""
    def keyify(s):
        return (f"User: {extract_user_profile(s.prompt)} | "
                f"Business: {extract_biz_profile(s.prompt)}")
    q_texts = [keyify(s) for s in query_samples]
    t_texts = [keyify(s) for s in train_samples]
    q = embed_texts_titan(embed_llm, q_texts, cache_path=cache_path,
                           max_workers=max_workers)
    t = embed_texts_titan(embed_llm, t_texts, cache_path=cache_path,
                           max_workers=max_workers)
    q_n = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    t_n = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    sims = q_n @ t_n.T
    q_keys = [(s.uid, s.iid) for s in query_samples]
    t_keys = [(t.uid, t.iid) for t in train_samples]
    out = []
    for i in range(len(query_samples)):
        order = np.argsort(-sims[i])
        picks = []
        for j in order:
            if t_keys[int(j)] == q_keys[i]: continue
            picks.append(train_samples[int(j)])
            if len(picks) >= k: break
        out.append(picks)
    return out


def featurize_big(pack, all_preds, embed_llm, ref_centroid, exemplar_ref,
                    trn_ref_embs_norm, cache_path, max_workers,
                    knn_refs_per_sample, cross_encoder, models):
    """Like featurize_final but iterates over ALL style keys in all_preds,
    not just A/B."""
    samples = pack["samples"]
    ranked = pack["ranked"]
    user_profiles = pack["user_profiles"]
    biz_profiles = pack["biz_profiles"]

    # Embed everything we'll need
    pred_texts = [v for v in all_preds.values() if v.strip()]
    to_embed = list(set(pred_texts + biz_profiles + user_profiles
                          + [exemplar_ref]))
    print(f"  Embedding {len(to_embed)} unique texts...")
    embs = embed_texts_titan(embed_llm, to_embed, cache_path=cache_path,
                              max_workers=max_workers)
    emb_idx = {t: k for k, t in enumerate(to_embed)}
    embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    def embn(t):
        return embs_n[emb_idx[t]] if t.strip() in emb_idx else None
    exemplar_vec = embs_n[emb_idx[exemplar_ref]]

    # Collect all candidate keys in deterministic order
    candidate_keys = []
    # group by (i, model, style) to keep features comparable
    for i in range(len(samples)):
        if not ranked[i]: continue
        for m in models:
            for style in ("A", "B", "C", "D", "E"):
                if style in ("A",):
                    for j in range(20):
                        k = (style, m, i, j)
                        if all_preds.get(k, "").strip():
                            candidate_keys.append(k)
                elif style == "D":
                    for j in range(10):
                        k = (style, m, i, j)
                        if all_preds.get(k, "").strip():
                            candidate_keys.append(k)
                else:
                    k = (style, m, i, -1 if style == "B" else 0)
                    if all_preds.get(k, "").strip():
                        candidate_keys.append(k)

    # kNN BERTScore features
    print(f"  kNN BERTScore pairs...")
    bert_pairs, pair_back = [], {}
    for key in candidate_keys:
        (style, m, i, j) = key
        pred = all_preds[key]
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
                  for rk in range(k_per)
                  if (style, m, i, j, rk) in pair_back]
        if scores:
            arr = np.array(scores)
            knn_stats[key] = (float(arr.mean()), float(arr.max()), float(arr.std()))
        else:
            knn_stats[key] = (0.0, 0.0, 0.0)

    # Cross-encoder features
    print(f"  Cross-encoder scoring...")
    up_targets = [f"{user_profiles[i]} {biz_profiles[i]}"
                   for i in range(len(samples))]
    ce_pairs_up, ce_pairs_knn = [], []
    ce_up_back, ce_knn_back = {}, {}
    for key in candidate_keys:
        (style, m, i, j) = key
        pred = all_preds[key]
        ce_up_back[key] = len(ce_pairs_up)
        ce_pairs_up.append((pred, up_targets[i]))
        knn_concat = " ".join(knn_refs_per_sample[i][:3])
        ce_knn_back[key] = len(ce_pairs_knn)
        ce_pairs_knn.append((pred, knn_concat))
    ce_up_scores = cross_encoder.predict(ce_pairs_up, show_progress_bar=True,
                                          batch_size=64)
    ce_knn_scores = cross_encoder.predict(ce_pairs_knn, show_progress_bar=True,
                                            batch_size=64)

    # Assemble rows in sample-contiguous order
    STYLE_ONEHOT = ["A", "B", "C", "D", "E"]

    def row(pred, p_vec, u_vec, b_vec, rev_sim_u, rev_sim_r, rating, src_len,
             style, m, knn, ce_up, ce_knn, cand_rank):
        cos_user = float(p_vec @ u_vec) if u_vec is not None else 0.0
        cos_biz = float(p_vec @ b_vec) if b_vec is not None else 0.0
        cos_ref = float(p_vec @ ref_centroid)
        cos_exe = float(p_vec @ exemplar_vec)
        sims_nearest = trn_ref_embs_norm @ p_vec
        top5 = float(np.sort(sims_nearest)[-5:].mean())
        feats = [
            cos_user, cos_biz, cos_ref, cos_exe, top5,
            float(len(pred.split())), float(len(pred)),
            1.0 if pred.lstrip().startswith("The user would enjoy the business") else 0.0,
            float(_count_numbers(pred)), float(_count_pos(pred)),
            float(_sent_count(pred)), float(_ne_hint(pred)),
            rating, src_len, rev_sim_u, rev_sim_r,
            float(cand_rank),
            knn[0], knn[1], knn[2], ce_up, ce_knn,
        ]
        # style one-hot
        feats += [1.0 if style == s_ else 0.0 for s_ in STYLE_ONEHOT]
        # model one-hot
        feats += [1.0 if m == mm else 0.0 for mm in models]
        return feats

    X, meta = [], []
    for i, s in enumerate(samples):
        if not ranked[i]: continue
        u_vec = embn(user_profiles[i])
        b_vec = embn(biz_profiles[i])
        top = ranked[i]
        # Pre-compute B/D/C/E source-review stats (use top-3 reviews)
        used = top[:min(4, len(top))]
        used_ratings = [t[0].get("rating") or 3 for t in used]
        used_srclen = float(np.mean([len(t[0]["text"].split())
                                      for t in used])) if used else 0.0

        for m in models:
            # Style A: per-review
            for j in range(20):
                k = ("A", m, i, j)
                pred = all_preds.get(k, "")
                if not pred.strip(): continue
                if j >= len(top): break
                rev, rev_su, rev_sr = top[j]
                p_vec = embn(pred)
                feats = row(pred, p_vec, u_vec, b_vec, rev_su, rev_sr,
                             float(rev.get("rating") or 3),
                             float(len(rev["text"].split())),
                             "A", m, knn_stats[k],
                             float(ce_up_scores[ce_up_back[k]]),
                             float(ce_knn_scores[ce_knn_back[k]]),
                             j)
                X.append(feats)
                meta.append((i, m, "A", j, pred, s.reference))

            # Style B
            k = ("B", m, i, -1)
            pred = all_preds.get(k, "")
            if pred.strip():
                p_vec = embn(pred)
                rev_su = float(np.mean([t[1] for t in used])) if used else 0.0
                rev_sr = float(np.mean([t[2] for t in used])) if used else 0.0
                feats = row(pred, p_vec, u_vec, b_vec, rev_su, rev_sr,
                             float(np.mean(used_ratings)) if used_ratings else 3.0,
                             used_srclen, "B", m, knn_stats[k],
                             float(ce_up_scores[ce_up_back[k]]),
                             float(ce_knn_scores[ce_knn_back[k]]),
                             0)
                X.append(feats)
                meta.append((i, m, "B", -1, pred, s.reference))

            # Style C (few-shot)
            k = ("C", m, i, 0)
            pred = all_preds.get(k, "")
            if pred.strip():
                p_vec = embn(pred)
                feats = row(pred, p_vec, u_vec, b_vec,
                             float(np.mean([t[1] for t in used])) if used else 0.0,
                             float(np.mean([t[2] for t in used])) if used else 0.0,
                             float(np.mean(used_ratings)) if used_ratings else 3.0,
                             used_srclen, "C", m, knn_stats[k],
                             float(ce_up_scores[ce_up_back[k]]),
                             float(ce_knn_scores[ce_knn_back[k]]),
                             0)
                X.append(feats)
                meta.append((i, m, "C", 0, pred, s.reference))

            # Style D (sampled)
            for j in range(10):
                k = ("D", m, i, j)
                pred = all_preds.get(k, "")
                if not pred.strip(): continue
                p_vec = embn(pred)
                feats = row(pred, p_vec, u_vec, b_vec,
                             float(np.mean([t[1] for t in used])) if used else 0.0,
                             float(np.mean([t[2] for t in used])) if used else 0.0,
                             float(np.mean(used_ratings)) if used_ratings else 3.0,
                             used_srclen, "D", m, knn_stats[k],
                             float(ce_up_scores[ce_up_back[k]]),
                             float(ce_knn_scores[ce_knn_back[k]]),
                             j)
                X.append(feats)
                meta.append((i, m, "D", j, pred, s.reference))

            # Style E (exemplar paraphrase)
            k = ("E", m, i, 0)
            pred = all_preds.get(k, "")
            if pred.strip():
                p_vec = embn(pred)
                feats = row(pred, p_vec, u_vec, b_vec,
                             float(np.mean([t[1] for t in used])) if used else 0.0,
                             float(np.mean([t[2] for t in used])) if used else 0.0,
                             float(np.mean(used_ratings)) if used_ratings else 3.0,
                             used_srclen, "E", m, knn_stats[k],
                             float(ce_up_scores[ce_up_back[k]]),
                             float(ce_knn_scores[ce_knn_back[k]]),
                             0)
                X.append(feats)
                meta.append((i, m, "E", 0, pred, s.reference))
    return np.array(X), meta


def feat_names_big(models):
    base = [
        "cos_user", "cos_biz", "cos_ref_centroid", "cos_exemplar",
        "top5_nearest_trn_ref", "n_words", "n_chars", "starts_ok",
        "n_nums", "n_generic_positive", "sent_count", "has_named_entity",
        "rating", "src_len", "rev_sim_user", "rev_sim_ref", "cand_rank",
        "knn_bert_mean", "knn_bert_max", "knn_bert_std",
        "ce_user_biz", "ce_knn_refs",
    ]
    base += [f"style={s}" for s in ("A", "B", "C", "D", "E")]
    base += [f"model={m}" for m in models]
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--n_single", type=int, default=6)
    ap.add_argument("--n_synth", type=int, default=4)
    ap.add_argument("--n_temp", type=int, default=2)
    ap.add_argument("--k_neighbors", type=int, default=5)
    ap.add_argument("--max_workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", type=str, nargs="+", default=[
        "us.amazon.nova-lite-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
    ])
    args = ap.parse_args()

    trn, tst = load_grefer_samples()
    rng2 = np.random.RandomState(args.seed)
    _ = rng2.choice(len(trn), size=min(args.n_train, len(trn)), replace=False)
    tst_sel = rng2.choice(len(tst), size=args.n_test, replace=False)
    tst_eval = [tst[i] for i in tst_sel]
    rng = np.random.RandomState(args.seed)
    if args.n_train >= len(trn): trn_eval = trn
    else:
        sel = rng.choice(len(trn), size=args.n_train, replace=False)
        trn_eval = [trn[i] for i in sel]
    print(f"  trn_eval={len(trn_eval)}  tst_eval={len(tst_eval)}")

    iid_reviews = load_reviews()
    cache_path = Path("results/rag_bandit/titan_cache.json")
    pool_cache = Path("results/rag_bandit/pool_cache")
    bigcache = pool_cache / f"big_new_styles_tr{len(trn_eval)}_te{len(tst_eval)}_t{args.n_temp}.pkl"

    clients = {m: BedrockLLM(model_id=m, max_tokens=300, temperature=0.0)
                for m in args.models}
    embed_llm = clients[args.models[0]]

    print("Building reference-style centroid...")
    all_trn_refs = [t.reference.replace("### ", "").strip() for t in trn]
    trn_ref_embs = embed_texts_titan(embed_llm, all_trn_refs,
                                       cache_path=cache_path,
                                       max_workers=args.max_workers)
    ref_centroid = trn_ref_embs.mean(axis=0)
    ref_centroid /= (np.linalg.norm(ref_centroid) + 1e-9)
    trn_ref_embs_norm = trn_ref_embs / (
        np.linalg.norm(trn_ref_embs, axis=1, keepdims=True) + 1e-9)
    exemplar_ref = all_trn_refs[int(np.argmax(trn_ref_embs_norm @ ref_centroid))]

    print("\n=== Stage 1: Load A/B pool (cached) ===")
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

    print("\n=== Stage 2: Find nearest train neighbors for Style C ===")
    trn_neighbors_trn = nearest_train_samples(
        trn_eval, trn, embed_llm, cache_path, args.max_workers, k=2)
    trn_neighbors_tst = nearest_train_samples(
        tst_eval, trn, embed_llm, cache_path, args.max_workers, k=2)

    print("\n=== Stage 3: Generate new-style candidates (C/D/E) ===")
    if bigcache.exists():
        print(f"  Loading cached new-style preds: {bigcache.name}")
        with open(bigcache, "rb") as f:
            big_new = pickle.load(f)
        new_trn = big_new["trn"]; new_tst = big_new["tst"]
    else:
        print("  [train]")
        new_trn = gen_new_styles(trn_eval, args.models, trn_pack, clients,
                                   args.max_workers, trn_neighbors_trn,
                                   exemplar_ref, iid_reviews, args.n_temp)
        print("  [test]")
        new_tst = gen_new_styles(tst_eval, args.models, tst_pack, clients,
                                   args.max_workers, trn_neighbors_tst,
                                   exemplar_ref, iid_reviews, args.n_temp)
        with open(bigcache, "wb") as f:
            pickle.dump({"trn": new_trn, "tst": new_tst}, f)
        print(f"  Cached → {bigcache}")

    # Merge predictions
    trn_all = dict(trn_pack["preds"]); trn_all.update(new_trn)
    tst_all = dict(tst_pack["preds"]); tst_all.update(new_tst)
    n_trn_new = len([k for k, v in new_trn.items() if v.strip()])
    n_tst_new = len([k for k, v in new_tst.items() if v.strip()])
    print(f"  Non-empty new preds: trn={n_trn_new}  tst={n_tst_new}")

    print("\n=== Stage 4: Cross-encoder + kNN refs ===")
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",
                                   max_length=512)
    trn_knn = knn_refs_for_samples(trn_eval, trn, embed_llm, cache_path,
                                    args.max_workers, args.k_neighbors)
    tst_knn = knn_refs_for_samples(tst_eval, trn, embed_llm, cache_path,
                                    args.max_workers, args.k_neighbors)

    print("\n=== Stage 5: Featurize ===")
    print("Train:")
    X_trn, meta_trn = featurize_big(
        trn_pack, trn_all, embed_llm, ref_centroid, exemplar_ref,
        trn_ref_embs_norm, cache_path, args.max_workers,
        trn_knn, cross_encoder, args.models,
    )
    print("Test:")
    X_tst, meta_tst = featurize_big(
        tst_pack, tst_all, embed_llm, ref_centroid, exemplar_ref,
        trn_ref_embs_norm, cache_path, args.max_workers,
        tst_knn, cross_encoder, args.models,
    )
    print(f"  X_train: {X_trn.shape}  X_test: {X_tst.shape}")

    print("\n=== Stage 6: True-F1 labels ===")
    y_trn = bert_f1_batch([m[4] for m in meta_trn], [m[5] for m in meta_trn])
    y_tst = bert_f1_batch([m[4] for m in meta_tst], [m[5] for m in meta_tst])
    grp_trn, by_sample_trn = _groups_from_meta(meta_trn)
    grp_tst, by_sample_tst = _groups_from_meta(meta_tst)
    oracle_trn = np.array([max(y_trn[list(ks)]) for ks in by_sample_trn.values()])
    oracle_tst = np.array([max(y_tst[list(ks)]) for ks in by_sample_tst.values()])
    print(f"  train mean={y_trn.mean():.4f}  test mean={y_tst.mean():.4f}")
    print(f"  train oracle: {oracle_trn.mean():.4f}")
    print(f"  test  oracle: {oracle_tst.mean():.4f}")

    # Per-style oracle breakdown (test)
    from collections import defaultdict
    by_style = defaultdict(list)
    for k, m in enumerate(meta_tst):
        by_style[m[2]].append(y_tst[k])
    print(f"\n  Mean F1 per style (test):")
    for s_, vs in sorted(by_style.items()):
        print(f"    {s_}: {np.mean(vs):.4f}  (n={len(vs)})")

    # Oracle achievable if we only kept style X on test
    print(f"\n  Oracle test if pool = only one style:")
    for s_ in ("A", "B", "C", "D", "E"):
        per_sample_max = []
        for i, ks in by_sample_tst.items():
            sub = [k for k in ks if meta_tst[k][2] == s_]
            if sub: per_sample_max.append(max(y_tst[sub]))
        if per_sample_max:
            print(f"    only {s_}: {np.mean(per_sample_max):.4f}  (n={len(per_sample_max)})")

    print("\n=== Stage 7: Fit ensemble ===")
    s_lgbm, s_xgb, s_gbr, lgbm = ensemble_rank_scores(
        X_trn, y_trn, grp_trn, X_tst, args.seed)

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
    s_ens = (pct_lgbm + pct_xgb + pct_gbr) / 3

    feat_ns = feat_names_big(args.models)
    print("\n  LambdaRank top-20 features:")
    imps = lgbm.feature_importances_
    for k in np.argsort(-imps)[:20]:
        print(f"    {feat_ns[k]:<30} {int(imps[k])}")

    print("\n=== Stage 8: Evaluate ===")
    proxy_texts = [
        f"{tst_pack['biz_profiles'][m[0]]} {tst_pack['user_profiles'][m[0]]} {exemplar_ref}".strip()
        for m in meta_tst
    ]
    proxy_f1 = bert_f1_batch([m[4] for m in meta_tst], proxy_texts)

    picks = {"ensemble": [], "lgbm": [], "xgb": [], "gbr": [],
              "proxy": [], "oracle": []}
    from collections import Counter
    pick_breakdown = Counter()
    for i, ks in by_sample_tst.items():
        ks = list(ks)
        k_ens = ks[int(np.argmax(s_ens[ks]))]
        picks["ensemble"].append(y_tst[k_ens])
        picks["lgbm"].append(y_tst[ks[int(np.argmax(s_lgbm[ks]))]])
        picks["xgb"].append(y_tst[ks[int(np.argmax(s_xgb[ks]))]])
        picks["gbr"].append(y_tst[ks[int(np.argmax(s_gbr[ks]))]])
        picks["proxy"].append(y_tst[ks[int(np.argmax(proxy_f1[ks]))]])
        picks["oracle"].append(y_tst[ks[int(np.argmax(y_tst[ks]))]])
        pick_breakdown[(meta_tst[k_ens][1], meta_tst[k_ens][2])] += 1

    def summ(name, vals):
        arr = np.array(vals)
        print(f"  {name:<12}: {arr.mean():.4f} \u00b1 {arr.std():.4f}  "
              f"%>=0.50 {100*(arr>=0.5).mean():.1f}%  "
              f"%>=0.55 {100*(arr>=0.55).mean():.1f}%  "
              f"%>=0.60 {100*(arr>=0.6).mean():.1f}%")

    n = len(picks["ensemble"])
    print(f"\n  === Test F1 (n={n}) ===")
    summ("ENSEMBLE", picks["ensemble"])
    summ("LambdaRank", picks["lgbm"])
    summ("XGBRanker", picks["xgb"])
    summ("PointwiseGBR", picks["gbr"])
    summ("PROXY", picks["proxy"])
    summ("ORACLE", picks["oracle"])

    print(f"\n  Ensemble picks (model, style): {dict(pick_breakdown)}")

    out_dir = Path("results/rag_bandit")
    out_file = out_dir / f"reranker_bigpool_n{args.n_train}tr_n{args.n_test}te.json"
    with open(out_file, "w") as f:
        json.dump({
            "summary": {
                "n_train": args.n_train, "n_test": args.n_test,
                "ensemble_f1": float(np.mean(picks["ensemble"])),
                "lgbm_f1": float(np.mean(picks["lgbm"])),
                "xgb_f1": float(np.mean(picks["xgb"])),
                "gbr_f1": float(np.mean(picks["gbr"])),
                "proxy_f1": float(np.mean(picks["proxy"])),
                "oracle_f1": float(np.mean(picks["oracle"])),
                "train_oracle": float(oracle_trn.mean()),
                "per_style_mean_f1": {s_: float(np.mean(vs))
                                       for s_, vs in by_style.items()},
                "pick_breakdown": {f"{k[0]}__{k[1]}": v
                                    for k, v in pick_breakdown.items()},
                "feature_importances": {n_: int(v) for n_, v in
                                          zip(feat_ns, imps)},
            },
        }, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
