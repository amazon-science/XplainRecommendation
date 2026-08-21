"""Add style-F candidates (length-tuned, biz-name-allowed) to the bigpool
and re-run the ensemble. F targets the actual G-Refer ref distribution
(median 28 words, 12% mention biz name).

Incremental: only generates ~800 new LLM calls (200 test * 2 models * 2 F-variants).

Usage:
  python3 scripts/reranker_styleF.py --n_train 800 --n_test 200 \
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
)
from scripts.reranker_pairwise import _groups_from_meta
from scripts.reranker_final import (
    build_or_load_pool_exp, knn_refs_for_samples, invoke_with_retry,
    ensemble_rank_scores,
)
from scripts.reranker_bigpool import (
    nearest_train_samples, featurize_big, feat_names_big,
)
from src.bedrock_llm import BedrockLLM
from scripts.rag_bandit_pipeline import (
    load_grefer_samples, invoke_bedrock,
    bert_f1_batch, embed_texts_titan, _short_context,
)


def extract_biz_title(prompt):
    m = re.search(r"Business title:\s*([^.\n]+)", prompt)
    return m.group(1).strip() if m else None


def build_prompt_F(sample, reviews, use_biz_name, word_target=28):
    """Length-tuned synthesis. Mentions biz name if use_biz_name=True."""
    ctx = _short_context(sample.prompt)
    biz = extract_biz_title(sample.prompt) or "the business"
    rev_block = "\n".join(f"- ({r['rating']}\u2605) {r['text']}"
                           for r in reviews[:4])

    if use_biz_name and biz != "the business":
        opener = f"\"The user would enjoy {biz} for its...\" or \"The user would enjoy {biz} because...\""
        name_rule = (f"- You may refer to the business by name: \"{biz}\"\n"
                      "- Do NOT name specific dishes unless in the reviews\n")
    else:
        opener = ("\"The user would enjoy the business because of its...\" or "
                   "\"The user would enjoy the business for its...\"")
        name_rule = ("- Refer to the business as \"the business\"\n"
                      "- Do NOT name specific dishes or the business\n")

    return (
        "You are writing a single-sentence explanation in a specific house "
        "style.\n"
        "RULES (strict):\n"
        f"- EXACTLY ONE sentence, {word_target-3}-{word_target+5} words\n"
        f"- Start with: {opener}\n"
        f"{name_rule}"
        "- List 3-4 generic reasons (food, service, atmosphere, prices, "
        "selection, quality, variety, staff, experience) connected with "
        "commas and 'and'\n"
        "- Optionally end with a summary clause like \"making it a pleasant "
        "experience\" or \"for a satisfying visit\"\n\n"
        f"Context:\n{ctx}\n\n"
        f"Real reviews:\n{rev_block}\n\n"
        "Write the explanation now, output only the sentence:"
    )


def gen_styleF(samples, models, pack, clients, max_workers):
    ranked = pack["ranked"]
    preds = {}
    jobs = []
    for model_id in models:
        for i, s in enumerate(samples):
            top = ranked[i]
            if not top: continue
            revs = [t[0] for t in top[:4]]
            # F0: no biz name
            jobs.append(("F", model_id, i, 0, revs, False))
            # F1: with biz name (will fall back to "the business" if extract fails)
            jobs.append(("F", model_id, i, 1, revs, True))

    print(f"  Generating {len(jobs)} F-style candidates...")

    def gen(job):
        style, m, i, j, revs, use_name = job
        s = samples[i]
        msg = build_prompt_F(s, revs, use_biz_name=use_name, word_target=28)
        out = invoke_with_retry(clients[m], msg, temperature=0.0)
        return style, m, i, j, out or ""

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(gen, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="  F-gen"):
            style, m, i, j, out = fut.result()
            preds[(style, m, i, j)] = out
    return preds


def featurize_with_F(pack, all_preds, embed_llm, ref_centroid, exemplar_ref,
                      trn_ref_embs_norm, cache_path, max_workers,
                      knn_refs_per_sample, cross_encoder, models):
    """Fork of featurize_big that also handles style F (j ∈ {0,1})."""
    from scripts.reranker_bigpool import featurize_big as _fb
    # We'll monkey-patch: iterate style F with j in 0..5
    # Easiest: call _fb which handles D for j<10, and extend by emitting F rows
    # manually. Instead, rewrite the loop inline.
    from scripts.reranker_bigpool import (
        featurize_big,  # we won't use it — inline instead
    )
    # Duplicate the logic from featurize_big but with F added
    from scripts.reranker_final import (
        _count_numbers, _count_pos, _sent_count, _ne_hint,
    )
    samples = pack["samples"]
    ranked = pack["ranked"]
    user_profiles = pack["user_profiles"]
    biz_profiles = pack["biz_profiles"]

    pred_texts = [v for v in all_preds.values() if v.strip()]
    to_embed = list(set(pred_texts + biz_profiles + user_profiles + [exemplar_ref]))
    print(f"  Embedding {len(to_embed)} unique texts...")
    embs = embed_texts_titan(embed_llm, to_embed, cache_path=cache_path,
                              max_workers=max_workers)
    emb_idx = {t: k for k, t in enumerate(to_embed)}
    embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    def embn(t):
        return embs_n[emb_idx[t]] if t.strip() in emb_idx else None
    exemplar_vec = embs_n[emb_idx[exemplar_ref]]

    candidate_keys = []
    for i in range(len(samples)):
        if not ranked[i]: continue
        for m in models:
            for j in range(20):
                k = ("A", m, i, j)
                if all_preds.get(k, "").strip(): candidate_keys.append(k)
            for style in ("B", "C", "E"):
                j = -1 if style == "B" else 0
                k = (style, m, i, j)
                if all_preds.get(k, "").strip(): candidate_keys.append(k)
            for j in range(10):
                k = ("D", m, i, j)
                if all_preds.get(k, "").strip(): candidate_keys.append(k)
            for j in range(6):
                k = ("F", m, i, j)
                if all_preds.get(k, "").strip(): candidate_keys.append(k)

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
            knn_stats[key] = (float(arr.mean()), float(arr.max()),
                               float(arr.std()))
        else:
            knn_stats[key] = (0.0, 0.0, 0.0)

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

    STYLE_ONEHOT = ["A", "B", "C", "D", "E", "F"]

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
            1.0 if pred.lstrip().startswith("The user would enjoy") or
                   pred.lstrip().startswith("You would enjoy") else 0.0,
            float(_count_numbers(pred)), float(_count_pos(pred)),
            float(_sent_count(pred)), float(_ne_hint(pred)),
            rating, src_len, rev_sim_u, rev_sim_r,
            float(cand_rank),
            knn[0], knn[1], knn[2], ce_up, ce_knn,
        ]
        feats += [1.0 if style == s_ else 0.0 for s_ in STYLE_ONEHOT]
        feats += [1.0 if m == mm else 0.0 for mm in models]
        return feats

    X, meta = [], []
    for i, s in enumerate(samples):
        if not ranked[i]: continue
        u_vec = embn(user_profiles[i])
        b_vec = embn(biz_profiles[i])
        top = ranked[i]
        used = top[:min(4, len(top))]
        used_ratings = [t[0].get("rating") or 3 for t in used]
        used_srclen = float(np.mean([len(t[0]["text"].split())
                                      for t in used])) if used else 0.0

        for m in models:
            for j in range(20):
                k = ("A", m, i, j)
                pred = all_preds.get(k, "")
                if not pred.strip(): continue
                if j >= len(top): break
                rev, rev_su, rev_sr = top[j]
                p_vec = embn(pred)
                X.append(row(pred, p_vec, u_vec, b_vec, rev_su, rev_sr,
                              float(rev.get("rating") or 3),
                              float(len(rev["text"].split())),
                              "A", m, knn_stats[k],
                              float(ce_up_scores[ce_up_back[k]]),
                              float(ce_knn_scores[ce_knn_back[k]]),
                              j))
                meta.append((i, m, "A", j, pred, s.reference))

            def add_styled(style, j, k_idx=None):
                k = (style, m, i, k_idx if k_idx is not None else j)
                pred = all_preds.get(k, "")
                if not pred.strip(): return
                p_vec = embn(pred)
                rv_su = float(np.mean([t[1] for t in used])) if used else 0.0
                rv_sr = float(np.mean([t[2] for t in used])) if used else 0.0
                X.append(row(pred, p_vec, u_vec, b_vec, rv_su, rv_sr,
                              float(np.mean(used_ratings)) if used_ratings else 3.0,
                              used_srclen, style, m, knn_stats[k],
                              float(ce_up_scores[ce_up_back[k]]),
                              float(ce_knn_scores[ce_knn_back[k]]),
                              j))
                meta.append((i, m, style, k[-1], pred, s.reference))

            add_styled("B", 0, k_idx=-1)
            add_styled("C", 0)
            for j in range(10):
                add_styled("D", j)
            add_styled("E", 0)
            for j in range(6):
                add_styled("F", j)
    return np.array(X), meta


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
    fcache = pool_cache / f"styleF_tr{len(trn_eval)}_te{len(tst_eval)}.pkl"

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

    print("\n=== Load cached A/B/C/D/E ===")
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
    with open(bigcache, "rb") as f:
        big = pickle.load(f)
    new_trn_old, new_tst_old = big["trn"], big["tst"]

    print("\n=== Generate style F ===")
    if fcache.exists():
        with open(fcache, "rb") as f:
            fpack = pickle.load(f)
        f_trn, f_tst = fpack["trn"], fpack["tst"]
        print(f"  Loaded cached F: trn={len(f_trn)} tst={len(f_tst)}")
    else:
        print("  [train]")
        f_trn = gen_styleF(trn_eval, args.models, trn_pack, clients,
                             args.max_workers)
        print("  [test]")
        f_tst = gen_styleF(tst_eval, args.models, tst_pack, clients,
                             args.max_workers)
        with open(fcache, "wb") as f:
            pickle.dump({"trn": f_trn, "tst": f_tst}, f)
        print(f"  Cached → {fcache}")

    trn_all = dict(trn_pack["preds"]); trn_all.update(new_trn_old); trn_all.update(f_trn)
    tst_all = dict(tst_pack["preds"]); tst_all.update(new_tst_old); tst_all.update(f_tst)

    # Quick F sanity: word counts
    f_lens = [len(p.split()) for p in f_tst.values() if p.strip()]
    print(f"  F test word-count: mean={np.mean(f_lens):.1f} median={np.median(f_lens):.0f}")

    print("\n=== Cross-encoder + kNN ===")
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",
                                   max_length=512)
    trn_knn = knn_refs_for_samples(trn_eval, trn, embed_llm, cache_path,
                                    args.max_workers, args.k_neighbors)
    tst_knn = knn_refs_for_samples(tst_eval, trn, embed_llm, cache_path,
                                    args.max_workers, args.k_neighbors)

    print("\n=== Featurize ===")
    print("Train:")
    X_trn, meta_trn = featurize_with_F(
        trn_pack, trn_all, embed_llm, ref_centroid, exemplar_ref,
        trn_ref_embs_norm, cache_path, args.max_workers,
        trn_knn, cross_encoder, args.models,
    )
    print("Test:")
    X_tst, meta_tst = featurize_with_F(
        tst_pack, tst_all, embed_llm, ref_centroid, exemplar_ref,
        trn_ref_embs_norm, cache_path, args.max_workers,
        tst_knn, cross_encoder, args.models,
    )
    print(f"  X_train: {X_trn.shape}  X_test: {X_tst.shape}")

    print("\n=== True-F1 labels ===")
    y_trn = bert_f1_batch([m[4] for m in meta_trn], [m[5] for m in meta_trn])
    y_tst = bert_f1_batch([m[4] for m in meta_tst], [m[5] for m in meta_tst])
    grp_trn, by_sample_trn = _groups_from_meta(meta_trn)
    grp_tst, by_sample_tst = _groups_from_meta(meta_tst)
    oracle_trn = np.array([max(y_trn[list(ks)]) for ks in by_sample_trn.values()])
    oracle_tst = np.array([max(y_tst[list(ks)]) for ks in by_sample_tst.values()])
    print(f"  train mean={y_trn.mean():.4f}  test mean={y_tst.mean():.4f}")
    print(f"  train oracle: {oracle_trn.mean():.4f}")
    print(f"  test  oracle: {oracle_tst.mean():.4f}")

    from collections import defaultdict
    by_style = defaultdict(list)
    for k, m in enumerate(meta_tst):
        by_style[m[2]].append(y_tst[k])
    print(f"\n  Mean F1 per style:")
    for s_, vs in sorted(by_style.items()):
        print(f"    {s_}: {np.mean(vs):.4f}  (n={len(vs)})")

    print(f"\n  Oracle if pool = only style:")
    for s_ in ("A", "B", "C", "D", "E", "F"):
        per_sample_max = []
        for i, ks in by_sample_tst.items():
            sub = [k for k in ks if meta_tst[k][2] == s_]
            if sub: per_sample_max.append(max(y_tst[sub]))
        if per_sample_max:
            print(f"    only {s_}: {np.mean(per_sample_max):.4f}  "
                  f"(n={len(per_sample_max)})")

    print("\n=== Fit ensemble ===")
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
            out[idx:idx + g] = pct; idx += g
        return out

    pct_lgbm = within_group_percentile(s_lgbm, grp_tst)
    pct_xgb = within_group_percentile(s_xgb, grp_tst)
    pct_gbr = within_group_percentile(s_gbr, grp_tst)
    s_ens = (pct_lgbm + pct_xgb + pct_gbr) / 3

    base_feats = [
        "cos_user", "cos_biz", "cos_ref_centroid", "cos_exemplar",
        "top5_nearest_trn_ref", "n_words", "n_chars", "starts_ok",
        "n_nums", "n_generic_positive", "sent_count", "has_named_entity",
        "rating", "src_len", "rev_sim_user", "rev_sim_ref", "cand_rank",
        "knn_bert_mean", "knn_bert_max", "knn_bert_std",
        "ce_user_biz", "ce_knn_refs",
    ]
    feat_ns = base_feats + [f"style={s}" for s in "ABCDEF"] + [f"model={m}" for m in args.models]
    print("\n  LambdaRank top-20 features:")
    imps = lgbm.feature_importances_
    for k in np.argsort(-imps)[:20]:
        print(f"    {feat_ns[k]:<30} {int(imps[k])}")

    print("\n=== Evaluate ===")
    from collections import Counter
    picks = {"ensemble": [], "lgbm": [], "xgb": [], "gbr": [],
              "proxy": [], "oracle": []}
    proxy_texts = [
        f"{tst_pack['biz_profiles'][m[0]]} {tst_pack['user_profiles'][m[0]]} {exemplar_ref}".strip()
        for m in meta_tst
    ]
    proxy_f1 = bert_f1_batch([m[4] for m in meta_tst], proxy_texts)
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


if __name__ == "__main__":
    main()
