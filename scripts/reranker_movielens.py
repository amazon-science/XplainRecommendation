"""MovieLens reranker: adapts the G-Refer style-F pipeline to MovieLens-1M.

Differences from the Google pipeline:
  - No free-text "reviews". The retrieval corpus IS the training set of
    Sonnet-generated reference explanations (data/movielens/train_sonnet45_refs.jsonl).
  - For each query (user_id, item_id), we retrieve neighbor training refs by:
      priority 1: same movie_id
      priority 2: genre-overlap + user age-group / gender match
      priority 3: text-embedding cosine on (movie + user) query text
    and use them as candidate contexts for prompts A / B / F.
  - Reference for scoring = the Sonnet-generated explanation for that sample.

Pipeline (identical to reranker_styleF.py in structure):
  style A  -- paraphrase a single neighbor ref
  style B  -- synthesize from top-k neighbor refs
  style F  -- length-tuned synthesis (target 28 words), 2 variants:
                F0 -- no movie title, F1 -- movie title allowed
  featurize (titan embeds + cross-encoder + kNN BERTScore) → LambdaRank +
  XGBRanker + pointwise GBR ensemble.

Usage:
  python3 scripts/reranker_movielens.py --n_train 1500 --n_test 400 \
      --n_single 4 --n_synth 4 --max_workers 10

Smaller (sanity) run:
  python3 scripts/reranker_movielens.py --n_train 200 --n_test 50 \
      --n_single 3 --n_synth 3 --max_workers 8
"""

import argparse, json, os, pickle, sys, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bedrock_llm import BedrockLLM
from scripts.rag_bandit_pipeline import (
    invoke_bedrock, bert_f1_batch, embed_texts_titan,
)
from scripts.reranker_final import (
    invoke_with_retry, ensemble_rank_scores,
    _count_numbers, _count_pos, _sent_count, _ne_hint,
)
from scripts.reranker_pairwise import _groups_from_meta


# DATA resolution: try FINAL_RESULT_DATA env var first (our bundle's data
# root), then sibling-of-scripts (original repo layout), then one more level
# up (scripts -> parent repo).
_env_root = os.environ.get("FINAL_RESULT_DATA")
if _env_root:
    DATA = Path(_env_root) / "data" / "movielens"
else:
    _sib = Path(__file__).resolve().parent.parent / "data" / "movielens"
    _par = Path(__file__).resolve().parent.parent.parent / "data" / "movielens"
    DATA = _sib if _sib.exists() else _par
REFS_JSONL = DATA / "train_sonnet45_refs.jsonl"
USER_PROF = DATA / "user_profile.json"
ITEM_PROF = DATA / "item_profile.json"


# ---------------------------- Data -------------------------------------------

@dataclass
class MLSample:
    uid: int           # node id (same as user_profile.json key)
    iid: int           # node id (same as item_profile.json key)
    reference: str     # sonnet explanation
    prompt: str        # reconstructed context string (movie + user)
    title: str
    genres: List[str]
    year: int
    age_group: str
    gender: str
    occupation: int


_OCC = {
    0: "not specified", 1: "academic/educator", 2: "artist",
    3: "clerical/admin", 4: "college/grad student", 5: "customer service",
    6: "doctor/healthcare", 7: "executive/managerial", 8: "farmer",
    9: "homemaker", 10: "K-12 student", 11: "lawyer", 12: "programmer",
    13: "retired", 14: "sales/marketing", 15: "scientist",
    16: "self-employed", 17: "technician/engineer", 18: "tradesman/craftsman",
    19: "unemployed", 20: "writer",
}


def _build_prompt(title, genres, year, age_group, gender, occupation):
    g = ", ".join(genres) if genres else "Unknown"
    occ = _OCC.get(int(occupation), "unspecified")
    gen = "female" if gender == "F" else "male"
    return (f"Movie title: {title}. "
            f"Genres: {g}. Year: {year}. "
            f"User: {age_group} {gen}, occupation: {occ}.")


def load_movielens_samples(seed=42):
    """Load all 2000 Sonnet-refs + profiles, split into train/test."""
    with open(USER_PROF) as f: up = json.load(f)
    with open(ITEM_PROF) as f: ip = json.load(f)

    samples: List[MLSample] = []
    with open(REFS_JSONL) as f:
        for line in f:
            d = json.loads(line)
            uid = d["user_id"]; iid = d["item_id"]
            u = up.get(str(uid), {})
            m = ip.get(str(iid), {})
            title = m.get("title", f"Movie#{iid}")
            genres = m.get("genres", [])
            year = m.get("year") or 2000
            ag = u.get("age_group", "25-34")
            gen = u.get("gender", "M")
            occ = u.get("occupation", 0)
            samples.append(MLSample(
                uid=int(uid), iid=int(iid),
                reference=d["explanation"].strip(),
                prompt=_build_prompt(title, genres, year, ag, gen, occ),
                title=title, genres=genres, year=int(year),
                age_group=ag, gender=gen, occupation=int(occ),
            ))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(samples))
    return [samples[i] for i in perm]


# ---------------------------- Retrieval --------------------------------------

def _genre_overlap(a, b):
    if not a or not b: return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def retrieve_neighbors(query, pool, pool_embs_norm, q_emb_norm, k,
                        exclude_self=True):
    """Priority: same iid → high genre+demo score → text-embedding cosine.
    Returns list of pool samples."""
    out, seen = [], set()
    if exclude_self:
        same = [p for p in pool if p.iid == query.iid
                and not (p.uid == query.uid and p.iid == query.iid)]
    else:
        same = [p for p in pool if p.iid == query.iid]
    for s in same:
        key = (s.uid, s.iid)
        if key in seen: continue
        out.append(s); seen.add(key)
        if len(out) >= k: return out

    # score all pool by genre overlap + demographic match + text cosine
    sims = pool_embs_norm @ q_emb_norm
    scores = np.array([
        0.6 * sims[j]
        + 0.3 * _genre_overlap(query.genres, p.genres)
        + 0.1 * (1.0 if p.age_group == query.age_group else 0.0)
        for j, p in enumerate(pool)
    ])
    order = np.argsort(-scores)
    for j in order:
        p = pool[int(j)]
        if exclude_self and p.uid == query.uid and p.iid == query.iid:
            continue
        key = (p.uid, p.iid)
        if key in seen: continue
        out.append(p); seen.add(key)
        if len(out) >= k: break
    return out


# ---------------------------- Prompts ----------------------------------------

def _opener_choices():
    return [
        "The user would enjoy this movie because of its",
        "The user would enjoy this movie for its",
        "You would enjoy this movie because of its",
    ]


def build_prompt_A(sample: MLSample, neighbor_ref: str):
    return (
        "You are writing a single-sentence movie recommendation explanation in "
        "a specific house style.\n"
        "RULES (strict):\n"
        "- EXACTLY ONE sentence, 25-33 words\n"
        "- Start with: \"The user would enjoy this movie because of its...\" "
        "or \"The user would enjoy this movie for its...\"\n"
        "- Do NOT name actors or directors unless they appear in the example\n"
        "- Do NOT invent plot details\n\n"
        f"Context:\n{sample.prompt}\n\n"
        f"Example explanation to paraphrase (keep same style, same specifics):\n"
        f"{neighbor_ref}\n\n"
        "Write the paraphrased explanation, output only the sentence:"
    )


def build_prompt_B(sample: MLSample, neighbor_refs: List[str]):
    block = "\n".join(f"- {r}" for r in neighbor_refs[:4])
    return (
        "You are writing a single-sentence movie recommendation explanation.\n"
        "RULES (strict):\n"
        "- EXACTLY ONE sentence, 25-33 words\n"
        "- Start with: \"The user would enjoy this movie because of its...\" "
        "or \"The user would enjoy this movie for its...\"\n"
        "- Synthesize the common themes across the examples\n"
        "- Do NOT name specific actors, directors, or invent plot details\n\n"
        f"Context:\n{sample.prompt}\n\n"
        f"Example explanations from similar users/movies:\n{block}\n\n"
        "Write the synthesized explanation, output only the sentence:"
    )


def build_prompt_G(sample: MLSample, word_target: int = 30):
    """World-knowledge: no neighbors, let LLM use its own knowledge of the movie.
    This unlocks specific plot/theme content that Sonnet refs have and that
    neighbor-paraphrase styles cannot produce."""
    return (
        "You are writing a single-sentence movie recommendation to a specific "
        "user, as if you personally know the film well.\n"
        "RULES (strict):\n"
        f"- EXACTLY ONE sentence, {word_target-4}-{word_target+6} words\n"
        "- Use your knowledge of the film to mention 1-2 specific things: "
        "a theme, tone, standout quality, era/style, or notable element "
        "(without naming actors unless iconic for the film)\n"
        "- Frame it as why THIS user would enjoy it\n"
        "- Start with a natural hook like \"You'll love\", \"If you enjoy\", "
        "\"This\", or \"Fans of\"\n"
        "- Do NOT invent facts; if unsure about the film, stay generic about "
        "its genre qualities\n\n"
        f"Movie: {sample.title}\n"
        f"Genres: {', '.join(sample.genres) if sample.genres else 'Unknown'}\n"
        f"Year: {sample.year}\n"
        f"User: {sample.age_group} {sample.gender}\n\n"
        "Write the explanation now, output only the sentence:"
    )


def build_prompt_H(sample: MLSample, neighbor_refs: List[str],
                    word_target: int = 30):
    """Hybrid: neighbors as style guide + encourage world knowledge for specifics."""
    block = "\n".join(f"- {r}" for r in neighbor_refs[:3])
    return (
        "You are writing a single-sentence movie recommendation. Match the "
        "style of the examples (tone, length, specificity) and use your own "
        "knowledge of the film to include concrete themes or qualities.\n"
        "RULES (strict):\n"
        f"- EXACTLY ONE sentence, {word_target-4}-{word_target+6} words\n"
        "- Match the voice and format of the example explanations\n"
        "- Mention 1-2 specific things about the film (theme, tone, era, "
        "standout quality) drawn from your knowledge; do NOT invent facts\n"
        "- Do NOT name actors unless iconic for the film\n\n"
        f"Movie: {sample.title} ({sample.year}) "
        f"— {', '.join(sample.genres) if sample.genres else ''}\n"
        f"User: {sample.age_group} {sample.gender}\n\n"
        f"Example explanations (style only, different movies):\n{block}\n\n"
        "Write the explanation now, output only the sentence:"
    )


def build_prompt_F(sample: MLSample, neighbor_refs: List[str], use_title: bool,
                    word_target: int = 28):
    block = "\n".join(f"- {r}" for r in neighbor_refs[:4])
    if use_title:
        opener = (f"\"You would enjoy {sample.title} because it...\" or "
                   f"\"You would enjoy {sample.title} for its...\"")
        name_rule = (f"- You may name the movie: \"{sample.title}\"\n"
                      "- Do NOT invent actors, directors, or plot details\n")
    else:
        opener = ("\"The user would enjoy this movie because of its...\" or "
                   "\"The user would enjoy this movie for its...\"")
        name_rule = ("- Refer to the movie as \"this movie\"\n"
                      "- Do NOT name actors, directors, or invent plot details\n")
    return (
        "You are writing a single-sentence movie recommendation explanation "
        "in a specific house style.\n"
        "RULES (strict):\n"
        f"- EXACTLY ONE sentence, {word_target-3}-{word_target+5} words\n"
        f"- Start with: {opener}\n"
        f"{name_rule}"
        "- Mention 2-3 generic qualities (tone, themes, pacing, atmosphere, "
        "visuals, humor, character dynamics, emotional arc, world-building) "
        "connected with commas and 'and'\n\n"
        f"Context:\n{sample.prompt}\n\n"
        f"Example explanations from similar users/movies:\n{block}\n\n"
        "Write the explanation now, output only the sentence:"
    )


# ---------------------------- Pool gen ---------------------------------------

def generate_pool(samples, pool_for_retrieval, pool_embs_norm,
                    q_embs_norm, models, clients, n_single, n_synth,
                    max_workers, k_neighbors=8):
    """Create candidate pool for every sample with neighbors precomputed."""
    neighbors = []
    for i, s in enumerate(samples):
        nbrs = retrieve_neighbors(s, pool_for_retrieval, pool_embs_norm,
                                    q_embs_norm[i], k=k_neighbors)
        neighbors.append(nbrs)

    jobs = []
    for m in models:
        for i, s in enumerate(samples):
            nbrs = neighbors[i]
            if not nbrs: continue
            for j in range(min(n_single, len(nbrs))):
                jobs.append(("A", m, i, j, nbrs[j].reference, None))
            jobs.append(("B", m, i, -1, None,
                          [n.reference for n in nbrs[:n_synth]]))
            jobs.append(("F", m, i, 0, None,
                          [n.reference for n in nbrs[:n_synth]]))
            jobs.append(("F", m, i, 1, None,
                          [n.reference for n in nbrs[:n_synth]]))
            # Style G: pure world-knowledge (no neighbors)
            jobs.append(("G", m, i, 0, None, None))
            jobs.append(("G", m, i, 1, None, None))  # two-shot diversity via temp
            # Style H: hybrid (neighbors-as-style + world knowledge)
            jobs.append(("H", m, i, 0, None,
                          [n.reference for n in nbrs[:n_synth]]))

    print(f"  Generating {len(jobs)} candidates...")
    preds: Dict[Tuple, str] = {}

    def gen(job):
        style, m, i, j, neighbor_ref, ctx = job
        s = samples[i]
        if style == "A":
            msg = build_prompt_A(s, neighbor_ref)
            temp = 0.0
        elif style == "B":
            msg = build_prompt_B(s, ctx)
            temp = 0.0
        elif style == "F":
            msg = build_prompt_F(s, ctx, use_title=(j == 1))
            temp = 0.0
        elif style == "G":
            msg = build_prompt_G(s)
            # j==1 uses sampling for diversity
            temp = 0.7 if j == 1 else 0.0
        else:  # H
            msg = build_prompt_H(s, ctx)
            temp = 0.0
        out = invoke_with_retry(clients[m], msg, temperature=temp)
        return style, m, i, j, out or ""

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(gen, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="  gen"):
            style, m, i, j, out = fut.result()
            preds[(style, m, i, j)] = out

    return {
        "samples": samples,
        "neighbors": neighbors,
        "preds": preds,
        "n_single": n_single,
        "n_synth": n_synth,
        "models": list(models),
    }


# ---------------------------- Features ---------------------------------------

def featurize(pack, embed_llm, ref_centroid, exemplar_ref, trn_ref_embs_norm,
               cache_path, max_workers, knn_refs_per_sample, cross_encoder,
               models):
    samples = pack["samples"]
    preds = pack["preds"]
    neighbors = pack["neighbors"]
    n_single = pack["n_single"]

    user_profiles = [f"{s.age_group} {s.gender}, {_OCC.get(s.occupation, '')}"
                      for s in samples]
    biz_profiles = [f"{s.title} {' '.join(s.genres)}" for s in samples]

    pred_texts = [v for v in preds.values() if v.strip()]
    to_embed = list(set(pred_texts + biz_profiles + user_profiles +
                         [exemplar_ref]))
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
        if not neighbors[i]: continue
        for m in models:
            for j in range(min(n_single, len(neighbors[i]))):
                k = ("A", m, i, j)
                if preds.get(k, "").strip(): candidate_keys.append(k)
            for style, j in [("B", -1), ("F", 0), ("F", 1),
                              ("G", 0), ("G", 1), ("H", 0)]:
                k = (style, m, i, j)
                if preds.get(k, "").strip(): candidate_keys.append(k)

    print(f"  kNN BERTScore pairs...")
    bert_pairs, pair_back = [], {}
    for key in candidate_keys:
        (style, m, i, j) = key
        pred = preds[key]
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
            a = np.array(scores)
            knn_stats[key] = (float(a.mean()), float(a.max()), float(a.std()))
        else:
            knn_stats[key] = (0.0, 0.0, 0.0)

    print(f"  Cross-encoder scoring...")
    up_targets = [f"{user_profiles[i]} {biz_profiles[i]}"
                   for i in range(len(samples))]
    ce_pairs_up, ce_pairs_knn = [], []
    ce_up_back, ce_knn_back = {}, {}
    for key in candidate_keys:
        (style, m, i, j) = key
        pred = preds[key]
        ce_up_back[key] = len(ce_pairs_up)
        ce_pairs_up.append((pred, up_targets[i]))
        knn_concat = " ".join(knn_refs_per_sample[i][:3])
        ce_knn_back[key] = len(ce_pairs_knn)
        ce_pairs_knn.append((pred, knn_concat))
    ce_up = cross_encoder.predict(ce_pairs_up, show_progress_bar=True,
                                    batch_size=64)
    ce_kn = cross_encoder.predict(ce_pairs_knn, show_progress_bar=True,
                                    batch_size=64)

    STYLE = ["A", "B", "F", "G", "H"]

    def row(pred, p_vec, u_vec, b_vec, rating, src_len, rev_su, rev_sr,
             style, m, knn, ce_u, ce_k, cand_rank):
        cos_user = float(p_vec @ u_vec) if u_vec is not None else 0.0
        cos_biz = float(p_vec @ b_vec) if b_vec is not None else 0.0
        cos_ref = float(p_vec @ ref_centroid)
        cos_exe = float(p_vec @ exemplar_vec)
        sims = trn_ref_embs_norm @ p_vec
        top5 = float(np.sort(sims)[-5:].mean())
        feats = [
            cos_user, cos_biz, cos_ref, cos_exe, top5,
            float(len(pred.split())), float(len(pred)),
            1.0 if pred.lstrip().startswith("The user would enjoy") or
                   pred.lstrip().startswith("You would enjoy") else 0.0,
            float(_count_numbers(pred)), float(_count_pos(pred)),
            float(_sent_count(pred)), float(_ne_hint(pred)),
            rating, src_len, rev_su, rev_sr, float(cand_rank),
            knn[0], knn[1], knn[2], ce_u, ce_k,
        ]
        feats += [1.0 if style == s_ else 0.0 for s_ in STYLE]
        feats += [1.0 if m == mm else 0.0 for mm in models]
        return feats

    X, meta = [], []
    for i, s in enumerate(samples):
        if not neighbors[i]: continue
        u_vec = embn(user_profiles[i])
        b_vec = embn(biz_profiles[i])
        nbrs = neighbors[i]
        used = nbrs[:pack["n_synth"]]
        used_srclen = float(np.mean([len(n.reference.split())
                                      for n in used])) if used else 0.0
        for m in models:
            for j in range(min(n_single, len(nbrs))):
                k = ("A", m, i, j)
                pred = preds.get(k, "")
                if not pred.strip(): continue
                p_vec = embn(pred)
                rv_su = 0.0; rv_sr = 0.0  # per-neighbor sim not computed here
                X.append(row(pred, p_vec, u_vec, b_vec, 3.0,
                              float(len(nbrs[j].reference.split())),
                              rv_su, rv_sr, "A", m, knn_stats[k],
                              float(ce_up[ce_up_back[k]]),
                              float(ce_kn[ce_knn_back[k]]), j))
                meta.append((i, m, "A", j, pred, s.reference))

            def add_styled(style, j):
                k = (style, m, i, j)
                pred = preds.get(k, "")
                if not pred.strip(): return
                p_vec = embn(pred)
                X.append(row(pred, p_vec, u_vec, b_vec, 3.0, used_srclen,
                              0.0, 0.0, style, m, knn_stats[k],
                              float(ce_up[ce_up_back[k]]),
                              float(ce_kn[ce_knn_back[k]]), 0))
                meta.append((i, m, style, j, pred, s.reference))

            add_styled("B", -1)
            add_styled("F", 0)
            add_styled("F", 1)
            add_styled("G", 0)
            add_styled("G", 1)
            add_styled("H", 0)
    return np.array(X), meta


# ---------------------------- kNN for features -------------------------------

def knn_refs_for_samples(queries, pool, q_embs_n, pool_embs_n, k,
                           exclude_self=True):
    sims = q_embs_n @ pool_embs_n.T
    out = []
    for i, q in enumerate(queries):
        order = np.argsort(-sims[i])
        picks = []
        for j in order:
            p = pool[int(j)]
            if exclude_self and p.uid == q.uid and p.iid == q.iid:
                continue
            picks.append(p.reference)
            if len(picks) >= k: break
        out.append(picks)
    return out


# ---------------------------- Main -------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=1500)
    ap.add_argument("--n_test", type=int, default=400)
    ap.add_argument("--n_single", type=int, default=4)
    ap.add_argument("--n_synth", type=int, default=4)
    ap.add_argument("--k_neighbors", type=int, default=5)
    ap.add_argument("--max_workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_dir", type=str,
                     default="results/rag_bandit/pool_cache_ml")
    ap.add_argument("--models", type=str, nargs="+", default=[
        "us.amazon.nova-lite-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
    ])
    args = ap.parse_args()

    print(f"=== MovieLens reranker (no-tune pipeline) ===")
    all_samples = load_movielens_samples(seed=args.seed)
    if args.n_train + args.n_test > len(all_samples):
        print(f"  WARNING: only {len(all_samples)} Sonnet refs available; "
               f"trimming test.")
        args.n_test = max(1, len(all_samples) - args.n_train)
    trn_eval = all_samples[: args.n_train]
    tst_eval = all_samples[args.n_train: args.n_train + args.n_test]
    print(f"  train={len(trn_eval)}  test={len(tst_eval)}")

    cache_path = Path("results/rag_bandit/titan_cache_movielens.json")
    cache_dir = Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)

    clients = {m: BedrockLLM(model_id=m, max_tokens=300, temperature=0.0)
                for m in args.models}
    embed_llm = clients[args.models[0]]

    # Precompute embeddings:
    print("\nEmbedding prompts + refs (for retrieval)...")
    prompt_texts = [s.prompt for s in all_samples]
    ref_texts = [s.reference for s in all_samples]
    prompt_embs = embed_texts_titan(embed_llm, prompt_texts,
                                      cache_path=cache_path,
                                      max_workers=args.max_workers)
    ref_embs = embed_texts_titan(embed_llm, ref_texts,
                                   cache_path=cache_path,
                                   max_workers=args.max_workers)
    def _norm(M):
        return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    prompt_embs_n = _norm(prompt_embs)
    ref_embs_n = _norm(ref_embs)

    # Train pool for retrieval = trn_eval only (so test leakage is impossible)
    trn_prompt_embs_n = prompt_embs_n[: args.n_train]
    trn_ref_embs_n = ref_embs_n[: args.n_train]

    # Centroid + exemplar computed from training refs
    centroid = trn_ref_embs_n.mean(axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-9)
    exemplar_ref = trn_eval[int(np.argmax(trn_ref_embs_n @ centroid))].reference

    # Pool caches (keyed by model set):
    mtag = "_".join(m.split(":")[0].split(".")[-1][:10] for m in args.models)
    trn_cache = cache_dir / f"poolA_tr{len(trn_eval)}_te{len(tst_eval)}_{mtag}.pkl"
    tst_cache = cache_dir / f"poolB_tr{len(trn_eval)}_te{len(tst_eval)}_{mtag}.pkl"

    print("\n=== Generate / load TRAIN pool ===")
    if trn_cache.exists():
        with open(trn_cache, "rb") as f: trn_pack = pickle.load(f)
        print(f"  Loaded cached → {trn_cache}")
    else:
        trn_pack = generate_pool(trn_eval, trn_eval, trn_prompt_embs_n,
                                   trn_prompt_embs_n, args.models, clients,
                                   args.n_single, args.n_synth,
                                   args.max_workers, k_neighbors=8)
        with open(trn_cache, "wb") as f: pickle.dump(trn_pack, f)

    print("\n=== Generate / load TEST pool ===")
    if tst_cache.exists():
        with open(tst_cache, "rb") as f: tst_pack = pickle.load(f)
        print(f"  Loaded cached → {tst_cache}")
    else:
        tst_prompt_embs_n = prompt_embs_n[args.n_train: args.n_train + args.n_test]
        tst_pack = generate_pool(tst_eval, trn_eval, trn_prompt_embs_n,
                                   tst_prompt_embs_n, args.models, clients,
                                   args.n_single, args.n_synth,
                                   args.max_workers, k_neighbors=8)
        with open(tst_cache, "wb") as f: pickle.dump(tst_pack, f)

    print("\n=== kNN + cross-encoder ===")
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",
                                   max_length=512)
    tst_prompt_embs_n = prompt_embs_n[args.n_train: args.n_train + args.n_test]
    trn_knn = knn_refs_for_samples(trn_eval, trn_eval, trn_prompt_embs_n,
                                     trn_prompt_embs_n, args.k_neighbors)
    tst_knn = knn_refs_for_samples(tst_eval, trn_eval, tst_prompt_embs_n,
                                     trn_prompt_embs_n, args.k_neighbors,
                                     exclude_self=False)

    print("\n=== Featurize ===")
    print("Train:")
    X_trn, meta_trn = featurize(trn_pack, embed_llm, centroid, exemplar_ref,
                                  trn_ref_embs_n, cache_path, args.max_workers,
                                  trn_knn, cross_encoder, args.models)
    print("Test:")
    X_tst, meta_tst = featurize(tst_pack, embed_llm, centroid, exemplar_ref,
                                  trn_ref_embs_n, cache_path, args.max_workers,
                                  tst_knn, cross_encoder, args.models)
    print(f"  X_train: {X_trn.shape}  X_test: {X_tst.shape}")

    print("\n=== True-F1 labels ===")
    y_trn = bert_f1_batch([m[4] for m in meta_trn], [m[5] for m in meta_trn])
    y_tst = bert_f1_batch([m[4] for m in meta_tst], [m[5] for m in meta_tst])
    grp_trn, by_sample_trn = _groups_from_meta(meta_trn)
    grp_tst, by_sample_tst = _groups_from_meta(meta_tst)
    oracle_trn = np.array([max(y_trn[list(ks)])
                            for ks in by_sample_trn.values()])
    oracle_tst = np.array([max(y_tst[list(ks)])
                            for ks in by_sample_tst.values()])
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
    for s_ in ("A", "B", "F", "G", "H"):
        per = []
        for i, ks in by_sample_tst.items():
            sub = [k for k in ks if meta_tst[k][2] == s_]
            if sub: per.append(max(y_tst[sub]))
        if per:
            print(f"    only {s_}: {np.mean(per):.4f}  (n={len(per)})")

    print("\n=== Fit ensemble ===")
    s_lgbm, s_xgb, s_gbr, lgbm = ensemble_rank_scores(
        X_trn, y_trn, grp_trn, X_tst, args.seed)

    def pct(scores, grp):
        out = np.zeros_like(scores); idx = 0
        for g in grp:
            blk = scores[idx:idx+g]; order = np.argsort(blk)
            p = np.zeros_like(blk); p[order] = np.linspace(0, 1, g)
            out[idx:idx+g] = p; idx += g
        return out
    s_ens = (pct(s_lgbm, grp_tst) + pct(s_xgb, grp_tst) + pct(s_gbr, grp_tst))/3

    base_feats = [
        "cos_user", "cos_biz", "cos_ref_centroid", "cos_exemplar",
        "top5_nearest_trn_ref", "n_words", "n_chars", "starts_ok",
        "n_nums", "n_generic_positive", "sent_count", "has_named_entity",
        "rating", "src_len", "rev_sim_user", "rev_sim_ref", "cand_rank",
        "knn_bert_mean", "knn_bert_max", "knn_bert_std",
        "ce_user_biz", "ce_knn_refs",
    ]
    feat_ns = base_feats + [f"style={s}" for s in "ABFGH"] + \
               [f"model={m}" for m in args.models]
    print("\n  LambdaRank top-15 features:")
    imps = lgbm.feature_importances_
    for k in np.argsort(-imps)[:15]:
        print(f"    {feat_ns[k]:<32} {int(imps[k])}")

    print("\n=== Evaluate ===")
    from collections import Counter
    picks = {"ensemble": [], "lgbm": [], "xgb": [], "gbr": [], "oracle": []}
    pb = Counter()
    for i, ks in by_sample_tst.items():
        ks = list(ks)
        k_ens = ks[int(np.argmax(s_ens[ks]))]
        picks["ensemble"].append(y_tst[k_ens])
        picks["lgbm"].append(y_tst[ks[int(np.argmax(s_lgbm[ks]))]])
        picks["xgb"].append(y_tst[ks[int(np.argmax(s_xgb[ks]))]])
        picks["gbr"].append(y_tst[ks[int(np.argmax(s_gbr[ks]))]])
        picks["oracle"].append(y_tst[ks[int(np.argmax(y_tst[ks]))]])
        pb[(meta_tst[k_ens][1], meta_tst[k_ens][2])] += 1

    def summ(name, vals):
        a = np.array(vals)
        print(f"  {name:<12}: {a.mean():.4f} \u00b1 {a.std():.4f}  "
              f"%>=0.50 {100*(a>=0.5).mean():.1f}%  "
              f"%>=0.55 {100*(a>=0.55).mean():.1f}%  "
              f"%>=0.60 {100*(a>=0.6).mean():.1f}%")
    n = len(picks["ensemble"])
    print(f"\n  === MovieLens Test F1 (n={n}) ===")
    summ("ENSEMBLE",    picks["ensemble"])
    summ("LambdaRank",  picks["lgbm"])
    summ("XGBRanker",   picks["xgb"])
    summ("PointwiseGBR", picks["gbr"])
    summ("ORACLE",      picks["oracle"])
    print(f"\n  Ensemble picks (model, style): {dict(pb)}")

    out_file = Path("results/rag_bandit") / (
        f"reranker_movielens_n{len(trn_eval)}tr_n{len(tst_eval)}te.json")
    pw = np.array(picks["ensemble"])
    with open(out_file, "w") as f:
        json.dump({
            "dataset": "movielens-1m-sonnet45-refs",
            "n_train": len(trn_eval), "n_test": len(tst_eval),
            "ensemble_f1": float(pw.mean()),
            "lgbm_f1": float(np.mean(picks["lgbm"])),
            "xgb_f1": float(np.mean(picks["xgb"])),
            "gbr_f1": float(np.mean(picks["gbr"])),
            "oracle_f1": float(np.mean(picks["oracle"])),
            "train_oracle": float(oracle_trn.mean()),
            "pick_breakdown": {f"{k[0]}__{k[1]}": v for k, v in pb.items()},
        }, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
