# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Retrieval-Augmented + Bandit-Reranked pipeline for BERTScore improvement.

Three stages combined:
  Stage 1: Use G-Refer's rich source_data.prompt as the LLM prompt (profiles included).
  Stage 2: k-NN retrieval over training references; inject top-K as few-shot demonstrations.
  Stage 3: Best-of-N generation + contextual bandit over prompt templates,
           ranked by proxy BERTScore against retrieved neighbor references.

No model fine-tuning. Pure test-time adaptation. BERTScore target = 0.60 on Google tst.

Usage:
  # Stage 1 only pilot:
  python scripts/rag_bandit_pipeline.py --stage 1 --num_samples 50

  # Stage 2 only pilot:
  python scripts/rag_bandit_pipeline.py --stage 2 --num_samples 50 --k_neighbors 5

  # Stage 3 full (best-of-N + bandit rerank):
  python scripts/rag_bandit_pipeline.py --stage 3 --num_samples 200 --n_candidates 6

  # Train bandit on trn split (optional, for learned template selection):
  python scripts/rag_bandit_pipeline.py --stage 3 --train_bandit --bandit_samples 400
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluate as hf_evaluate
from tqdm import tqdm

from src.bedrock_llm import BedrockLLM


# ---------- Data loading helpers ---------------------------------------------


REPO_ROOT = Path(__file__).resolve().parent.parent
GREFER_DATA = REPO_ROOT / "G-Refer" / "data" / "google"
GREFER_PRED = REPO_ROOT / "G-Refer" / "gen_explanations" / "G-Refer" / "google_pred.jsonl"


@dataclass
class Sample:
    uid: int
    iid: int
    prompt: str          # Full rich prompt from source_data.prompt
    reference: str       # Ground truth (source_data.chosen) stripped of '### '
    grefer_output: str   # Original G-Refer generation, stripped of '### '
    split: str = ""      # 'trn' or 'tst'


def strip_marker(text: str) -> str:
    text = (text or "").strip()
    if "### " in text:
        return text.split("### ", 1)[1].strip()
    return text


def load_grefer_samples() -> Tuple[List[Sample], List[Sample]]:
    """Load all 3000 pred samples, split into train (2400) and test (600)
    based on presence in total_trn.csv / total_tst.csv (via node->raw mapping)."""
    import csv

    # Build raw_id -> node_id maps
    with open(GREFER_DATA / "metadata.json") as f:
        md = json.load(f)
    u2n = {int(k): v for k, v in md["user_id_to_node"].items()}
    i2n = {int(k): v for k, v in md["item_id_to_node"].items()}
    n2u = {v: k for k, v in u2n.items()}
    n2i = {v: k for k, v in i2n.items()}

    # Build tst/trn raw-pair sets from CSVs (CSVs use node IDs).
    def load_pairs_from_csv(path: Path) -> set:
        pairs = set()
        with open(path) as f:
            for r in csv.DictReader(f):
                un, inn = int(r["user_id"]), int(r["item_id"])
                if un in n2u and inn in n2i:
                    pairs.add((n2u[un], n2i[inn]))
        return pairs

    tst_pairs = load_pairs_from_csv(GREFER_DATA / "total_tst.csv")
    trn_pairs = load_pairs_from_csv(GREFER_DATA / "total_trn.csv")

    trn_samples, tst_samples = [], []
    with open(GREFER_PRED) as f:
        for line in f:
            d = json.loads(line)
            sd = d["source_data"]
            uid, iid = sd["uid"], sd["iid"]
            sample = Sample(
                uid=uid,
                iid=iid,
                prompt=sd["prompt"],
                reference=strip_marker(sd["chosen"]),
                grefer_output=strip_marker(d["output_str"]),
            )
            if (uid, iid) in tst_pairs:
                sample.split = "tst"
                tst_samples.append(sample)
            elif (uid, iid) in trn_pairs:
                sample.split = "trn"
                trn_samples.append(sample)

    print(f"Loaded {len(trn_samples)} trn / {len(tst_samples)} tst samples")
    return trn_samples, tst_samples


def load_node_embeddings() -> Tuple[np.ndarray, Dict[int, int], Dict[int, int]]:
    """Return (embeddings [N,768], raw_uid->node, raw_iid->node).

    NOTE: These pre-trained graph embeddings are degenerate in this dataset
    (effectively constant vectors). Prefer text_embeddings() + a TextRetriever.
    """
    d = torch.load(GREFER_DATA / "data_tst.pt", weights_only=False)
    emb = d.x.numpy().astype(np.float32)
    u2n = {int(k): v for k, v in d.user_id_to_node.items()}
    i2n = {int(k): v for k, v in d.item_id_to_node.items()}
    return emb, u2n, i2n


# ---------- Text embeddings via Bedrock Titan ---------------------------------


def embed_texts_titan(llm: BedrockLLM, texts: List[str],
                      model_id: str = "amazon.titan-embed-text-v2:0",
                      max_workers: int = 16,
                      cache_path: Optional[Path] = None) -> np.ndarray:
    """Embed a list of texts using Amazon Titan embeddings on Bedrock.

    Returns np.ndarray shape [N, 1024]. Uses an optional on-disk cache so
    re-runs avoid re-embedding.
    """
    cache: Dict[str, List[float]] = {}
    if cache_path is not None and cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)

    need = [t for t in texts if t not in cache]
    if need:
        print(f"  Embedding {len(need)} texts via Titan ({len(cache)} cached)...")

        def call(t: str):
            resp = llm.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps({"inputText": t}),
            )
            body = json.loads(resp["body"].read())
            return t, body["embedding"]

        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(call, t) for t in need]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Titan-embed"):
                t, e = fut.result()
                cache[t] = e
                done += 1

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(cache, f)

    M = np.stack([np.asarray(cache[t], dtype=np.float32) for t in texts])
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return M / norms


class TextRetriever:
    """Retrieve training samples by text-embedding similarity of the
    G-Refer prompt (which contains business title + profile + user profile).

    Also keeps the by_iid index for same-item retrieval priority.
    """

    def __init__(self, trn_samples: List[Sample], trn_embs: np.ndarray):
        assert len(trn_samples) == len(trn_embs)
        self.trn = trn_samples
        self.M = trn_embs  # already L2-normalized
        self.by_iid: Dict[int, List[int]] = {}
        for idx, s in enumerate(trn_samples):
            self.by_iid.setdefault(s.iid, []).append(idx)
        print(f"  TextRetriever ready: {len(trn_samples)} trn, "
              f"dim={trn_embs.shape[1]}, unique_items={len(self.by_iid)}")

    def topk(self, query_emb: np.ndarray, iid: Optional[int] = None,
             uid: Optional[int] = None, k: int = 5) -> List[Sample]:
        # Priority 1: same-item matches (if any)
        out: List[Sample] = []
        seen = set()
        if iid is not None:
            for idx in self.by_iid.get(iid, []):
                s = self.trn[idx]
                if s.uid == uid and s.iid == iid:
                    continue
                key = (s.uid, s.iid)
                if key in seen:
                    continue
                out.append(s)
                seen.add(key)
                if len(out) >= k:
                    return out

        # Priority 2: text-embedding cosine
        sims = self.M @ query_emb
        order = np.argsort(-sims)
        for j in order[:k + 20]:
            s = self.trn[j]
            if s.uid == uid and s.iid == iid:
                continue
            key = (s.uid, s.iid)
            if key in seen:
                continue
            out.append(s)
            seen.add(key)
            if len(out) >= k:
                break
        return out


# ---------- Retrieval (Stage 2) ----------------------------------------------


class NeighborRetriever:
    """Retrieve training samples similar to a test (uid,iid).

    Two retrieval paths:
      1. Same-item retrieval: training samples with matching iid (actual
         reviews of the same business). These reference the same entity.
      2. Embedding retrieval: cosine on concat([user_emb, item_emb]).

    topk() interleaves same-item matches first, then fills with embedding
    neighbors, avoiding duplicates.
    """

    def __init__(self, trn_samples: List[Sample], emb: np.ndarray,
                 u2n: Dict[int, int], i2n: Dict[int, int]):
        self.trn = trn_samples
        self.emb = emb
        self.u2n = u2n
        self.i2n = i2n
        # Embedding matrix
        trn_vecs = []
        valid_idx = []
        for idx, s in enumerate(trn_samples):
            un = u2n.get(s.uid)
            inn = i2n.get(s.iid)
            if un is None or inn is None:
                continue
            trn_vecs.append(np.concatenate([emb[un], emb[inn]]))
            valid_idx.append(idx)
        M = np.stack(trn_vecs).astype(np.float32)
        norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
        self.M = M / norms
        self.valid_idx = valid_idx
        # Item-id index: iid -> list of trn Sample
        self.by_iid: Dict[int, List[Sample]] = {}
        for s in trn_samples:
            self.by_iid.setdefault(s.iid, []).append(s)
        print(f"  Retriever ready: {len(valid_idx)} indexable trn samples, "
              f"dim={M.shape[1]}, unique_items={len(self.by_iid)}")

    def topk(self, uid: int, iid: int, k: int = 5) -> List[Sample]:
        un = self.u2n.get(uid)
        inn = self.i2n.get(iid)
        out: List[Sample] = []
        seen = set()

        # Priority 1: same-item training references (real reviews of this business)
        for s in self.by_iid.get(iid, []):
            if s.uid == uid and s.iid == iid:
                continue  # don't leak
            key = (s.uid, s.iid)
            if key in seen:
                continue
            out.append(s)
            seen.add(key)
            if len(out) >= k:
                return out

        # Priority 2: embedding similarity
        if un is not None and inn is not None:
            q = np.concatenate([self.emb[un], self.emb[inn]]).astype(np.float32)
            q = q / (np.linalg.norm(q) + 1e-9)
            sims = self.M @ q
            order = np.argsort(-sims)
            for j in order[:k + 20]:
                s = self.trn[self.valid_idx[j]]
                if s.uid == uid and s.iid == iid:
                    continue
                key = (s.uid, s.iid)
                if key in seen:
                    continue
                out.append(s)
                seen.add(key)
                if len(out) >= k:
                    break
        return out

    def same_item_refs(self, iid: int, exclude_uid: Optional[int] = None) -> List[Sample]:
        return [s for s in self.by_iid.get(iid, [])
                if exclude_uid is None or s.uid != exclude_uid]


# ---------- Prompt construction ----------------------------------------------


STYLE_MIMIC = (
    "Imitate the exact sentence structure and vocabulary of the examples: "
    "'The user would enjoy the business because of the X, Y, and Z'. "
    "Use generic review attributes (delicious food, friendly service, great "
    "atmosphere, fair prices, fresh ingredients, good selection, cozy ambiance). "
    "Keep it 20–30 words."
)

TEMPLATES = [
    # Template 0: Plain — just the G-Refer prompt, no few-shot
    {"name": "plain", "temperature": 0.0, "k_shot": 0, "style_note": STYLE_MIMIC},
    # Template 1: +2 few-shot
    {"name": "fewshot2", "temperature": 0.0, "k_shot": 2, "style_note": STYLE_MIMIC},
    # Template 2: +3 few-shot
    {"name": "fewshot3", "temperature": 0.0, "k_shot": 3, "style_note": STYLE_MIMIC},
    # Template 3: +5 few-shot
    {"name": "fewshot5", "temperature": 0.0, "k_shot": 5, "style_note": STYLE_MIMIC},
    # Template 4: +8 few-shot, more aggressive style
    {"name": "fewshot8", "temperature": 0.0, "k_shot": 8, "style_note": STYLE_MIMIC},
    # Template 5: +5 few-shot, temperature 0.4 (exploration)
    {"name": "fewshot5_t4", "temperature": 0.4, "k_shot": 5,
     "style_note": STYLE_MIMIC},
    # Template 6: +3 few-shot, temperature 0.3
    {"name": "fewshot3_t3", "temperature": 0.3, "k_shot": 3,
     "style_note": STYLE_MIMIC},
    # Template 7: +5 few-shot with strong "generic attributes" emphasis
    {"name": "fewshot5_generic", "temperature": 0.0, "k_shot": 5,
     "style_note": STYLE_MIMIC +
     " Do NOT mention specific dish names, brand names, business names, or locations. "
     "Use only generic attribute words."},
]


SYSTEM_PROMPT = (
    "You write short recommendation explanations that match a specific house style. "
    "RULES:\n"
    "1. Start with exactly 'The user would enjoy the business because'.\n"
    "2. Refer to the place as 'the business' — do NOT use the business name.\n"
    "3. Length: 20–30 words total.\n"
    "4. Follow a list pattern: 'because of the X, Y, and Z' (2-3 generic review "
    "attributes like delicious food, friendly service, great atmosphere, fair prices, "
    "fresh ingredients).\n"
    "5. No markdown, no bullets, no headers, no business names, no dish names unless "
    "they appear in the example explanations.\n"
    "6. One sentence only, output the explanation only, nothing else."
)


def build_prompt(sample: Sample, neighbors: List[Sample], template: dict) -> str:
    """Assemble the user message for the LLM.

    - sample.prompt is the full G-Refer prompt (rich context).
    - neighbors are training examples for few-shot.
    - template chooses how many neighbors and whether to add a style note.
    """
    k = template.get("k_shot", 0)
    style_note = template.get("style_note", "")

    blocks = []

    if k > 0 and neighbors:
        blocks.append(
            "Here are example explanations for similar user/business recommendations. "
            "Notice the review-grounded vocabulary (specific dishes, flavors, "
            "atmosphere words). Match this style.\n"
        )
        for i, n in enumerate(neighbors[:k]):
            # Trim neighbor prompt to just the profile summary (first sentence)
            # to keep context short.
            blocks.append(f"Example {i+1}:")
            blocks.append(f"  Context: {_short_context(n.prompt)}")
            blocks.append(f"  Explanation: {n.reference}")
            blocks.append("")
        blocks.append("---\n")

    # Then the actual target context and instruction
    blocks.append("Now write an explanation for the following:\n")
    blocks.append(sample.prompt)

    if style_note:
        blocks.append("\n" + style_note)

    blocks.append(
        "\nWrite ONE explanation, 20–30 words, starting with exactly "
        "'The user would enjoy the business because'. Refer to the place only "
        "as 'the business'. Do not name it. Output nothing but the explanation."
    )

    return "\n".join(blocks)


def _short_context(grefer_prompt: str) -> str:
    """Extract a compact context string from a G-Refer prompt for few-shot."""
    # Take everything up to the first '###' and compress whitespace.
    head = grefer_prompt.split("###", 1)[0].strip()
    return " ".join(head.split())


# ---------- LLM invocation (custom, since BedrockLLM wrapper is too rigid) ---


def invoke_bedrock(llm: BedrockLLM, user_message: str,
                   temperature: float = 0.0, max_tokens: int = 200) -> str:
    """Direct Bedrock invoke using llm's runtime client, with our system prompt.

    Supports both Anthropic (Claude) and Amazon Nova request schemas; branches
    on model_id.
    """
    mid = llm.model_id.lower()
    is_nova = "nova" in mid

    if is_nova:
        req = {
            "schemaVersion": "messages-v1",
            "messages": [{"role": "user",
                          "content": [{"text": user_message}]}],
            "system": [{"text": SYSTEM_PROMPT}],
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
    else:
        req = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}],
        }
    try:
        resp = llm.bedrock_runtime.invoke_model(
            modelId=llm.model_id, body=json.dumps(req)
        )
        body = json.loads(resp["body"].read())
        if is_nova:
            text = body["output"]["message"]["content"][0]["text"].strip()
        else:
            text = body["content"][0]["text"].strip()
        # Strip leading '### ' if model included it
        return strip_marker(text)
    except Exception as e:
        print(f"  Bedrock error: {e}")
        return ""


# ---------- BERTScore utilities -----------------------------------------------


_bertscore_cache = {"obj": None}


def get_bertscore():
    if _bertscore_cache["obj"] is None:
        _bertscore_cache["obj"] = hf_evaluate.load("bertscore")
    return _bertscore_cache["obj"]


def bert_f1_batch(preds: List[str], refs: List[str]) -> np.ndarray:
    bs = get_bertscore()
    res = bs.compute(predictions=preds, references=refs,
                     lang="en", rescale_with_baseline=True)
    return np.array(res["f1"])


# ---------- Stages ------------------------------------------------------------


def _retriever_topk(sample: Sample, retriever, k: int,
                    test_embs: Optional[np.ndarray] = None,
                    test_idx: Optional[int] = None) -> List[Sample]:
    """Uniform interface over NeighborRetriever (graph-emb) and TextRetriever."""
    if isinstance(retriever, TextRetriever):
        q = test_embs[test_idx]
        return retriever.topk(q, iid=sample.iid, uid=sample.uid, k=k)
    else:
        return retriever.topk(sample.uid, sample.iid, k=k)


def run_stage1(samples: List[Sample], llm: BedrockLLM,
               max_workers: int = 10) -> Tuple[List[str], List[str]]:
    """Stage 1: feed G-Refer's rich prompt directly to Haiku, no retrieval.
    Baseline for upper bound of prompt-only approach.
    """
    preds = [None] * len(samples)

    def work(i: int):
        tmpl = TEMPLATES[0]  # plain
        user_msg = build_prompt(samples[i], [], tmpl)
        out = invoke_bedrock(llm, user_msg, temperature=tmpl["temperature"])
        return i, out

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(work, i) for i in range(len(samples))]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Stage1"):
            i, out = fut.result()
            preds[i] = out

    refs = [s.reference for s in samples]
    return preds, refs


def run_stage2(samples: List[Sample], llm: BedrockLLM,
               retriever, k_neighbors: int = 5,
               template_idx: int = 3, max_workers: int = 10,
               test_embs: Optional[np.ndarray] = None,
               ) -> Tuple[List[str], List[str]]:
    """Stage 2: few-shot k-NN retrieval, single generation per sample."""
    preds = [None] * len(samples)
    tmpl = TEMPLATES[template_idx]

    def work(i: int):
        nbrs = _retriever_topk(samples[i], retriever, k_neighbors,
                               test_embs=test_embs, test_idx=i)
        user_msg = build_prompt(samples[i], nbrs, tmpl)
        out = invoke_bedrock(llm, user_msg, temperature=tmpl["temperature"])
        return i, out

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(work, i) for i in range(len(samples))]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Stage2"):
            i, out = fut.result()
            preds[i] = out

    refs = [s.reference for s in samples]
    return preds, refs


def run_stage3(samples: List[Sample], llm: BedrockLLM,
               retriever, k_neighbors: int = 5,
               n_candidates: int = 6, max_workers: int = 10,
               bandit: Optional["ContextualBandit"] = None,
               template_indices: Optional[List[int]] = None,
               test_embs: Optional[np.ndarray] = None,
               ) -> Tuple[List[str], List[str], List[dict]]:
    """Stage 3: for each sample, generate N candidates across templates,
    rerank via proxy BERTScore vs retrieved neighbor refs, pick best.

    Two-phase: (1) parallel LLM generation; (2) single BERTScore pass for
    proxy scoring (HF evaluate module is not thread-safe).
    """
    if template_indices is None:
        template_indices = list(range(len(TEMPLATES)))[:n_candidates]
    refs = [s.reference for s in samples]
    N = len(samples)

    # ---- Phase 1: pick arms & retrieve neighbors --------------------------
    per_sample = []
    for i, s in enumerate(samples):
        nbrs = _retriever_topk(s, retriever, k_neighbors,
                               test_embs=test_embs, test_idx=i)
        if bandit is not None:
            arms = bandit.top_arms(s.uid, s.iid, retriever, n=n_candidates)
        else:
            arms = template_indices[:n_candidates]
        per_sample.append({"i": i, "nbrs": nbrs, "arms": arms})

    # ---- Phase 2: parallel LLM generation for all (sample, arm) pairs -----
    jobs = []
    for rec in per_sample:
        for a in rec["arms"]:
            jobs.append((rec["i"], a))

    def gen(job):
        i, a = job
        s = samples[i]
        tmpl = TEMPLATES[a]
        msg = build_prompt(s, per_sample[i]["nbrs"], tmpl)
        return (i, a, invoke_bedrock(llm, msg, temperature=tmpl["temperature"]))

    results_by_job: Dict[Tuple[int, int], str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(gen, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Stage3-gen"):
            i, a, out = fut.result()
            results_by_job[(i, a)] = out

    # ---- Phase 3: flatten candidates × neighbor refs into one BERTScore call
    # Proxy score = mean of top-3 F1s when scored against {nearest k neighbor
    # refs}. Nearest neighbors are sample-specific, but we also include a
    # "stylistic prior" of each candidate's F1 against the same-item refs
    # (strongest signal) if available.
    flat_preds, flat_refs, index_map = [], [], []
    for i in range(N):
        nbr_refs = [n.reference for n in per_sample[i]["nbrs"]]
        for a in per_sample[i]["arms"]:
            cand = results_by_job.get((i, a), "") or ""
            for nr in nbr_refs:
                flat_preds.append(cand)
                flat_refs.append(nr)
                index_map.append((i, a))

    print(f"  Scoring {len(flat_preds)} candidate×neighbor pairs...")
    if flat_preds:
        flat_f1 = bert_f1_batch(flat_preds, flat_refs)
    else:
        flat_f1 = np.array([])

    # Aggregate: mean of top-3 neighbor F1s (robust to outliers)
    from collections import defaultdict
    grouped: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for idx, (i, a) in enumerate(index_map):
        grouped[(i, a)].append(float(flat_f1[idx]))
    agg_by_arm: Dict[Tuple[int, int], float] = {}
    for key, scores in grouped.items():
        arr = np.sort(np.array(scores))
        topk = min(3, len(arr))
        agg_by_arm[key] = float(arr[-topk:].mean()) if topk else 0.0

    # ---- Phase 4: pick best per sample ------------------------------------
    chosen_preds = [None] * N
    per_sample_info = [None] * N
    for i in range(N):
        arms = per_sample[i]["arms"]
        cand_texts = [results_by_job.get((i, a), "") or "" for a in arms]
        scores = [agg_by_arm.get((i, a), 0.0) for a in arms]
        # Filter out empty candidates
        valid = [(a, t, s) for a, t, s in zip(arms, cand_texts, scores) if t]
        if not valid:
            chosen_preds[i] = ""
            per_sample_info[i] = {"i": i, "chosen": "", "cands": [],
                                  "proxy_scores": []}
            continue
        best = max(valid, key=lambda x: x[2])
        chosen_preds[i] = best[1]
        per_sample_info[i] = {
            "i": i,
            "chosen": best[1],
            "chosen_arm": best[0],
            "chosen_name": TEMPLATES[best[0]]["name"],
            "cands": cand_texts,
            "arms": arms,
            "arm_names": [TEMPLATES[a]["name"] for a in arms],
            "proxy_scores": scores,
        }

    return chosen_preds, refs, per_sample_info


# ---------- Contextual bandit -------------------------------------------------


class ContextualBandit:
    """Simple LinUCB-style bandit over template arms.

    Context = [user_emb | item_emb] (1536 dims).
    Reward = per-sample BERTScore F1 in [0,1].
    """

    def __init__(self, n_arms: int, d: int = 1536, alpha: float = 1.0):
        self.n_arms = n_arms
        self.d = d
        self.alpha = alpha
        self.A = [np.eye(d, dtype=np.float64) for _ in range(n_arms)]
        self.b = [np.zeros(d, dtype=np.float64) for _ in range(n_arms)]

    def context_from_pair(self, uid: int, iid: int,
                          retriever: NeighborRetriever) -> np.ndarray:
        un = retriever.u2n.get(uid)
        inn = retriever.i2n.get(iid)
        if un is None or inn is None:
            return np.zeros(self.d, dtype=np.float64)
        return np.concatenate([retriever.emb[un], retriever.emb[inn]]
                              ).astype(np.float64)

    def ucb(self, x: np.ndarray) -> np.ndarray:
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            Ainv = np.linalg.inv(self.A[a])
            theta = Ainv @ self.b[a]
            mean = float(theta @ x)
            var = float(np.sqrt(max(x @ Ainv @ x, 1e-9)))
            scores[a] = mean + self.alpha * var
        return scores

    def top_arms(self, uid: int, iid: int, retriever: NeighborRetriever,
                 n: int = 3) -> List[int]:
        x = self.context_from_pair(uid, iid, retriever)
        s = self.ucb(x)
        return list(np.argsort(-s)[:n])

    def update(self, uid: int, iid: int, retriever: NeighborRetriever,
               arm: int, reward: float):
        x = self.context_from_pair(uid, iid, retriever)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x

    def save(self, path: Path):
        np.savez(path,
                 A=np.stack(self.A),
                 b=np.stack(self.b),
                 alpha=self.alpha,
                 n_arms=self.n_arms,
                 d=self.d)

    @classmethod
    def load(cls, path: Path) -> "ContextualBandit":
        z = np.load(path)
        obj = cls(int(z["n_arms"]), int(z["d"]), float(z["alpha"]))
        obj.A = [z["A"][i] for i in range(obj.n_arms)]
        obj.b = [z["b"][i] for i in range(obj.n_arms)]
        return obj


def train_bandit(trn_samples: List[Sample], llm: BedrockLLM,
                 retriever: NeighborRetriever, n_episodes: int = 300,
                 k_neighbors: int = 5, max_workers: int = 10,
                 alpha: float = 1.0) -> ContextualBandit:
    """Train LinUCB bandit: for a sampled training sample, pick arm by UCB,
    generate, score BERTScore vs the TRAINING reference, update bandit."""
    bandit = ContextualBandit(n_arms=len(TEMPLATES), alpha=alpha)

    # Pre-sample episodes deterministically for reproducibility
    rng = random.Random(42)
    idx_pool = rng.sample(range(len(trn_samples)), k=min(n_episodes, len(trn_samples)))

    pbar = tqdm(total=len(idx_pool), desc="Bandit-train")

    # We still batch by grouping a handful of episodes, doing LLM calls in
    # parallel, then updating bandit with the rewards (sequential update).
    BATCH = max_workers

    for start in range(0, len(idx_pool), BATCH):
        batch = idx_pool[start:start + BATCH]
        # Pick arm per sample (greedy UCB from current bandit state)
        picks = []
        for idx in batch:
            s = trn_samples[idx]
            x = bandit.context_from_pair(s.uid, s.iid, retriever)
            ucb = bandit.ucb(x)
            a = int(np.argmax(ucb))
            picks.append((idx, a))

        # Parallel LLM generation
        outs = [None] * len(batch)

        def gen(j):
            idx, a = picks[j]
            s = trn_samples[idx]
            nbrs = retriever.topk(s.uid, s.iid, k=k_neighbors)
            tmpl = TEMPLATES[a]
            msg = build_prompt(s, nbrs, tmpl)
            return j, invoke_bedrock(llm, msg, temperature=tmpl["temperature"])

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for fut in as_completed([ex.submit(gen, j) for j in range(len(batch))]):
                j, out = fut.result()
                outs[j] = out

        # Score rewards in one BERTScore batch
        preds = [o or "" for o in outs]
        refs = [trn_samples[picks[j][0]].reference for j in range(len(batch))]
        f1 = bert_f1_batch(preds, refs)

        # Update bandit
        for j, (idx, a) in enumerate(picks):
            s = trn_samples[idx]
            # Clip to [0,1] for stability
            r = float(np.clip(f1[j], 0.0, 1.0))
            bandit.update(s.uid, s.iid, retriever, a, r)

        pbar.update(len(batch))
    pbar.close()
    return bandit


# ---------- Evaluation runner -------------------------------------------------


def evaluate_and_report(preds: List[str], refs: List[str],
                        label: str, out_dir: Path,
                        per_sample_info: Optional[List[dict]] = None):
    # Drop empties (shouldn't happen but be safe)
    cleaned = [(p, r) for p, r in zip(preds, refs) if p and r]
    if not cleaned:
        print(f"  [{label}] No valid predictions!")
        return None
    preds_c = [p for p, _ in cleaned]
    refs_c = [r for _, r in cleaned]
    f1 = bert_f1_batch(preds_c, refs_c)
    mean, std = float(np.mean(f1)), float(np.std(f1))
    print(f"\n=== {label} ===")
    print(f"  Samples scored: {len(preds_c)}")
    print(f"  BERTScore F1: {mean:.4f} ± {std:.4f}")
    print(f"  vs G-Refer 0.4592: {(mean / 0.4592 - 1) * 100:+.1f}%")
    print(f"  >=0.60 target: {'✓ HIT' if mean >= 0.60 else '✗'}")

    out = {
        "label": label,
        "f1_mean": mean,
        "f1_std": std,
        "n_samples": len(preds_c),
        "per_sample_f1": f1.tolist(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{label}_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    # Save a handful of examples for manual inspection
    preview = []
    for i in range(min(15, len(preds_c))):
        item = {"pred": preds_c[i], "ref": refs_c[i], "f1": float(f1[i])}
        if per_sample_info and per_sample_info[i]:
            item.update({
                "chosen_arm": per_sample_info[i].get("chosen_name"),
                "all_cands": per_sample_info[i].get("cands"),
                "proxy_scores": per_sample_info[i].get("proxy_scores"),
            })
        preview.append(item)
    with open(out_dir / f"{label}_preview.json", "w") as f:
        json.dump(preview, f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2, 3], default=3)
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--k_neighbors", type=int, default=5)
    ap.add_argument("--n_candidates", type=int, default=6)
    ap.add_argument("--max_workers", type=int, default=10)
    ap.add_argument("--template_idx", type=int, default=3,
                    help="Which template to use in Stage 2 single-gen")
    ap.add_argument("--train_bandit", action="store_true")
    ap.add_argument("--bandit_samples", type=int, default=300)
    ap.add_argument("--bandit_path", type=str,
                    default="results/rag_bandit/bandit.npz")
    ap.add_argument("--out_root", type=str, default="results/rag_bandit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_id", type=str,
                    default="anthropic.claude-3-haiku-20240307-v1:0",
                    help="Bedrock model ID")
    ap.add_argument("--use_titan", action="store_true",
                    help="Use Titan text embeddings for retrieval instead of "
                         "degenerate graph embeddings.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"stage{args.stage}_n{args.num_samples}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {run_dir}")

    # Load data
    trn_samples, tst_samples = load_grefer_samples()

    # Sample test set
    rng = np.random.RandomState(args.seed)
    n = min(args.num_samples, len(tst_samples))
    sel = rng.choice(len(tst_samples), size=n, replace=False)
    eval_samples = [tst_samples[i] for i in sel]
    print(f"Evaluating on {len(eval_samples)} tst samples")

    # Init LLM
    print(f"\nInitializing Bedrock LLM ({args.model_id})...")
    llm = BedrockLLM(model_id=args.model_id, max_tokens=300, temperature=0.0)

    # Build retriever
    test_embs = None
    if args.use_titan:
        cache_path = Path("results/rag_bandit/titan_cache.json")
        print("\nUsing Titan text embeddings for retrieval.")
        # Embed training prompts
        trn_texts = [_short_context(s.prompt) for s in trn_samples]
        trn_embs = embed_texts_titan(llm, trn_texts, cache_path=cache_path,
                                     max_workers=args.max_workers)
        # Embed test prompts (only the eval subset)
        tst_texts = [_short_context(s.prompt) for s in eval_samples]
        test_embs = embed_texts_titan(llm, tst_texts, cache_path=cache_path,
                                      max_workers=args.max_workers)
        retriever = TextRetriever(trn_samples, trn_embs)
    else:
        emb, u2n, i2n = load_node_embeddings()
        retriever = NeighborRetriever(trn_samples, emb, u2n, i2n)

    # === Stage 1: G-Refer prompt only ==========================================
    if args.stage == 1:
        preds, refs = run_stage1(eval_samples, llm, max_workers=args.max_workers)
        evaluate_and_report(preds, refs, "stage1_plain", run_dir)

    # === Stage 2: +k-NN few-shot ==============================================
    elif args.stage == 2:
        preds, refs = run_stage2(
            eval_samples, llm, retriever,
            k_neighbors=args.k_neighbors,
            template_idx=args.template_idx,
            max_workers=args.max_workers,
            test_embs=test_embs,
        )
        evaluate_and_report(
            preds, refs,
            f"stage2_fewshot{args.k_neighbors}_tmpl{args.template_idx}",
            run_dir
        )

    # === Stage 3: best-of-N + bandit ==========================================
    else:
        bandit = None
        if args.train_bandit:
            print(f"\nTraining bandit over {len(TEMPLATES)} templates, "
                  f"{args.bandit_samples} episodes...")
            bandit = train_bandit(
                trn_samples, llm, retriever,
                n_episodes=args.bandit_samples,
                k_neighbors=args.k_neighbors,
                max_workers=args.max_workers,
            )
            bpath = Path(args.bandit_path)
            bpath.parent.mkdir(parents=True, exist_ok=True)
            bandit.save(bpath)
            print(f"Bandit saved to {bpath}")
        else:
            bpath = Path(args.bandit_path)
            if bpath.exists():
                bandit = ContextualBandit.load(bpath)
                print(f"Loaded bandit from {bpath}")

        preds, refs, info = run_stage3(
            eval_samples, llm, retriever,
            k_neighbors=args.k_neighbors,
            n_candidates=args.n_candidates,
            max_workers=args.max_workers,
            bandit=bandit,
            test_embs=test_embs,
        )
        label = "stage3_bandit" if bandit else "stage3_uniform"
        evaluate_and_report(preds, refs, label, run_dir, per_sample_info=info)

        # Ensemble: take max of {G-Refer, our best}
        grefer_preds = [s.grefer_output for s in eval_samples]
        # Compute true F1 per sample vs ref for both and take best
        our_f1 = bert_f1_batch(preds, refs)
        grefer_f1 = bert_f1_batch(grefer_preds, refs)
        ens_preds = [preds[i] if our_f1[i] >= grefer_f1[i] else grefer_preds[i]
                     for i in range(len(preds))]
        # NOTE: Picking by true F1 uses test-set references, which is
        # oracle-selection (upper bound). We report this only for diagnostic.
        evaluate_and_report(ens_preds, refs, label + "_oracle_ensemble", run_dir)

        # A fairer ensemble: pick by proxy score (higher of our proxy vs
        # estimated proxy for G-Refer too). G-Refer doesn't have a proxy,
        # so we approximate by scoring G-Refer output vs retrieved neighbors.
        print("\nComputing fair ensemble (proxy-based, no test-ref leakage)...")
        # One big batch across all samples × neighbor refs × {ours, grefer}
        flat_preds, flat_refs, tag = [], [], []
        for i, s in enumerate(eval_samples):
            nbrs = _retriever_topk(s, retriever, args.k_neighbors,
                                   test_embs=test_embs, test_idx=i)
            nbr_refs = [n.reference for n in nbrs] or [""]
            for nr in nbr_refs:
                flat_preds.append(preds[i]); flat_refs.append(nr); tag.append((i, "ours"))
                flat_preds.append(grefer_preds[i]); flat_refs.append(nr); tag.append((i, "grefer"))
        flat_f1 = bert_f1_batch(flat_preds, flat_refs)
        from collections import defaultdict
        by = defaultdict(list)
        for idx, (i, lbl) in enumerate(tag):
            by[(i, lbl)].append(float(flat_f1[idx]))
        fair_preds = []
        for i in range(len(eval_samples)):
            ours_p = max(by.get((i, "ours"), [0.0]))
            grefer_p = max(by.get((i, "grefer"), [0.0]))
            fair_preds.append(preds[i] if ours_p >= grefer_p else grefer_preds[i])
        evaluate_and_report(fair_preds, refs, label + "_fair_ensemble", run_dir)

    print(f"\n✓ Done. Results saved to {run_dir}")


if __name__ == "__main__":
    main()
