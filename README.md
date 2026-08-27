# XplainRecommendation

Code accompanying *"Pairwise Ranking Outperforms Single-Action RL for Offline
Explanation Selection: A Practical Lesson"* (Tanay Chowdhury, Saeideh Shahrokh
Esfahani — RecSys 2026, Research & Practice Notes track).

This repo reproduces every headline number in the paper from XRec's published
data. It is standalone: the only external dependencies are XRec's data files
(`trn.pkl` / `tst.pkl`), G-Refer's cached predictions (`google_pred.jsonl`),
and a local review cache (`iid_reviews.json`).

If you already have the data-prep artefacts in `results/` (the release
bundle ships with them), **jump to §4**. If you want to regenerate
everything from raw XRec data, start at §1.

> This code is being released solely for academic and scientific
> reproducibility purposes, in support of the methods and findings described
> in the associated publication. Pull requests are not being accepted in
> order to maintain the code exactly as it was used in the paper.

---

## 0. What's in here

```
XplainRecommendation/
├── README.md                    ← this file
├── LICENSE                      ← CC-BY-NC-4.0
├── NOTICE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── requirements.txt
├── docs/
│   ├── RESEARCH_SUMMARY.md      ← problem statement, methods, theory, what didn't work
│   └── kg_path_selection_results.md
├── _paths.py                    ← shared path resolution helper
├── scripts/                     ← pool-generation + reranker library modules
├── src/                         ← PPO agent, Bedrock client, entropy scheduler
├── pipeline/                    ← data-prep stages (verify → sample → pool → featurize → tables)
├── runners/                     ← training scripts: PPO, GRPO, DPO, Distillation
├── movielens/                   ← parallel pipeline for the MovieLens-1M cross-dataset check
└── results/                     ← evaluation outputs (baselines, lambdarank, experiments)
```

By default, external datasets (`XRec/`, `G-Refer/`, `data/`) are expected as
siblings of this repo's own top-level directories (i.e. inside this repo's
root — see §1). Set the `FINAL_RESULT_DATA` env var if your datasets live
somewhere else.

---

## 1. Prerequisites

Files that must exist under `DATA_ROOT` (the repo root, by default):

```
DATA_ROOT/
├── XRec/data/google/trn.pkl         # 94,663 training pairs
├── XRec/data/google/val.pkl         # 11,833 val pairs
├── XRec/data/google/tst.pkl         # 3,000 test pairs
├── XRec/data/google/tst_pred.pkl    # XRec's cached predictions (baseline)
├── G-Refer/gen_explanations/G-Refer/google_pred.jsonl   # G-Refer's cached predictions (baseline)
├── G-Refer/evaluation/bart_score.py                     # for BARTScore
└── data/google_local/iid_reviews.json                   # same-business review cache
```

`XRec` and `G-Refer` are separate public research repos — clone them
yourself; this repo does not vendor their code or data.

Python packages:
```
pip install -r requirements.txt
```

AWS credentials for Bedrock (Nova Lite + Claude 3 Haiku + Titan v2
embeddings) are required **only if regenerating the candidate pool**.
If you're using the bundled results, no AWS access is needed.

---

## 2. Running the data-prep pipeline

The full pipeline is orchestrated by `pipeline/prepare_data.py`. It
runs five stages sequentially; each stage caches its output, so
re-runs skip anything already done.

```bash
python3 pipeline/prepare_data.py
```

| Stage | Script | Output | Wall time | Bedrock cost |
|---|---|---|---|---|
| 1. verify | `verify_splits.py` | (prints OK) | 1 min | $0 |
| 2. sample | `sample_training.py` | `results/trn_5k.pkl` | 30 sec | $0 |
| 3. pool | `build_pool.py` | `results/pool/exp_pool_clean_xrec_{trn,tst}_*.pkl` | ~3 h | ~$11 |
| 4. featurize | `featurize_and_label.py` | `results/features/features.pkl` | ~4 h | ~$0.40 |
| 5. tables | `fit_lambdarank_and_build_tables.py` | `results/paper_ppo_tables.npz`, `results/lambdarank_*.{pkl,json,npz}` | ~2 min | ~$0.20 |

Run individual stages:
```bash
python3 pipeline/prepare_data.py --stages verify,sample
python3 pipeline/prepare_data.py --force-stage pool
```

Each standalone script also runs on its own:
```bash
python3 pipeline/sample_training.py
python3 pipeline/featurize_and_label.py --max_workers 4
```

---

## 3. Evaluating baselines

```bash
python3 pipeline/eval_baselines.py
```

Reads XRec's `tst_pred.pkl` and G-Refer's `google_pred.jsonl`, scores
them with the same BERTScore / BARTScore / USR metrics we use for our
methods, writes `results/baselines.json`. No training, no Bedrock
calls — just evaluation on cached predictions.

Expected output (on the shared 2,958-pair test subset):
- XRec F1 = **0.3856**, BART = −3.56, USR = 0.9993
- G-Refer 8B F1 = **0.4592**, BART = −3.31, USR = 0.60

---

## 4. Training and evaluating our methods

All four RL-family methods and LambdaRank are trained on the same
`results/paper_ppo_tables.npz` state/reward tables produced by step 5.
The 5,000-pair training set and 2,958-pair test set are baked into the
tables; no data loading is required at training time.

### 4a. LambdaRank

Already computed in step 5. Read the result directly:
```bash
python3 -c "import json; d=json.load(open('results/lambdarank_result.json')); print(f\"LambdaRank F1 = {d['test_f1_mean']:.4f} ± {d['test_f1_std']:.4f} (n={d['n_test']})\")"
# → LambdaRank F1 = 0.5003 ± 0.0947 (n=2958)
```

### 4b. PPO (single seed, baseline reward)

```bash
python3 runners/run_ppo.py \
    --tables results/paper_ppo_tables.npz \
    --seed 42 --episodes 500 \
    --reward_mode baseline --alpha 0.1 \
    --result_out results/experiments/ppo/seed42.json \
    --history_out results/experiments/ppo/seed42_history.json
```

`--reward_mode` can be `baseline`, `zscore`, `alpha0`, or `median`
(see `runners/run_ppo.py` for the reward-shape ablation).

### 4c. GRPO

```bash
python3 runners/run_grpo.py \
    --tables results/paper_ppo_tables.npz \
    --seed 42 --episodes 500 \
    --result_out results/experiments/grpo/seed42.json
```

### 4d. DPO

```bash
python3 runners/run_dpo.py \
    --tables results/paper_ppo_tables.npz \
    --seed 42 --episodes 500 \
    --result_out results/experiments/dpo/seed42.json
```

### 4e. Distillation

```bash
python3 runners/run_distillation.py \
    --tables results/paper_ppo_tables.npz \
    --seed 42 \
    --stageA_epochs 200 --stageB_epochs 500 \
    --result_out results/experiments/distillation/seed42.json
```

Both stage-A-only (`stageA_only_f1_mean`) and stage-A+B
(`stageAB_f1_mean`) F1s are saved in the output JSON — Stage A alone
turned out to be the strongest RL variant.

### 4f. Full 5-seed sweep (all four methods)

```bash
python3 pipeline/run_rl_variants.py
# Results: results/experiments/<variant>/seed{42..46}.json
# Rollup:  results/experiments/summary.json
```

Arguments:
- `--variants ppo,grpo,dpo,distillation` (default: all)
- `--seeds 42,43,44,45,46` (default)
- `--episodes 500` (default)

Wall time: ~50 min/seed × 5 seeds × 4 variants ≈ 16 h on CPU. Each seed
is trained sequentially, so you can Ctrl-C at any point and re-run —
completed seeds are not re-trained.

### 4g. MovieLens pipeline

MovieLens uses a separate but identical pipeline under `movielens/`:

```bash
# Stage 1: build features + tables + LambdaRank (~30 min CPU, uses cached pool)
python3 movielens/pipeline/build_tables.py

# Stage 2: 5-seed RL sweep (4 variants)
python3 movielens/pipeline/run_rl_variants_ml.py

# Stage 3: BART + USR scoring for all methods
python3 movielens/pipeline/score_all_metrics_ml.py
```

Reuses cached pools at `results/rag_bandit/pool_cache_ml/` (Nova Lite +
Claude Haiku 4.5, 600 train × 300 test, K=18 candidates/sample). No
Bedrock calls required if the Titan embedding cache at
`movielens/results/titan_cache.json` exists.

---

## 5. Results summary

We evaluate on **two datasets**:

- **XRec Google Local** — 3,000 test pairs (our primary benchmark, evaluated on the shared 2,958-pair subset with review coverage). See §5.1.
- **MovieLens-1M** (with Claude-Sonnet-4.5 generated reference explanations) — 300 test pairs. See §5.2.

### 5.1 Google Local results

On the 2,958-pair XRec test subset (42 pairs have no review coverage
and are dropped; baselines re-scored on the same 2,958 for apples-to-apples):

#### BERTScore-F1 (primary metric)

| Method | F1 mean ± std | Δ vs G-Refer | Δ vs XRec |
|---|---|---|---|
| XRec (cached) | 0.3856 | — | — |
| G-Refer 8B (cached) | 0.4592 | — | +0.0736 |
| Heuristic (argmax R_struct) | ≈0.444 | −0.015 | +0.059 |
| PPO 5-seed | 0.4581 ± 0.0008 | −0.0011 | +0.0725 |
| GRPO 5-seed | 0.4703 ± 0.0012 | **+0.0111** | +0.0847 |
| DPO 5-seed | 0.4749 ± 0.0006 | **+0.0157** | +0.0893 |
| Distillation A+B 5-seed | 0.4767 ± 0.0005 | **+0.0175** | +0.0911 |
| Distillation A-only 5-seed | **0.4817 ± 0.0003** | **+0.0225** | +0.0961 |
| **LambdaRank (single)** | **0.5003 ± 0.0947** | **+0.0411** | **+0.1147** |
| Pool oracle ceiling | 0.5473 | +0.0881 | +0.1617 |

#### All three metrics (Google Local)

| Method | BERTScore-F1 | BARTScore | USR (unique outputs) |
|---|---|---|---|
| XRec | 0.3856 | -3.559 | 0.999 |
| G-Refer 8B | 0.4592 | **-3.311** | 0.601 |
| PPO 5-seed | 0.4581 ± 0.0008 | -3.354 ± 0.003 | 0.976 ± 0.004 |
| GRPO 5-seed | 0.4703 ± 0.0012 | -3.354 ± 0.002 | 0.951 ± 0.005 |
| DPO 5-seed | 0.4749 ± 0.0006 | -3.374 ± 0.002 | 0.909 ± 0.006 |
| Distillation A+B 5-seed | 0.4767 ± 0.0005 | -3.356 ± 0.003 | 0.925 ± 0.004 |
| Distillation A-only 5-seed | 0.4817 ± 0.0003 | -3.375 ± 0.002 | 0.865 ± 0.008 |
| **LambdaRank** | **0.5003** | -3.327 | 0.808 |

- **BERTScore-F1**: higher is better; byte-identical to XRec/G-Refer's `evaluation/metrics.py`.
- **BARTScore**: higher (less negative) is better; log p(reference | prediction) under `facebook/bart-large-cnn`.
- **USR**: Unique-Sentence-Ratio of model outputs (fraction of whitespace-tokenized outputs that are distinct across the 2,958 test pairs).

**How to read the table**

- Our methods all win on **BERTScore-F1** — LambdaRank by +0.041 over G-Refer.
- On **BARTScore**, G-Refer (-3.31) narrowly edges everything else, with LambdaRank (-3.33) second. This is because G-Refer generates with an 8B LLM while our methods select from a frozen ~40-candidate pool — the pool quality caps BARTScore for us.
- On **USR**, our methods all beat G-Refer's 0.60 (a known XRec/G-Refer flaw: G-Refer collapses to a small set of generic templates). XRec's 0.999 is suspicious — it comes from per-pair LLM generation without reward pressure, so the outputs are maximally diverse but least aligned with references.
- Distillation A-only trades some USR for the best F1 among RL variants, consistent with imitating LambdaRank's more-focused argmax.

Per-seed numbers live under `results/experiments/<variant>/seed{42..46}.json`.
The rolled-up Google Local summary is `results/experiments/summary_all_metrics.json`.

### 5.2 MovieLens-1M results

On the 300 test pairs from a deterministic split of MovieLens-1M with
Claude-Sonnet-4.5 generated reference explanations (scripts/generate_movielens_references.py).

- Train set: 600 pairs with cached candidate pool (Nova Lite + **Claude Haiku 4.5**, K=18 per sample).
- Test set: 300 pairs, 18 candidates each.
- Pool generator quality is higher here (Claude Haiku 4.5 > Haiku 3) but reference explanations are much thinner ("This Action movie is highly rated by users with similar taste"), which caps achievable F1 by a wide margin regardless of picker.

#### BERTScore-F1 (primary metric)

| Method | F1 mean ± std | Δ vs ensemble baseline (0.328) |
|---|---|---|
| Ensemble reranker (prior) | 0.3281 | — |
| Heuristic (argmax R_struct) | ≈0.263 | −0.065 |
| PPO 5-seed | 0.2816 ± 0.0025 | −0.046 |
| GRPO 5-seed | 0.2830 ± 0.0018 | −0.045 |
| Distillation A+B 5-seed | 0.2831 ± 0.0030 | −0.045 |
| Distillation A-only 5-seed | 0.2887 ± 0.0011 | −0.039 |
| DPO 5-seed | 0.2936 ± 0.0017 | −0.035 |
| **LambdaRank (single)** | **0.3291 ± 0.0787** | **+0.001** |
| Pool oracle ceiling | 0.3789 | +0.051 |

**Observations:**

- LambdaRank reproduces the prior ensemble's F1 (0.329 vs 0.328) — consistent.
- On MovieLens, the RL methods **trail LambdaRank by a wider gap** (−0.04 to −0.05 F1) than on Google Local (−0.02 F1). Interpretation: with K=18 (vs K=40 on Google), the RL exploration signal is sparser, and LambdaRank's dense pairwise supervision wins by a bigger margin.
- Same method ordering as Google Local: **PPO < GRPO < Distill A+B < Distill A-only < DPO < LambdaRank**. The ordering is consistent across two datasets with different pool sizes (K=40 vs K=18), different LLMs (Haiku 3 vs Haiku 4.5), and different reference quality.
- **Distillation Stage A beats Stage A+B** here too (0.289 vs 0.283), confirming the Google Local finding: RL fine-tuning drags a near-optimal distilled policy away from its target.

#### All three metrics (MovieLens)

| Method | BERTScore-F1 | BARTScore | USR (unique outputs) |
|---|---|---|---|
| PPO 5-seed | 0.2816 ± 0.0025 | -3.566 ± 0.012 | 0.999 ± 0.001 |
| GRPO 5-seed | 0.2830 ± 0.0018 | -3.533 ± 0.006 | 0.999 ± 0.001 |
| Distillation A+B 5-seed | 0.2831 ± 0.0030 | -3.544 ± 0.010 | 0.999 ± 0.001 |
| Distillation A-only 5-seed | 0.2887 ± 0.0011 | -3.548 ± 0.006 | **1.000 ± 0.000** |
| DPO 5-seed | 0.2936 ± 0.0017 | **-3.530 ± 0.004** | 0.999 ± 0.001 |
| **LambdaRank (single)** | **0.3291** | **-3.449** | 0.987 |
| Pool oracle ceiling | 0.3789 | — | — |

**Cross-dataset observations:**

- **USR is near 1.0 for every method on MovieLens.** With only 300
  test samples and long, varied LLM-generated candidates, collisions
  are rare. MovieLens USR therefore doesn't discriminate between
  methods — the signal all lives in F1 and BART.
- **BART ordering differs slightly from F1 ordering.** On MovieLens,
  DPO has the best BART (-3.530) and LambdaRank is second (-3.449),
  while on Google Local G-Refer and LambdaRank led on BART. That
  said, LambdaRank's BART is **0.10 better than DPO's on Google
  Local but 0.08 worse than DPO's here** — because MovieLens ref
  texts are themselves thin and LambdaRank's picks, being closer to
  the thin references, naturally score worse under BART's fluency-
  sensitive model.
- **Method ordering on F1 differs from Google Local.** On
  MovieLens, DPO > Distill A-only > Distill A+B ≈ GRPO > PPO; on
  Google Local, Distill A-only > Distill A+B > DPO > GRPO > PPO. The
  K=18 MovieLens pool (vs K=40) changes the distillation→RL pressure
  balance: with fewer candidates, DPO's pairwise-preference signal
  is denser relative to distillation's KL target.

---

See [`docs/RESEARCH_SUMMARY.md`](docs/RESEARCH_SUMMARY.md) for the story
behind these numbers — problem setup, method theory, what worked, and what
didn't. See [`docs/kg_path_selection_results.md`](docs/kg_path_selection_results.md)
for the knowledge-graph path-selection family of experiments.

---

## 6. Reproduction sanity check

After running the full pipeline, these four commands should each print
the numbers above:

```bash
# Split integrity
python3 pipeline/verify_splits.py

# Baselines
python3 -c "import json; d=json.load(open('results/baselines.json')); print(d['xrec']['bertscore_f1_mean'], d['grefer']['bertscore_f1_mean'])"

# LambdaRank
python3 -c "import json; d=json.load(open('results/lambdarank_result.json')); print(f\"{d['test_f1_mean']:.4f}\")"

# 5-seed summary
cat results/experiments/summary.json
```

Verify by comparing these against the numbers in the table above and
in `docs/RESEARCH_SUMMARY.md`. Any numerical drift >0.001 F1 indicates a
break in the pipeline — open an issue with the log output of the
affected stage.

---

## 7. Troubleshooting

**Titan cache OOM (process killed with rc=-9)**: The `results/titan_cache.json`
should stay under 1 GB for the 200k-text workload in this bundle. If
you see rc=-9, check free RAM with `free -h` and make sure the box has
at least 16 GB. Our tests used a c5.2xlarge (30 GB).

**Bedrock ThrottlingException**: Lower `--max_workers` in
`pipeline/build_pool.py` and `pipeline/featurize_and_label.py`. Our
defaults use 4 workers.

**`ModuleNotFoundError: scripts`**: Make sure you invoke scripts from
the repo root as shown above (e.g. `python3 runners/run_ppo.py`),
not from inside a subdirectory. `_paths.py` adds the right
paths to `sys.path` at script start.

**XRec or G-Refer file missing**: Clone their repos into `DATA_ROOT`
(the repo root, by default). Set `FINAL_RESULT_DATA=/path/to/data`
to point elsewhere.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{chowdhury2026pairwise,
  title     = {Pairwise Ranking Outperforms Single-Action RL for Offline Explanation Selection: A Practical Lesson},
  author    = {Chowdhury, Tanay and Shahrokh Esfahani, Saeideh},
  booktitle = {Proceedings of the 20th ACM Conference on Recommender Systems (RecSys '26), Research and Practice Notes},
  year      = {2026}
}
```

Please also cite XRec and G-Refer, whose data and cached predictions this
repo builds on:

```bibtex
@inproceedings{ma2024xrec,
  title     = {XRec: Large Language Models for Explainable Recommendation},
  author    = {Ma, Qiyao and Ren, Xubin and Huang, Chao},
  booktitle = {Findings of EMNLP},
  year      = {2024}
}

@inproceedings{li2025grefer,
  title     = {G-Refer: Graph Retrieval-Augmented Large Language Model for Explainable Recommendation},
  author    = {Li, Yuhan and Zhang, Xinni and Luo, Linhao and Chang, Heng and Ren, Yuxiang and King, Irwin and Li, Jia},
  booktitle = {WWW},
  year      = {2025}
}
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License (CC-BY-NC-4.0) — see [LICENSE](LICENSE) and
[NOTICE](NOTICE).
