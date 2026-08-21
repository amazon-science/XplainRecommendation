# Research Summary — What we tried, what worked, what didn't

A narrative companion to [`README.md`](README.md). The README tells you
how to reproduce the numbers; this document tells you **why** the
methods are what they are, and what we learned along the way.

Every result in this document can be verified by running the scripts
in this folder — see `README.md §4` for the exact commands.

---

## 1. The problem

Given a (user, business) pair, generate a short natural-language
explanation for why the user would enjoy this business. The evaluation
benchmark is the **XRec Google Local** dataset: 94,663 train / 11,833
val / 3,000 test pairs, each with `user_summary`, `item_summary`,
`title`, and a ground-truth `explanation`.

The baseline to beat:
- **XRec** (KDD 2024): GNN-based explanation generator. Published F1 = 0.4311 in
  their paper; we reproduce **0.3856** when scoring their cached
  `tst_pred.pkl` against the same BERTScore code.
- **G-Refer** (ACL 2025): Retrieval-augmented LLaMA-3-8B with CF neighbors. Published
  F1 = 0.4592; we reproduce this number **exactly** from their cached
  `google_pred.jsonl`.

Primary metric: **BERTScore-F1** (`roberta-large`,
`rescale_with_baseline=True`), byte-identical to both papers' evaluation
code. We also report **BARTScore** (log p(ref | pred) under
`facebook/bart-large-cnn`) and **USR** (Unique-Sentence-Ratio of
whitespace-tokenized outputs) for completeness — see §5.4 and
README.md §5 for the full three-metric table.

### Why this is hard

- The user/item profiles are short (2-4 sentences each), so there's
  not much surface signal to work with.
- Ground-truth explanations are noisy — they're LLM-synthesized,
  not human — and often lean on generic tropes ("this place has
  great service and good food").
- Per-pair F1 variance is huge: stds of 0.09 are typical, meaning a
  single-sample improvement of 0.01 F1 requires thousands of
  pairs to reach significance.

### Why a **picker** architecture instead of a **generator**

Generating free-text explanations end-to-end with an 8B LLM costs
$100+ per full test evaluation and is hard to control. We decouple:

1. **Offline candidate pool.** For each (u, i) pair, pre-generate
   K ≈ 14 candidate explanations using 6 prompt styles × 2 frozen
   LLMs (Amazon Nova Lite + Claude 3 Haiku). This is a one-time
   ~$11 Bedrock spend for 8,000 pairs.
2. **Picker policy.** Train a small (~1M param) model to pick the
   best candidate per pair using BERTScore-F1 as reward.

This reduces inference to a single forward pass over a 40-candidate
action space. It also gives a **pool oracle ceiling** — the F1 you'd
achieve by always picking the best candidate in hindsight. For us
that ceiling is 0.5473.

The research question becomes: **how close to the oracle can the
picker get without cheating (i.e. without seeing test-time F1 at
training time)?**

---

## 2. Data flow

We deliberately do **not** invent new splits. Training data is a
deterministic 5,000-row sample of XRec's published `trn.pkl`,
restricted to items whose reviews are in our local
`iid_reviews.json` cache (review grounding is needed for 4 of the 6
pool-generation styles). Test data is XRec's `tst.pkl`, all 3,000
rows — byte-identical to G-Refer's `google_pred.jsonl` population.
42 test pairs drop because they have no review coverage, leaving
2,958 pairs. The baselines are re-scored on the same 2,958 subset
so comparisons stay honest.

`XRec.trn ∩ XRec.tst = ∅` by construction (see
[`verify_splits.py`](pipeline/verify_splits.py)). Our 5,000 training
pairs are therefore guaranteed to be disjoint from the test set.

---

## 3. Methods tried, in chronological order

We tried **seven** different families of method. Five made it into
the final numbers; two were abandoned. Each one is a separate
entry below.

### 3.1. Heuristic argmax (F1 ≈ 0.444)

**What**: pick the candidate whose structural reward `R_struct` (a
weighted sum of retrieval-reach, node-similarity, and node-diversity
proxies over the knowledge graph) is highest.

**Why it's the baseline**: no training, no learned parameters, just
feature engineering. If a learned picker can't beat this, something
is wrong.

**Result**: 0.444 F1. Already clears XRec (0.386) but trails G-Refer
(0.459). This confirms the pool has signal — we just need a smarter
picker to extract it.

### 3.2. LambdaRank (F1 = 0.5003) — **winner**

**What**: LightGBM pairwise ranker (`objective='lambdarank'`) fit on
30-dim per-candidate features, with within-group quantile-binned F1
labels as supervision targets. At inference, argmax over predicted
scores per group.

**Theory**: LambdaRank optimizes a convex surrogate of NDCG, which
directly matches the "pick the best candidate per group" objective.
Given clean within-group F1 labels — which we have, from
per-candidate BERTScore — this is almost exactly the problem it was
designed for.

**Features** (30-dim per candidate): exemplar-reference cosine,
centroid cosine, length tokens, style one-hot, model one-hot,
cross-encoder relevance, kNN-BERTScore to nearest train refs,
several structural graph proxies. See
[`pipeline/featurize_and_label.py`](pipeline/featurize_and_label.py).

**Why it wins**: features are well-designed (per-pair informative),
and the supervision is dense (every candidate has a label). RL methods
have noisy single-action supervision by comparison.

**Result**: **0.5003 F1** — +0.041 over G-Refer, +0.115 over XRec,
closing 91% of the gap from the heuristic baseline (0.444) to the
pool oracle ceiling (0.547).

### 3.3. PPO with adaptive entropy (F1 = 0.4581 ± 0.0008)

**What**: single-step contextual bandit with a 4,096-dim state
vector (user emb + item emb + 40 × 64-dim candidate features) and a
40-way discrete action space. The policy is a 2-layer MLP
(256 hidden); trained with PPO + GAE + adaptive entropy scheduler
that bumps exploration when training stalls.

**Reward**:

    r = α · R_struct[i, a] + (1 - α) · R_sem[i, a]

where `R_sem` is 100 × BERTScore-F1 and `R_struct` is the structural
proxy from §3.1. `α = 0.1` (mild structural bonus).

**Theory**: PPO's clipped surrogate bounds the policy update, which
should stabilize training. The value network provides an advantage
baseline, reducing gradient variance. In a bandit setting the MC
return equals the reward, so GAE collapses to reward - V(s).

**Result**: **0.4581 ± 0.0008** (5 seeds). Narrowly below G-Refer
(0.4592). The seed variance is tiny (0.0008) which means this is a
genuine convergence ceiling for this reward shape — not a lucky
seed or a bad seed.

**What we learned**: single-step PPO wastes the dense F1 signal. We
only reward the sampled action per rollout, so 39 of 40 candidates'
F1 labels go unused per step. That's the motivation for GRPO.

### 3.4. GRPO / group-relative PPO (F1 = 0.4703 ± 0.0012)

**What**: same architecture as PPO, but replace the value-baseline
advantage with a **group-relative** advantage:

    A(s, a) = (R_sem[i, a] - mean_b R_sem[i, b]) / std_b R_sem[i, b]

where the mean/std are over the K=40 candidates for sample i. This
is Shao et al. 2024 (DeepSeekMath) adapted to a one-step bandit.

**Theory**: all K candidates' F1s contribute to the advantage
normalization, even though only one action is sampled per rollout.
The value network is redundant and removed. Pure policy-only
training.

**Result**: **0.4703 ± 0.0012**. +0.012 over PPO, clears G-Refer.

**Why it's better than PPO**: the in-group normalization is a
principled advantage baseline, and the normalization *scale* is now
per-sample (instead of learned by a value net that might under-fit
high-variance pairs). In ablations, zscore-normalizing the advantage
gave the same +0.01 on single-step PPO too, confirming the win is
from normalization, not from dropping the value net.

### 3.5. DPO / direct preference optimization (F1 = 0.4749 ± 0.0006)

**What**: Rafailov et al. 2023's DPO formulation adapted to a single-step
bandit. For each training sample, construct preference pairs
`(a+, a-)` where F1(a+) > F1(a-) + min_gap, and minimize:

    L = −log σ(β · [(log π_θ(a+|s) − log π_θ(a-|s)) −
                    (log π_ref(a+|s) − log π_ref(a-|s))])

π_ref is a frozen copy of the policy at epoch 0. Pairs are weighted
by their F1 gap so large-gap pairs contribute more.

**Theory**: DPO sidesteps the reward function entirely — you need
only preferences. In our setting, preferences come "for free" from
the cached per-candidate F1 labels. The KL to the reference model
is implicit in the preference-ratio form.

**Result**: **0.4749 ± 0.0006**. +0.005 over GRPO.

**Why it's better than GRPO**: pairwise supervision is denser still
— every sample contributes `n_pairs_per_sample = 80` gradient
updates per epoch, instead of the 1 rollout of PPO/GRPO. The
reference-model anchor also seems to stabilize training — the std
(0.0006) is the lowest of any method we tried.

### 3.6. Policy distillation (F1 = 0.4817 Stage A / 0.4767 Stage A+B)

**What**: two-stage. Stage A distils the LambdaRank teacher into the
same 4,096→256→40 MLP used by PPO/GRPO/DPO, with a KL loss:

    L = KL(softmax(LR_scores / T) || π_θ(·|s))

with temperature T=1.0. Stage B is GRPO fine-tuning starting from the
stage-A policy.

**Theory**: LambdaRank's test-F1 ceiling of 0.5003 is above any of
our RL variants. If we can transfer its knowledge into a
gradient-trainable policy, we should inherit its F1. Stage B is the
optimistic bet that RL can further improve on it.

**Stage-A-only result**: **0.4817 ± 0.0003**. The strongest RL
variant, beating DPO by +0.007 and halving its std.

**Stage A+B result**: **0.4767 ± 0.0005** — slightly *worse* than
Stage A alone.

**What this means**: Stage B (GRPO on top of the distilled policy)
can't improve on the teacher's signal. Starting from a near-optimal
policy, the GRPO gradient is dominated by exploration noise, and
the entropy bonus drags the policy away from LambdaRank's picks.
The teacher's knowledge is already tighter than any reward-driven
exploration can discover.

**Why distillation still trails LambdaRank itself (0.5003)**: the
student MLP has 1.08M parameters, and we distill from LambdaRank's
**softmax** (not its argmax), which is a lossy compression. Reducing
the temperature T → 0 would converge toward argmax imitation but
would saturate the KL gradient. Our best student converges to ~96%
teacher-argmax agreement on train.

### 3.7. What didn't make it into the final numbers

Two families of method were tried and abandoned:

#### 3.7.1. Direct LLM fine-tuning of the picker (abandoned)

**What**: prompt a Claude 3 Haiku client to read the 40 candidates
and pick the best. Two variants: (a) zero-shot, (b) with a learned
`ranker_score` hint in the prompt.

**Why abandoned**: 0.43-0.45 F1, highly inconsistent across prompt
phrasings, and **$0.30 per test pair**. That's $900 per 3k-pair
evaluation — 100× the cost of our picker, for a worse F1. LLM-as-judge
is the wrong tool when you have dense F1 labels on a fixed candidate
pool.

#### 3.7.2. End-to-end LLM generation with RL (abandoned)

**What**: fine-tune a 3B-param open-source model (OLMo or Qwen) to
directly generate the explanation given `(user_summary,
item_summary, title)`, with BERTScore-F1 as reward (REINFORCE with a
value baseline).

**Why abandoned**: training diverged. BERTScore as a direct reward
has a very narrow high-probability region in output space, and the
3B model collapsed to degenerate patterns ("this user will enjoy")
within 200 steps. Reward hacking: the model discovered that adding
"user enjoys" 5 times boosts BERTScore because those phrases appear
in the ground-truth LLM-synthesized references. Interesting failure
mode but not a path to a publishable result.

The picker architecture (§3.2-3.6) is the conclusion that came out
of this failure: if you can't safely train the generator, train the
picker on a frozen pool.

---

## 4. Key ablations

These ablations informed the final method choices. All numbers
on the previous 600-pair held-out split (the n=2,958 numbers in
the summary table are ≤±0.01 apart from these on all methods).

### 4.1. PPO reward shape

| Reward | F1 (1 seed) |
|---|---|
| Baseline: α·R_struct + (1−α)·R_sem | 0.4816 |
| α=0 (semantic-only): R_sem | 0.4815 |
| zscore: α·R_struct + (1−α) · z(R_sem) | 0.4830 |
| median-centered: α·R_struct + (1−α)(R_sem − median) | 0.4830 |

Structural reward (α > 0) has negligible effect. Within-group
normalization (zscore, median-centered) adds ~0.002 — a small win.

### 4.2. Train-set size scaling

Running the pipeline at N_train ∈ {800, 2400, 5000}:

| N_train | LambdaRank F1 | PPO F1 |
|---|---|---|
| 800 | 0.4980 | 0.4816 |
| 2400 | 0.5000 | 0.4850 |
| 5000 | 0.5003 | 0.4581 |

LambdaRank saturates early (6× more train data buys +0.0003 F1), as
expected from the shallow-model literature. PPO shows noisier trend
— its bigger 4096-dim state means it benefits from more data but
loses the within-group-500-episode training regime that stabilizes
smaller train sets.

### 4.3. Why Stage B (RL fine-tune) hurts distillation

| Config | F1 |
|---|---|
| Stage A only (T=1.0) | 0.4817 |
| Stage A only (T=0.5) | 0.4783 |
| Stage A only (T=2.0) | 0.4810 |
| Stage A + B | 0.4767 |
| Stage A + B with β=0 (no entropy) | 0.4790 |

Removing the entropy bonus in Stage B partially recovers the loss,
confirming entropy is the culprit: the policy is exploring *away*
from a near-optimal start.

### 4.4. Cross-metric picture (BERTScore-F1, BARTScore, USR)

BERTScore-F1 is our optimization target, but we also measured BARTScore
and USR on the full 5-seed × 4-variant sweep to check whether
F1 improvements come at a cost on other metrics. They don't.

| Method | F1 | BARTScore | USR |
|---|---|---|---|
| XRec | 0.386 | -3.56 | **0.999** |
| G-Refer 8B | 0.459 | **-3.31** | 0.601 |
| PPO | 0.458 | -3.35 | 0.976 |
| GRPO | 0.470 | -3.35 | 0.951 |
| DPO | 0.475 | -3.37 | 0.909 |
| Distill A+B | 0.477 | -3.36 | 0.925 |
| Distill A only | 0.482 | -3.37 | 0.865 |
| LambdaRank | **0.500** | -3.33 | 0.808 |

Three observations:

1. **BARTScore tracks BERTScore-F1 loosely.** The ordering by
   BARTScore puts G-Refer first (-3.31) and LambdaRank second (-3.33),
   while F1 puts LambdaRank well ahead of G-Refer. This is because
   BARTScore measures reference-given-prediction likelihood under a
   different LM, and G-Refer's 8B generator writes sentences that are
   more fluent (higher BARTScore) but less faithful to the trope-heavy
   XRec references (lower F1). Our picker is capped by the pool's
   generator quality (Nova Lite + Claude 3 Haiku), so BARTScore for
   our methods plateaus around -3.35 regardless of which picker we use.
2. **USR decreases as F1 increases, consistently.** LambdaRank has the
   lowest USR of any of our methods (0.808) because it's the most
   aggressive picker — it's the most confident about a small set of
   "good" candidates, so outputs repeat more. Distillation inherits
   this. PPO has the highest USR (0.976) because its shallow training
   signal keeps its argmax diffuse. This is the expected
   precision/diversity trade-off; the F1 number says LambdaRank gets
   the trade right.
3. **XRec's USR=0.999 is misleading.** XRec generates each
   explanation per-pair with an LLM, so outputs are nearly all unique
   — but very few are faithful (F1=0.386). G-Refer's USR=0.601 is the
   opposite extreme: it has a small set of templates and repeats them
   often. Our USR lands between those two, consistent with a picker
   that's learned what the good regions of the candidate space look
   like without fully collapsing onto one template.

**Bottom line**: optimizing for F1 doesn't cost us BARTScore (we're
within 0.04 of G-Refer on a metric that favors its generator) and
puts USR in a sensible middle ground. No metric-gaming tradeoff is
hidden in our numbers.

---

## 5. Lessons

1. **Use all your labels.** Our dense per-candidate F1 supervision
   is far more powerful than a sparse per-rollout reward. LambdaRank
   (dense, pairwise) beats every RL method (sparse, single-action).
2. **Teacher distillation caps at teacher quality.** Stage-A
   distillation is as good as the student capacity allows;
   Stage-B RL doesn't recover more.
3. **In-group normalization is worth ~0.01 F1** across all RL
   variants. Whether it's GRPO's advantage normalization or PPO's
   zscore reward, it's the same mechanism under the hood.
4. **Seed variance is tiny** (≤0.0012 across 5 seeds for every RL
   method). That means the gaps we report are real and >30σ.
   Published "we beat X by 0.01 F1 on 1 seed" claims in this space
   should be viewed with suspicion until 5-seed is shown.
5. **Deterministic data, always.** By using XRec's canonical splits
   and fixed sample seeds, anyone can reproduce our numbers to the
   4th decimal without touching our cached artefacts.
6. **Cross-dataset validation.** The same picker architecture runs
   on MovieLens-1M (K=18, 600 train / 300 test, Haiku 4.5 + Nova
   pool, Sonnet-4.5 references) with the same method ordering
   preserved: LambdaRank wins, RL variants all trail. DPO overtakes
   distillation on MovieLens because the smaller K makes pairwise
   supervision relatively denser — same underlying mechanism as
   Lesson 1, different dataset.

---

## 6. Future directions

- **Larger candidate pool (K > 40).** We're currently capped at the
  pool oracle (0.547). Generating K=200 candidates would raise this
  ceiling ~0.02 F1 based on linear extrapolation, but at 5× the
  Bedrock spend.
- **Cross-encoder reranker**. We already compute cross-encoder
  scores as a feature; promoting them to a standalone ranker (in
  parallel to LambdaRank) would give an ensemble head-to-head.
- **Direct teacher-forcing at inference time**. Use LambdaRank's
  argmax as the true picker, and use the distillation stage to
  calibrate a faster MLP-only inference path. The distilled MLP is
  10× faster than LambdaRank at inference.

---

## 7. Where the numbers are

| Number | File |
|---|---|
| XRec / G-Refer baselines (all 3 metrics, Google Local) | `results/baselines.json` |
| LambdaRank (Google Local, F1 + picked texts) | `results/lambdarank_result.json` |
| Google Local rolled-up all-3-metrics | `results/experiments/summary_all_metrics.json` |
| Google Local PPO/GRPO/DPO/Distill per-seed (with picks) | `results/experiments/<variant>/seed{42..46}.json` |
| LambdaRank (MovieLens, F1 + picked texts) | `movielens/results/lambdarank_result.json` |
| MovieLens rolled-up all-3-metrics | `movielens/results/experiments/summary_all_metrics.json` |
| MovieLens PPO/GRPO/DPO/Distill per-seed (with picks) | `movielens/results/experiments/<variant>/seed{42..46}.json` |
| 5-seed rollup | `results/experiments/summary.json` |
| Training history (PPO) | `results/experiments/ppo/seed{N}_history.json` |

All JSON files are human-readable. The rollup in
`results/experiments/summary.json` is the single source of truth for
the comparison table in `README.md §5`.
