# Saeideh's KG Path-Selection Results

Results from a colleague's parallel experiments on the same problem, using a
different approach (reinforcement learning over knowledge-graph paths) on the
same two datasets (Google Local, MovieLens-1M).

**Source:** `saeideh-kg-approach-experiments/`
**Approach:** KG path extraction + prompt-style generation + PPO selection
**Evaluation:** Same BERTScore-F1 / BARTScore / USR metrics we use, same
RoBERTa-large scoring code (`rescale_with_baseline=True`).

---

## Headline results (EXP-006, final split, no test leakage)

Google Local, n = **1,000 test pairs**, seed = 42:

| Method | BERTScore F1 | BARTScore | USR |
|---|---|---|---|
| G-Refer 8B (Li et al. 2025) | 0.4592 | — | — |
| KG Heuristic argmax | 0.3329 ± 0.077 | −3.565 | 1.000 |
| KG PPO single-style | 0.3291 ± 0.076 | −3.576 | 1.000 |
| **KG PPO multi-style** | **0.3306 ± 0.076** | **−3.570** | **1.000** |
| Training best checkpoint (ep200) | 0.320 | — | — |

MovieLens-1M, n = **300 test pairs**, seed = 42:

| Method | BERTScore F1 | BARTScore | USR |
|---|---|---|---|
| KG Heuristic argmax | 0.2680 ± 0.071 | −3.627 | 1.000 |
| KG PPO single-style | 0.2651 ± 0.070 | −3.627 | 1.000 |
| **KG PPO multi-style** | **0.2621 ± 0.070** | **−3.624** | **0.997** |
| Training best checkpoint (ep200) | 0.245 | — | — |

**Core finding (from their README):**
> The KG semantic ceiling is a function of graph density, not the RL
> algorithm. Google Local (dense, 329k edges): F1 ≈ 0.333. MovieLens-1M
> (sparse, 18 genre intermediaries): F1 ≈ 0.265. Same architecture, same
> code — ceiling drops −0.065 F1 purely from graph sparsity.

Source: [`saeideh-kg-approach-experiments/README.md`](saeideh-kg-approach-experiments/README.md)

---

## Experiment-by-experiment breakdown

### exp-001 — Reward Scaling Fix

Tested whether re-centering the reward signal at the empirical mean (0.278)
instead of a hand-picked 0.3 would prevent entropy collapse.

- Selection entropy (final): **0.035** (target > 1.0) — failed
- Mean BERTScore-F1: **0.274** (baseline 0.279) — no improvement

**Conclusion:** Reward scaling was not the bottleneck. Entropy still collapsed
to 0.035, indicating the issue is structural — path similarity prevents the
policy from distinguishing options.

Source: `experiments/exp-001-reward-scaling/{README.md, PROGRESS.md, STATUS.md}`

---

### exp-002 — Path Diversity Investigation

Diagnostic analysis of path extraction methods (shortest path, multihop, random
walks, path variations, detours).

- No quantitative metric generated; script had import issues.
- Manual code review identified 5 extraction methods, most producing shortest-
  path-like outputs.
- Random walk temperature = 0.5 flagged as too greedy.

**Conclusion:** Investigation incomplete. Root cause identified qualitatively:
path extraction produces structurally similar paths despite 5 different
methods. Proposed fixes: edge-disjoint paths, higher temperature, meta-path
templates.

Source: `experiments/exp-002-path-diversity/{README.md, SUMMARY.md, PLAN.md}`

---

### exp-003 — Temperature Fix (Quick Test)

Raised random walk temperature 0.5 → 1.5 to reduce path similarity.

- Selection entropy (final): **1.925** ✅ (vs baseline 0.048)
- Mean BERTScore-F1: **0.271** (baseline 0.279) — flat

**Conclusion:** Temperature raised entropy by ~40× but F1 didn't move. Paths
are still semantically similar at *generation* level even when structurally
more random.

Source: `experiments/exp-003-temperature-fix/PLAN.md`

---

### exp-004 — Edge-Disjoint Paths

Guaranteed structurally different paths (no shared edges).

- Selection entropy (final): **0.015** — worse than baseline
- Mean BERTScore-F1: **0.266** — worse

**Conclusion:** Structural diversity isn't sufficient in sparse graphs. Graph
sparsity caps how different any two paths can be at the semantic level. This
pushed the team toward generation-level diversity.

Source: `experiments/exp-004-edge-disjoint/{README.md, EXPLANATION.md, PLAN.md}`

---

### exp-005 — Path-Grounded Multi-Style Generation

MMR path selection (top-5 from 20 candidates, node-embedding diversity) ×
2 prompt styles (factual vs personal) = 10 candidates per (u, i) pair for
PPO to choose from.

**Dryrun training (Google Local, 5 episodes each):**

| Run | Avg F1 | Sel. entropy | Avg reward |
|---|---|---|---|
| dryrun 1 | 0.197 | 2.299 | 1.682 |
| dryrun 2 | 0.262 | 2.301 | 2.220 |
| dryrun 3 | 0.341 | 2.300 | 2.807 |

**Test eval (100 samples, full Google Local):**

| Method | BERTScore F1 | Median | Max | Min |
|---|---|---|---|---|
| Heuristic (no training) | 0.319 ± 0.068 | 0.316 | 0.486 | 0.154 |
| Single-style PPO | 0.322 ± 0.068 | 0.314 | 0.502 | 0.158 |
| Multi-style PPO | **0.323 ± 0.066** | 0.314 | 0.502 | 0.143 |

**Conclusion:** Multi-style generation achieved stable training with healthy
entropy (~2.3 throughout). Best-of-3 dryrun hit F1 = 0.341, but test eval
showed only marginal improvement over heuristic (+0.004). Key insight:
**reward-signal collapse, not entropy collapse, was the real bottleneck** —
generation diversity helped but still hit a graph-sparsity ceiling.

Source: `experiments/exp-005-multistyle/{PLAN.md, dryrun/*/metrics.json,
test_eval/summary.json}`

---

### exp-006 — Clean Train/Test Split (Headline Run)

Full 300-episode training with proper split (training refs from `trn_5k.pkl`,
entirely disjoint from test set). Numbers identical to the headline table
at the top of this document.

**Google Local (n=1,000):** Multi-style PPO F1 = **0.3306 ± 0.076**, BART
= −3.570, USR = 1.000.

**MovieLens-1M (n=300):** Multi-style PPO F1 = **0.2621 ± 0.070**, BART
= −3.624, USR = 0.997.

**Cross-dataset conclusion:** Multi-style generation enables PPO to learn on
both datasets, but the F1 ceiling follows graph density. Google Local (329k
edges) tops out around 0.333; MovieLens-1M (18 genre intermediaries) tops
out around 0.265. Same code, same algorithm — the −0.065 F1 gap is
**attributable to graph structure, not RL method**.

Source: `experiments/exp-006-proper-split/run_training.sh`,
[`saeideh-kg-approach-experiments/README.md`](saeideh-kg-approach-experiments/README.md)

---

## Cross-experiment takeaways

1. **exp-001**: reward scaling was misdiagnosed as the root cause.
2. **exp-002**: path similarity identified as the real bottleneck (analysis incomplete).
3. **exp-003**: temperature raises entropy but doesn't raise F1 — semantic diversity > structural diversity.
4. **exp-004**: structural diversity insufficient on sparse graphs.
5. **exp-005**: multi-style generation unlocks learning — but hits a sparsity ceiling.
6. **exp-006**: cross-dataset, **graph density caps F1**, not the RL algorithm.

---

## Comparison with our picker approach (same test sets)

On the **same** Google Local test set (note: Saeideh uses n=1,000, we use
n=2,958; both seeded at 42 from XRec `tst.pkl`):

| Approach | F1 on Google | F1 on MovieLens |
|---|---|---|
| Saeideh — KG PPO multi-style | 0.331 | 0.262 |
| Saeideh — heuristic baseline | 0.333 | 0.268 |
| **Ours — LambdaRank picker (Haiku 3 pool)** | **0.500** | **0.329** |
| Ours — best RL variant (Distill A-only) | 0.482 | 0.289 |
| G-Refer 8B (cached paper numbers) | 0.459 | — |
| XRec (cached paper numbers) | 0.386 | — |

Saeideh's approach generates explanations from scratch grounded in KG paths.
Our approach ranks pre-generated LLM candidates with a learned picker. Our
F1 is substantially higher on both datasets; her BARTScore (-3.57 on Google,
-3.62 on MovieLens) is roughly consistent with ours, and her USR ≈ 1.0 is
expected for per-pair generation (no template repetition).

**Direct comparison on a shared test subset is pending** — her 1,000-pair
subset intersects our 2,958-pair subset by construction, so with the picks
already saved in both pipelines we can score the 1,000-pair overlap with
identical BERTScore code if useful for the paper.
