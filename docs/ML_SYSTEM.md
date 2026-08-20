# Woof ML System: Evidence, Baselines and Promotion Gates

Woof contains ambitious machine-learning research, including graph and temporal models. This document defines how those artifacts should be interpreted and how a model earns the right to influence the product.

## 1. Product objective

The objective is not generic pet similarity. The eventual model should improve real-world relationship outcomes while respecting safety and user preference.

Primary outcome candidates:

- accepted meetup probability,
- attended meetup probability,
- positive post-meetup feedback,
- repeat meetup probability,
- confirmed relationship probability.

Negative/guardrail outcomes:

- avoid edge,
- block/report,
- immediate conversation abandonment,
- meetup cancellation/no-show,
- safety-related feedback.

## 2. Evidence levels

### E0: architecture only

Code structure or model class exists. No claim about predictive value.

### E1: trains successfully

A model can complete a deterministic training run and produce a versioned artifact. This proves software execution, not product quality.

### E2: offline benchmarked

A model is compared against explicit baselines on a held-out, leakage-controlled dataset with confidence intervals or repeated splits where appropriate.

### E3: integrated

The model is served through the same compatibility contract used by the product, including timeout behavior, versioning, provenance and deterministic fallback.

### E4: calibrated and operationally validated

Latency, error rate, drift, confidence calibration and fallback behavior meet defined thresholds.

### E5: online outcome lift

A controlled experiment shows statistically and practically meaningful improvement in real product outcomes without violating safety or fairness guardrails.

Only E5 should support a claim such as “the model improves matching.”

## 3. Current repository interpretation

The Python `ml/` package contains research and serving infrastructure for graph, similarity, diffusion, temporal and ensemble approaches. A trained artifact or completed architecture is valuable engineering evidence, but it is not equivalent to an end-to-end product result.

The primary NestJS compatibility path historically contained a random placeholder score. The portfolio-hardening pass replaces that behavior with a deterministic baseline so the application has a stable reference point.

The advanced model stack should be treated as **experimental** until it is promoted through the gates above.

## 4. Baseline first

A useful baseline has four properties:

1. deterministic,
2. explainable,
3. cheap,
4. available even when ML infrastructure is down.

The baseline should combine only fields that are actually persisted, such as species and temperament profile, with conservative priors for missing data.

Why this matters:

- regression tests can assert exact behavior,
- model outages have a defined fallback,
- online experiments have a meaningful control,
- complex models must demonstrate incremental value,
- users can receive human-readable reasons.

## 5. Dataset design

A robust compatibility dataset should represent a temporal decision process rather than randomly splitting individual rows.

### Candidate features

Pet features:

- species / breed representation,
- age,
- temperament dimensions,
- activity history,
- social exposure,
- prior relationship graph statistics.

Owner/context features:

- coarse distance,
- schedule overlap,
- preferred activity types,
- meetup venue type,
- time/day context.

Graph features:

- shared neighbors,
- edge recency,
- repeated co-activity,
- local graph density,
- interaction history.

### Labels

Prefer outcome labels ordered by strength:

1. repeat successful meetup,
2. positive attended meetup,
3. attended meetup,
4. accepted proposal,
5. meaningful conversation,
6. weak engagement proxy.

Avoid training the system primarily on easy-to-game clicks when the product thesis is real-world compatibility.

## 6. Split strategy

Random row splits are likely to leak identity and relationship structure.

Useful evaluations include:

- time-based holdout,
- pet-disjoint holdout,
- owner-disjoint holdout,
- neighborhood/geographic holdout,
- cold-start cohort,
- repeat-user cohort.

The exact production split should mirror the decision the model will make at inference time.

## 7. Metrics

Ranking quality:

- NDCG@K,
- MAP@K,
- Recall@K,
- pairwise ranking accuracy.

Probability quality:

- log loss,
- Brier score,
- expected calibration error,
- reliability plots.

Product-proxy quality:

- expected successful meetups per 1,000 recommendations,
- coverage,
- diversity,
- concentration by neighborhood/breed/activity cohort.

Operational quality:

- p50/p95/p99 inference latency,
- timeout rate,
- model load time,
- memory footprint,
- fallback rate.

## 8. Model families in the repository

### Graph Attention Network

Potential strength: relationship structure and neighborhood context.

Primary risk: graph leakage and poor cold-start behavior if node identity or future edges bleed across splits.

### Graph similarity / SimGNN-style models

Potential strength: learned pairwise structural similarity.

Primary risk: complexity may exceed the information content available in early-stage product data.

### Temporal models

Potential strength: activity rhythm and changing energy/context.

Primary risk: time leakage, sparse histories and unnecessary inference cost.

### Ensemble approaches

Potential strength: combine complementary graph, profile and temporal signals.

Primary risk: calibration and debugging complexity. Ensembles should be justified by reproducible incremental lift.

### Diffusion research

Potential strength: uncertainty-aware or generative graph exploration.

Primary risk: unclear product advantage. This should remain a research track until tied to a concrete decision metric.

## 9. Serving contract

The ML service should never return only a score.

Recommended response:

```json
{
  "score": 0.81,
  "confidence": 0.73,
  "source": "gat-v3",
  "factors": {
    "temperament": 0.88,
    "activity": 0.79,
    "graph": 0.74
  },
  "explanations": [
    "similar play energy",
    "compatible recent activity patterns"
  ],
  "modelVersion": "2026-08-20.gat-v3"
}
```

The API layer should validate ranges, enforce a timeout, record provenance and fall back when the response is invalid.

## 10. Promotion gates

A candidate model is promoted only when all relevant gates pass.

### Software gate

- deterministic training seed/config,
- versioned artifact,
- dependency lock,
- automated smoke test,
- schema contract test.

### Offline gate

- beats deterministic baseline,
- survives pet/owner-disjoint evaluation,
- calibrated enough for product display or confidence bucketing,
- no critical cohort regression.

### Serving gate

- p95 latency under the product budget,
- no unacceptable memory/CPU cost,
- timeout and malformed-response tests pass,
- fallback path tested.

### Online gate

- experiment assignment is stable,
- primary outcome improves,
- safety guardrails remain within bounds,
- no unacceptable concentration or exclusion effects.

## 11. Experiment design

A/B test the **decision policy**, not merely the model endpoint.

Example:

- Control: deterministic baseline ranking.
- Treatment A: GAT ranking.
- Treatment B: blended GAT + baseline with calibration.

Measure the full funnel through attended meetup and repeat interaction. If treatment increases clicks but decreases attended meetups, it has failed the product objective.

## 12. Model monitoring

For every production inference, capture non-sensitive metadata sufficient to answer:

- which model generated this score?
- did the system fall back?
- how long did inference take?
- what confidence bucket was shown?
- what downstream outcome occurred?

Monitor drift by feature distributions and outcome residuals rather than only endpoint uptime.

## 13. Research roadmap

The highest-value next work is not another architecture. It is closing the evidence loop:

1. define the stable compatibility response contract,
2. build leakage-resistant evaluation splits,
3. benchmark the deterministic baseline,
4. benchmark existing trained models on the same dataset,
5. calibrate the strongest model,
6. integrate it behind fallback/provenance,
7. run a synthetic end-to-end experiment harness,
8. collect beta outcome labels,
9. run a controlled online test.

This turns the ML directory from an impressive collection of models into a defensible learning system.
