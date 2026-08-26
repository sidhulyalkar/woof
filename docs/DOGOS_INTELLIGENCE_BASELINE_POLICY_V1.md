# dogOS Intelligence: Baseline Policy v1

`baseline-policy-v1` is the first deterministic longitudinal intelligence policy in dogOS.

It is intentionally a **pure policy oracle**. It does not own persistence, REST APIs, UI, reminders, diagnosis, or model inference. Those layers may consume its output later, but they may not silently redefine its semantics.

## Input authority

The evaluator receives:

- one explicit signal dimension;
- a bounded collection of normalized observations; and
- an explicit `now` timestamp.

Every observation carries a canonical ID/dedupe key, `observedAt`, an upstream-normalized `localDate`, a relative delta bucket from `-2..2`, source type, reliability class, bounded confidence, and optional supersession reference.

The engine never reads the process clock, environment variables, database state, network state, Nest services, Prisma, or an LLM.

Timezone interpretation belongs upstream. `localDate` is already normalized to the household/pet context before policy evaluation. The oracle uses that local-day identity for evidence sufficiency while using `observedAt` for chronology and future-event rejection.

## Policy receipt

The complete v1 receipt is code-owned in `baseline-policy-v1.receipt.ts` and pinned by a SHA-256 mutation-resistance test.

Current constants:

- retention window: 31 days
- recent window: 3 days
- baseline window: preceding 28 days
- learning: at least 2 distinct valid local days
- established baseline: at least 7 distinct baseline local days
- stale gap: more than 7 days without valid evidence
- repeated directional evidence required: 2 samples
- direction threshold: 0.35 delta units
- moderate magnitude threshold: 0.8
- large magnitude threshold: 1.35
- source reliability weights: weak 0.5, standard 1.0, strong 1.5
- confidence agreement bonus: 1 point per sample in the largest fixed recent delta-sign group
- medium state confidence: baseline evidence mass >= 5 and recent evidence score >= 3
- high state confidence: baseline evidence mass >= 9 and recent evidence score >= 6

Any semantic change to these constants or their interpretation requires an explicit policy-version decision. Do not silently update fixtures while continuing to call materially changed behavior `baseline-policy-v1`.

## Evidence canonicalization

Before inference, v1:

1. selects only the requested dimension;
2. excludes invalid timestamps, future timestamps, evidence outside the bounded retention window, invalid local-day strings, zero/non-finite confidence, and confidence above 1;
3. deterministically deduplicates by `dedupeKey`;
4. applies correction/supersession without mutating historical evidence; and
5. orders retained evidence by `observedAt` plus canonical ID.

Missing evidence is not converted to normal evidence.

## Baseline state

Per dimension the oracle emits:

- `INSUFFICIENT`
- `LEARNING`
- `ESTABLISHED`
- `STALE`

An established baseline requires seven distinct valid local days in the baseline window. A previously established baseline becomes stale after the configured evidence gap.

No heterogeneous dimensions are combined into a universal health, wellness, bond, anxiety, or readiness score.

## Change state

For an established, non-stale baseline with recent evidence, v1 compares the reliability/confidence-weighted recent mean against the weighted baseline mean.

Possible directions:

- `NEAR_BASELINE`
- `LOWER`
- `HIGHER`
- `MIXED`
- `UNAVAILABLE`

One isolated directional sample cannot create a directional claim. Opposing recent evidence above the configured conflict ratio produces `MIXED` rather than pretending one source is truth.

Magnitude is deliberately bucketed as `SMALL`, `MODERATE`, `LARGE`, or `UNAVAILABLE`. Internal delta math is never exposed as a clinical score.

## Confidence monotonicity

`confidence` means confidence in the **emitted state classification**. It is not a probability that a dog is healthy and it is not a probability that a directional change exists.

Confidence uses two monotone evidence quantities:

1. retained baseline evidence mass; and
2. a recent evidence score equal to total recent reliability/confidence weight plus an agreement bonus for the largest fixed normalized-delta sign group (`lower`, `usual`, or `higher`).

The fixed groups are important. They are derived from each observation's normalized delta bucket and therefore do not move when the baseline mean changes.

This gives v1 both required properties:

- **removing effective evidence cannot increase confidence**, because total weight and the largest fixed-group sample count can only stay the same or fall;
- **disagreement lowers confidence relative to the same evidence agreeing**, because an agreeing observation contributes both its evidence weight and agreement bonus, while a conflicting observation contributes evidence weight but may not enlarge the dominant agreement group.

Duplicated evidence is deduped and therefore cannot add authority. Replacing a source with a weaker reliability class cannot increase confidence.

`MIXED` remains a structural statement with no directional action authority regardless of confidence. A high-confidence mixed state would mean the system has strong evidence that recent observations conflict, not high confidence in a medical conclusion.

An earlier diagnostic design used a baseline-relative mutually-consistent support group inside confidence. It was rejected before release because deleting one baseline observation could move the weighted baseline mean, reclassify recent evidence, and raise confidence. The exact counterexample is permanently encoded in the tests alongside an adversarial deletion matrix.

Monotonicity applies to the **effective evidence set after deterministic dedupe/supersession resolution**. Reverting a correction is a semantic history rewrite, not simple evidence removal; #33's projection layer must represent those lifecycle semantics explicitly and replayably.

## Explanations

Explanation copy is deterministic and generated from the structured policy state. It is not LLM-generated.

The explanation may say that Woof is still learning, that a baseline is stale, that recent evidence is near baseline, that repeated evidence points higher/lower, or that sources are mixed. It must not infer disease, prescribe treatment, or convert weak evidence into urgency.

## Qualification

The dedicated `dogOS Intelligence Baseline Policy CI` lane proves:

- no database/schema ownership;
- read-only formatting and zero-warning lint;
- no Nest, Prisma, environment, network, randomness, implicit clock, or LLM dependency in production policy files;
- API type-check and build;
- 20 named fixture receipts;
- policy-receipt SHA-256 pinning;
- permutation and byte determinism;
- irrelevant-dimension and out-of-window immunity;
- confidence monotonicity under effective-evidence removal;
- the historical confidence-removal counterexample stays monotone;
- a deterministic adversarial deletion matrix spanning mixed baseline shapes and every recent delta sequence of length 1–3;
- agreement receives more confidence authority than the same amount of conflicting evidence;
- weaker-source monotonicity;
- dedupe authority resistance;
- isolated weak-source abstention;
- no non-finite output or aggregate health/wellness score;
- deterministic explanation/state correspondence; and
- a generous bounded-history performance guard.

## Non-goals

This release does not add:

- Prisma migrations or the `dogos_intelligence` projection;
- Daily Signals endpoints;
- Today UI;
- correction APIs;
- Concierge or Health Lens integration;
- learned anomaly detection;
- diagnosis, disease probability, medication advice, or automatic veterinary escalation.

Those belong to separately qualified successor releases after this policy oracle is accepted.
