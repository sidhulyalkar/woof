# Compatibility Promotion Uncertainty Authority

Woof does not promote a learned Compatibility model because one held-out score happens to look better than the deterministic baseline.

The authoritative question is stronger:

> Is the candidate's improvement practically meaningful, stable under relationship-level resampling, calibrated under uncertainty, robust to cold-start cohorts, and operationally safe enough to earn production authority?

This document defines the v2 statistical promotion contract.

## 1. Why point estimates are insufficient

A Brier improvement of `0.006` can mean very different things:

- a stable improvement repeated across many independent owner/pet relationships;
- a handful of unusually favorable outcomes;
- repeated observations from the same relationship that make the nominal row count look larger than the effective evidence base;
- a candidate whose average improves while a cold-start cohort can plausibly regress beyond the product tolerance.

The old promotion gate checked point estimates and minimum row counts. Those remain useful, but they are no longer sufficient for authoritative promotion.

## 2. Paired evidence

Baseline and learned scores are evaluated on the same outcome rows.

For each outcome `i`, define the paired Brier improvement:

`(baseline_i - label_i)^2 - (learned_i - label_i)^2`

Positive values favor the learned candidate.

The bootstrap always resamples the baseline and learned loss for an outcome together. Independent model resampling is prohibited because it would discard the strongest available experimental control: both scorers saw the same held-out event.

## 3. Relationship-cluster bootstrap

Rows are not assumed independent.

The leakage-resistant evaluation dataset already owns canonical relationship identifiers:

- `pair_key`: unordered pet pair;
- `owner_pair_key`: unordered owner pair.

Canonical training prediction files already carry `outcome_id`. The uncertainty evaluator joins predictions back to the exact evaluation split by `outcome_id`, so relationship identity remains evaluation metadata rather than a serving feature.

For authoritative evidence, the split is also the source of truth for the held-out label. The prediction label must agree with the split label for every joined `outcome_id`; missing outcomes, duplicate prediction outcomes, or label disagreement fail closed. A prediction file cannot self-declare its own cluster identities and still qualify for promotion.

Authoritative cluster policy:

| Slice | Bootstrap cluster |
| --- | --- |
| future temporal test | `owner_pair_key` |
| cold-pair test | `pair_key` |
| cold-owner test | `owner_pair_key` |
| safety cohort, when supplied | `owner_pair_key` |

A bootstrap draw samples relationship clusters with replacement and includes every outcome belonging to each sampled cluster.

This intentionally gives repeated observations from one relationship less statistical authority than the same number of observations spread across many relationships.

## 4. Future-test promotion rule

The temporal holdout must still satisfy all point-estimate policies, including minimum rows, Brier improvement, ECE, and AUC non-regression.

In addition, the paired cluster bootstrap must satisfy:

- at least the configured minimum number of relationship clusters;
- at least the configured bootstrap resample count;
- at least the configured confidence level;
- the lower confidence bound for Brier improvement must be at least the practical promotion threshold;
- the upper confidence bound for learned ECE must be at or below the calibration threshold.

The default confidence level is 95%.

The gate therefore does not ask merely whether the candidate probably beats zero improvement. It asks whether the uncertainty-adjusted lower bound still clears Woof's minimum *practical* improvement.

## 5. Cold-start and safety rule

Cold-pair, cold-owner, and optional safety cohorts retain the existing bounded-regression policy.

For each cohort, the bootstrap computes:

`learned Brier - baseline Brier`

The upper confidence bound must remain at or below the configured maximum tolerated regression.

This means a cohort cannot pass merely because its average is acceptable while its uncertainty still admits a materially harmful regression.

AUC non-regression remains a separate point-estimate guard because its role is discrimination, while the bootstrap gate is currently focused on paired probability-quality loss and calibration.

## 6. Evidence provenance

Every authoritative uncertainty slice records:

- prediction file path and SHA-256;
- evaluation split / cluster-source path and SHA-256;
- join key (`outcome_id`);
- explicit confirmation that the prediction label agreed with the split label;
- baseline and learned score columns;
- the policy-approved cluster column for that slice;
- row count;
- cluster count;
- bootstrap method;
- deterministic seed;
- bootstrap resample count;
- confidence level;
- point estimates;
- interval bounds.

The top-level uncertainty policy must explicitly state that paired rows are used, relationship clusters are resampled, cluster identity comes from evaluation splits, and split-label agreement is required.

`promotion_gate.py` independently verifies that uncertainty point estimates reconcile with the aggregate calibration reports. It also rejects missing split hashes, self-declared cluster policy, the wrong cluster dimension, an invalid join key, missing label verification, or score-column drift. A reassuring uncertainty report generated from different or weaker evidence therefore fails closed.

## 7. Statistical receipt v2

Authoritative Compatibility promotion now requires:

`woof-model-promotion-receipt-v2`

The receipt includes an `uncertaintyEvidence` block and an `authoritativeEligible` field.

A receipt can say `promote` only when:

1. aggregate future/cold/safety policy passes;
2. required split-backed uncertainty evidence passes;
3. required shadow telemetry passes;
4. the evaluation was run in authoritative mode.

Research flags may explicitly omit uncertainty or service telemetry, but such a successful run is labeled:

`research_only`

It can never emit `promote`.

## 8. Signed release authority

`attest_promotion.py` refuses to sign a promoted release unless the statistical receipt:

- uses v2 schema;
- says `passed: true`;
- says `decision: promote`;
- says `authoritativeEligible: true`;
- requires uncertainty;
- reports the uncertainty gate passed;
- identifies `woof-compatibility-uncertainty-v1`;
- preserves the paired, split-backed cluster and label-verification policy;
- contains valid prediction and split SHA-256 provenance for future, cold-pair, and cold-owner slices;
- uses `outcome_id` as the evidence join key;
- uses the policy-approved cluster dimension for each slice.

If a safety slice participates in the statistical decision, the signer requires the same provenance for its uncertainty evidence.

A legacy v1 point-estimate receipt, research-only receipt, or fabricated v2-looking receipt without the required provenance cannot be used to bypass the new statistical authority.

The signed release receipt remains bound to the exact model, calibration artifact, feature contract, training manifest, and statistical receipt hash.

Because the statistical receipt carries the hashed uncertainty evidence sources, this creates an authority chain from held-out predictions and evaluation splits through statistical promotion to the signed model release.

## 9. Defaults are policy, not scientific constants

Current defaults are conservative beta release policy:

- future rows: 500;
- cold rows: 75;
- future relationship clusters: 25;
- cold relationship clusters: 10;
- bootstrap resamples: 2,000 minimum;
- confidence level: 95% minimum;
- future Brier improvement lower bound: at least 0.005;
- learned ECE upper bound: at most 0.08;
- cold/safety Brier-regression upper bound: at most 0.01.

These values may evolve as Woof accumulates real beta evidence. Policy changes must be explicit and preserved in the statistical receipt; they must never be silently tuned until a candidate passes.

## 10. What this does not prove

Passing v2 promotion does not prove that a model improves real-world relationships.

It provides stronger E2-E4 evidence that the model:

- improves held-out probability quality beyond a practical floor;
- is not relying on an obviously lucky relationship sample;
- remains bounded under important cold-start cohorts;
- is calibrated within the current product tolerance;
- meets operational shadow constraints.

A controlled online outcome experiment is still required before claiming real product lift. The ML evidence ladder in `docs/ML_SYSTEM.md` remains authoritative on that distinction.
