# Woof ML public priors and synthetic-data strategy

Status: beta research architecture, not a claim that external datasets are approved for production use.

Woof's learned system should answer a narrow product question: **given what Woof knows about two individual pets, their context, and outcomes observed before prediction time, how likely is a low-stress, mutually useful interaction?** Public animal datasets can make the upstream representation richer, but they do not provide that outcome label by themselves.

## The three-layer model

```text
video / pose / activity / owner observations
                 │
                 ▼
       1. perception encoder
  "what is the animal doing?"
                 │
                 ▼
       2. behavior state model
 energy • sociability • caution
 excitability • trainability • social risk
                 │
       + context + prior outcomes
                 ▼
       3. compatibility outcome model
 calibrated P(low-stress positive outcome)
                 │
                 ▼
 deterministic baseline / learned shadow / promoted scorer
```

This separation is a safety property. A video model recognizing play posture is not allowed to jump directly to “safe meetup.” The final mapping must be calibrated against outcome data collected for that purpose.

## Public priors we can responsibly exploit

The machine-readable source registry lives in [`ml/public_sources.json`](../ml/public_sources.json). Nothing in the registry is downloaded automatically. Every source remains blocked on its own access and rights review before training.

### Dog Aging Project

Source: https://data.dogagingproject.org/

The 2025 curated release summarizes tens of thousands of de-identified dogs. HLES contains more than 200 questions across lifestyle, environment, behavior and health. Its behavior survey includes excitability, aggression, fear/anxiety, attachment and trainability.

**Best Woof use:** population-level normalization and behavior-state priors. For example, Woof can test whether its owner-entered behavior vectors have implausible distributions or whether a learned state encoder shifts sharply by age or cohort.

**Do not use as:** pair compatibility ground truth. DAP is not a dataset of Woof-style meetup outcomes. Access currently goes through the project's data-access workflow and Terra, so the exact data-use agreement and intended application must be reviewed before any training run.

### Animal Kingdom

Source: https://sutdcv.github.io/Animal-Kingdom/

Animal Kingdom reports 50 hours of annotated video, 30K fine-grained action sequences, 33K pose frames and 850 species.

**Best Woof use:** generic animal behavior representation pretraining, multi-label action auxiliaries and pose/action robustness. Fine-tuning should be dog-heavy before any feature enters the behavior-state model.

**Do not use as:** evidence that two pets are socially compatible.

### SyDog-Video

Source: https://link.springer.com/article/10.1007/s11263-023-01946-z

SyDog-Video reports 500 synthetic dog videos of 175 frames each, 87,500 frames total, with 33 keypoints plus bounding-box and segmentation supervision. It randomizes lighting, backgrounds, camera parameters, appearance and pose. The paper reports that synthetic pretraining was useful when later adapting to a small real dog-video set.

**Best Woof use:** dog pose/video pretraining, temporal occlusion robustness and a blueprint for our own controlled domain randomization.

**Important limit:** evidence that synthetic pose data transfers does not imply synthetic compatibility outcomes transfer. Woof synthetic pair outcomes remain training augmentation only and never enter final promotion holdouts.

### DECADE

Source: https://openaccess.thecvf.com/content_cvpr_2018/html/Ehsani_Who_Let_the_CVPR_2018_paper.html

DECADE pairs egocentric dog video with movement/IMU supervision.

**Best Woof use:** motion and wearable representation learning, especially if Woof later accepts opt-in collar IMU/activity signals. It offers a more relevant auxiliary target than generic image classification because the representation is grounded in dog movement.

### Animal Pose

Source: https://openaccess.thecvf.com/content_ICCV_2019/html/Cao_Cross-Domain_Adaptation_for_Animal_Pose_Estimation_ICCV_2019_paper.html

The work focuses on cross-domain animal pose learning with weak/semi-supervision and confidence-gated pseudo-label refinement.

**Best Woof use:** pose warm starts and dog-domain adaptation. High-confidence pseudo-labels may enrich the perception encoder, but pseudo-labels cannot be promoted into social or safety outcomes.

### EthoCLIP / AnimalBand

Source: https://openaccess.thecvf.com/content/CVPR2026/html/Jing_EthoCLIP_Ontology-Enhanced_Video-Language_Pretraining_for_Animal_Behavior_Understanding_CVPR_2026_paper.html

The CVPR 2026 paper reports AnimalBand with 74,671 animal behavior videos and an ontology-enhanced contrastive video-language model.

**Best Woof use:** a candidate behavior encoder and ontology-alignment teacher once the current code/data/checkpoint release and usage rights are verified. We should distill relevant behavior dimensions into a smaller encoder rather than place a large video-language model in Woof's latency-critical compatibility path.

### Animal-CoT

Source: https://github.com/WesLee88524/Animal-CoT

The repository reports 8,220 structured video/text reasoning pairs across CBVD-L and KABR-L and a Qwen2.5-VL-7B-Instruct LoRA recipe. Its annotations decompose behavior recognition into pose, context, hypotheses and verification.

**Best Woof use:** teacher-generated structured behavior annotations and explanation-schema research. A compact student can learn canonical behavior states from richer teacher features.

**Rights caveat:** the project repository is Apache-2.0, but underlying CBVD/KABR media have their own provenance and terms. We review each layer independently.

## Model-development ladder

### Stage P0: deterministic product baseline

`behavior-outcome-baseline-v2` remains the product safety floor. It provides an interpretable target to beat and remains available on every ML timeout, invalid response or unapproved model.

### Stage P1: canonical tabular outcome model

Train a small, order-invariant model on:

- individual Woof behavior state vectors
- pairwise behavior gaps and means
- maximum/mean social-risk signal
- life-stage distance
- prior outcomes available strictly before prediction time
- interaction context that is known before the meetup

The first learned candidate should be intentionally boring: regularized logistic regression plus a shallow gradient booster, blended and calibrated on later validation data. This gives us feature audits, CPU inference, stable latency and an intelligible benchmark before multimodal complexity.

Implementation: [`ml/training/train_canonical_compatibility.py`](../ml/training/train_canonical_compatibility.py).

### Stage P2: perception-pretrained behavior state

Add dog-centric video/pose/activity features behind the same canonical behavior dimensions. Candidate teachers include SyDog-Video/Animal Pose/Animal Kingdom, with DECADE for motion and EthoCLIP/Animal-CoT once artifact/rights checks pass.

The compatibility model receives the **state vector and its uncertainty**, not raw video embeddings. That makes missing-video fallback natural and keeps explainability anchored to product concepts.

### Stage P3: multimodal temporal state

For owners who explicitly opt in, incorporate longitudinal activity/video/wearable observations. A temporal model should estimate state and uncertainty over time. It should not infer stable personality from one clip.

### Stage P4: real-outcome adaptation

As beta outcomes accumulate, retrain with global temporal splits. New data can update calibration more frequently than representation weights. This reduces catastrophic drift and lets Woof learn the product-specific exchange rate between upstream representation improvements and real meetup outcomes.

## Synthetic simulation policy

Synthetic generation is valuable for combinations that real beta data will initially undersample:

- high caution + high excitability introductions
- asymmetric energy levels
- strong social-risk signals
- noisy or sparse owner observations
- novel environments
- high crowd/resource pressure
- previously successful pairs encountering changed context

The simulator at [`ml/simulation/generate_pair_scenarios.py`](../ml/simulation/generate_pair_scenarios.py) is intentionally causal-ish rather than a lookup-table label generator. It samples individual latent traits, environmental context and observation noise, then generates outcomes from those variables. It records `data_source=synthetic`, simulation version and a reduced sample weight.

Synthetic rules:

1. Synthetic rows may augment **training only**.
2. Synthetic provenance is never removed.
3. Final validation/test/cold-owner/cold-pair promotion slices must be real or explicitly designated non-synthetic research data.
4. Synthetic influence has a configurable weight cap.
5. Safety-regime oversampling is allowed, but the deployment decision is evaluated on real safety slices.
6. Synthetic generation must not include protected human attributes or attempt to infer sensitive owner characteristics.
7. A model that improves only on synthetic data does not advance a maturity level.

## Leakage-resistant evaluation

The existing `ml/evaluation/build_compatibility_dataset.py` uses global time ordering and computes pair-history features before incorporating the current row. Promotion additionally examines:

- full future test
- cold-pair test
- cold-owner test
- high social-risk / safety slice when sufficient real rows exist
- Brier score and expected calibration error
- ROC-AUC / PR-AUC as secondary discrimination metrics
- p95 learned-model latency
- fallback rate
- minimum shadow sample count

A model is not promoted because its average AUC is impressive. It must remain calibrated and useful when encountering unfamiliar owners/pairs and must not regress the safety slice.

## Promotion policy

[`ml/evaluation/promotion_gate.py`](../ml/evaluation/promotion_gate.py) turns promotion into a reproducible decision artifact. Default beta thresholds are policy defaults, not scientific constants. They require, among other things:

- enough future labeled rows to make a decision
- positive Brier-score improvement over baseline
- bounded ECE
- no material cold-pair/cold-owner regression
- sufficient shadow traffic
- bounded p95 latency
- low fallback rate

The output is a JSON receipt containing every criterion and failure reason. `ML_COMPATIBILITY_MODE=authoritative` should only be set from a model release that has a passing receipt tied to the exact model/version/data hashes.

## What we should not do

- Train a giant VLM and call its textual explanation a compatibility score.
- Treat breed as a dominant behavioral proxy.
- Let public population data leak into the final real Woof outcome holdout.
- Mix future meetups into past pair-history features.
- hide synthetic provenance.
- optimize feed clicks as the model objective.
- interpret confidence as safety certification.
- promote a model because it beats the baseline only on familiar pairs.

## Practical next experiments

1. **Canonical CPU benchmark:** train the tabular candidate on the seeded temporal dataset and quantify the amount of synthetic augmentation that helps before calibration worsens.
2. **State-distillation benchmark:** use a permitted animal behavior encoder as a teacher and predict Woof's six behavior dimensions from dog clips. Measure per-dimension calibration and uncertainty, including unseen breeds.
3. **Pose + motion ablation:** compare no perception features vs pose summary vs temporal pose vs optional IMU summary. Ask whether each adds outcome lift after owner-entered behavior is already present.
4. **Cold-owner transfer:** pretrain state encoders publicly, but train compatibility only on Woof outcomes. This cleanly tests whether public representation knowledge improves a genuinely new user/pet.
5. **Counterfactual context:** for the same pair, vary environment structure/crowding/resource pressure in simulation. Verify that the learned candidate responds monotonically in sensible directions without overpowering individual evidence.
6. **Calibration-first online shadow:** send no user-facing score changes at first. Compare baseline and candidate against later outcomes, then recalibrate before considering ranking changes.

The goal is not “more ML.” It is **more useful evidence per prediction, with less fragility and clearer reasons to trust or reject the score**.
