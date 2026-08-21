# Woof Behavior Vision

## Product promise

Behavior Vision should help an owner answer three different questions without collapsing them into one:

1. **What is visibly happening?**
2. **What patterns are compatible with those observations, and how uncertain are they?**
3. **What tends to help this individual dog in this context?**

The third question is the product advantage. A generic video model can recognize actions. Woof should build an individual longitudinal model from repeated, owner-corrected observations and paired before/after interventions.

Behavior Vision must never claim that a camera can directly read a dog's internal emotional state or intention.

## A crucial example: barking toward another dog

Barking, pulling, pacing, or strong orientation toward another dog is not sufficient evidence that the dog "needs to say hello." Similar outward behavior can occur with excitement, barrier frustration, uncertainty, fear, learned leash behavior, or mixed motivation.

The Woof policy is therefore:

- record objective cues first;
- estimate dimensions such as arousal, body tension, approach movement, avoidance movement, handler engagement, and recovery;
- show multiple compatible hypotheses rather than a single emotion label;
- collect paired observations before and after one low-risk handler change;
- learn which changes are associated with faster recovery for this individual dog;
- never automatically recommend a direct greeting.

Recent canine-frustration research explicitly describes barrier frustration when access to a desired stimulus is blocked by a door or leash, with possible vocalization and lunging. Recent 2026 work on dog-reactive dogs also finds that superficially similar reactive behavior separates into multiple components/subtypes rather than one monolithic condition. These findings support measuring behavior dimensions and context rather than treating barking as a single diagnosis or intention.

References:

- https://pmc.ncbi.nlm.nih.gov/articles/PMC8698056/
- https://www.sciencedirect.com/science/article/abs/pii/S0168159126000523
- https://www.aaha.org/trends-magazine/publications/understanding-reactive-dogs/

## Perception stack

The intended production pipeline is modular:

```text
raw image/video
      │
      ├── object/instance tracking
      │       SAM 2 / equivalent
      │
      ├── body pose + geometry
      │       ViTPose++ AP-10K / SLEAP dog adapter
      │
      ├── action/video representation
      │       Animal-CLIP
      │       EthoCLIP when code/weights are actually available and reviewed
      │
      ├── optional facial movement vocabulary
      │       DogFACS-compatible objective action units
      │
      └── optional audio features
              bark / whine / growl / pant temporal features

             ↓

     temporal observation head
             ↓

observable evidence + calibrated dimensions
             ↓

     Woof individual adapter
             ↓

context baseline + handler-response model
             ↓

       conservative Coach policy
```

No individual perception component is allowed to directly issue training advice.

## Current public model candidates

### SAM 2

SAM 2 provides image/video segmentation and streaming object tracking. In Woof it is useful for maintaining the target dog's identity across frames, separating multiple dogs, and measuring relative geometry. It is not a behavior classifier.

- https://ai.meta.com/research/sam2/

### ViTPose++ + AP-10K

ViTPose++ publishes generic animal-pose results on AP-10K/APT-36K. AP-10K contains 10,015 animal images with keypoints across 54 species. This is useful as a generic quadruped pose prior before dog-specific adaptation.

- https://github.com/ViTAE-Transformer/ViTPose
- https://github.com/AlexTheBad/AP-10K

### SLEAP

SLEAP provides multi-animal pose estimation plus active-learning/annotation workflows and can be trained with relatively few labeled examples. It is especially attractive for dog-specific/domain adaptation when a generic pose model struggles with unusual morphology, fur, camera placement, mobility aids, or home environments.

- https://sleap.ai/

### Animal-CLIP

Animal-CLIP is an open implementation for animal action recognition with pretrained weights advertised by the upstream project. It should enter Woof only as a shadow representation candidate until benchmarked on real dog-owner video and license-reviewed end to end.

- https://github.com/PRIS-CV/Animal-CLIP

### EthoCLIP / AnimalBand

The CVPR 2026 EthoCLIP work reports ontology-enhanced video-language pretraining using AnimalBand, a 74,671-video, 160-behavior resource. This is strategically attractive because its ontology deliberately focuses on visually observable animal behavior rather than unverifiable internal states.

However, the upstream repository currently lists EthoCLIP code/pretrained weights as a TODO. Woof must not make an unavailable artifact a production dependency.

- https://openaccess.thecvf.com/content/CVPR2026/html/Jing_EthoCLIP_Ontology-Enhanced_Video-Language_Pretraining_for_Animal_Behavior_Understanding_CVPR_2026_paper.html
- https://github.com/PRIS-CV/AnimalBand

### DogFACS

DogFACS is valuable because it is explicitly an objective facial-movement coding system. Its maintainers also explicitly state that DogFACS does **not** infer underlying emotion or context. This matches Woof's product policy.

The DogFACS manual is accessible, but associated training/test videos cannot be reused without written permission. Woof may use the scientific vocabulary while respecting media restrictions.

- https://animalfacs.github.io/AnimalFACS/DogFACS

## What gets extracted

Woof's canonical observation contract focuses on evidence that can be inspected and corrected:

### Pose / geometry

- body elongation/compactness;
- spine orientation;
- head direction and changes;
- body orientation relative to handler, dog, exit, or trigger;
- approach and retreat velocity;
- stationary/freezing intervals;
- play-bow-compatible geometry;
- repeated pacing path;
- leash-line geometry when visible;
- interpersonal/inter-dog distance when scale is available.

### Motion

- approach / retreat;
- acceleration bursts;
- pacing;
- jumping/lunging-compatible movement;
- shake-off-compatible motion;
- sniffing-compatible head-down sequences;
- recovery latency after an event.

### Facial/upper-body cues when visible

- head turn;
- eye/head orientation;
- lip/nose movement;
- ear movement/posture only when morphology/viewpoint makes it scorable;
- mouth opening/closing and panting-compatible movement.

Tail/ear posture must be calibrated carefully because morphology, cropping, breed anatomy, docked tails, coat, and camera viewpoint can make generic thresholds misleading.

### Audio, later

Audio should remain a separate evidence stream rather than converting "bark" directly into emotion:

- bark count/rate;
- whine-compatible vocalization;
- growl-compatible low-frequency event;
- pant rhythm;
- onset relative to trigger/handler action.

## Individual Dog State Engine

The personal model intentionally excludes breed as a direct behavior-policy feature. Breed/life-stage knowledge can inform broad context elsewhere in Coach, but direct observations should dominate individualized behavior inference.

### Baselines

For each dog and context, maintain confidence-weighted distributions over:

- arousal;
- body tension;
- social orientation;
- approach tendency;
- avoidance tendency;
- handler engagement;
- environmental engagement;
- recovery.

These are **measurement dimensions**, not personality labels.

### Context keys

A dog's state can differ radically by context. At minimum condition on:

- home / street / park / trail / daycare / vet / training class;
- leash state;
- dog presence;
- approximate dog distance when owner can estimate it;
- familiar vs unfamiliar dog;
- handler action;
- baseline vs intervention vs recovery phase.

Future versions can add weather, time of day, prior exercise load, pain/health changes, household member, familiar location, and known triggers.

## N-of-1 paired experiments

The most valuable personalization loop is a within-dog micro-experiment:

```text
10–20 s baseline clip
        ↓
change ONE low-risk variable
        ↓
10–20 s intervention/recovery clip
        ↓
compare dimensions within the same session
        ↓
repeat on another day
        ↓
learn an individual handler-response association
```

Examples of low-risk variables:

- increase distance;
- add leash slack;
- use one cue instead of repeating cues;
- perform a U-turn;
- scatter/search for food when appropriate;
- walk parallel rather than head-on;
- pause and observe;
- end the interaction.

Woof should **record** owner-selected `allow-greeting` episodes when they happen, but the automated Coach policy must never recommend direct greeting as the next experiment from video inference alone.

## Human-dog co-behavior model

This is the deeper product opportunity.

Instead of treating the dog's behavior as the only target, model transitions like:

```text
trigger appears
     ↓
dog response at t0
     ↓
human action
     ↓
dog response at t1
     ↓
recovery latency
```

Over time Woof can discover statements of the form:

> For this dog, when another dog is visible at moderate distance, tightening the leash and repeating cues has been associated with slower recovery than adding distance and allowing sniffing.

That is useful personalized coaching because it is about a repeated empirical pattern in the pair, not a generic commandment and not an emotion diagnosis.

## Owner correction is a first-class label

After every analysis, ask:

> Did Woof describe the visible behavior correctly?

Owner confirmations increase evidence weight slightly. Owner-rejected observations are excluded from the personal baseline.

Future active learning should prioritize clips where:

- model confidence is low;
- owner disagrees;
- a new context appears;
- the dog's morphology causes pose failures;
- intervention effects disagree across sessions.

Professional trainer/veterinary-behaviorist labels should be represented separately from owner feedback rather than silently merged.

## Privacy

Behavior media is sensitive household/location context even when it does not contain human faces.

Beta policy:

- raw image/video is transient;
- raw media is not placed in the public/social storage path;
- timeline persists derived observations + an irreversible SHA-256 fingerprint;
- owner explicitly controls deletion;
- raw user clips are **not** training data by default;
- future dataset contribution requires separate explicit opt-in and should support face/license-plate/background redaction.

## Safety boundary

Behavior Vision is not appropriate for autonomous exposure experiments involving:

- biting or snapping history;
- serious dog-dog aggression;
- severe fear/panic;
- resource guarding with injury risk;
- suspected pain or sudden behavior change;
- unsafe child/animal interactions.

For these cases Woof should document observable context and route the owner toward a veterinarian or qualified reward-based behavior professional.

AVSAB's current public position continues to recommend reward-based training methods and rejects aversive methods for dog training and behavior modification:

- https://avsab.org/resources/position-statements/

## Evaluation before promotion

A behavior model is not ready because clips look plausible in a demo.

### Perception metrics

- keypoint PCK/AP on dog-specific holdout;
- identity tracking failures;
- temporal boundary F1 for observable actions;
- confidence calibration / ECE;
- per-morphology and per-viewpoint error;
- multi-dog identity swaps;
- low-light and occlusion robustness.

### Behavior contract metrics

- inter-rater agreement against trained human observers for objective cues;
- false certainty rate;
- emotion/intent overclaim rate (target effectively zero);
- abstention quality on unusable clips;
- owner correction rate;
- professional correction rate on an expert-reviewed audit set.

### Personalized-policy metrics

- paired-session sample size;
- stability of intervention-effect estimates across days;
- held-out prediction of recovery direction;
- rate of unsafe/direct-greeting recommendations (must be zero by policy);
- improvement in owner cue repetition, leash tension, recovery time, and dog voluntary engagement where these can be measured safely.

## Long-term moat

The moat is not the foundation model. Public models will improve rapidly.

The durable asset is a consented, structured longitudinal dataset of:

```text
individual pet
+ context
+ objective behavior evidence
+ human action
+ short-term response
+ owner correction
+ professional correction when available
+ longitudinal outcome
```

That dataset makes Woof increasingly useful for the **pair**, not merely better at classifying generic dog videos.
