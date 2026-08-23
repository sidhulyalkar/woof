# dogOS Behavior Moments — Shadow v1

Behavior Moments is the evidence-gathering stage for pet video analysis. This release deliberately does **not** promote model output into canonical pet truth, compatibility authority, or safety decisions.

## Product contract

Raw image/video bytes are processed transiently by the existing Behavior Vision path. Woof stores derived observation telemetry plus a content fingerprint, not the raw media. Audio stays opt-in per observation.

Timestamped model evidence is grouped into bounded **Behavior Moments** so owners and developers can review the portions of a clip that produced evidence. A moment is an index into derived evidence, not stored video and not a claim about a dog's internal emotional state.

The shadow snapshot reports:

- observation volume and usable-observation rate
- owner-confirmed, rejected, and unreviewed observations
- owner confirmation rate
- context breadth
- paired baseline/recovery session count
- model versions represented in the evidence
- timestamped reviewable Behavior Moments
- whether the evidence has met research-readiness thresholds

## Hard authority boundary

`woof-behavior-shadow-v1` always reports:

- `canInfluenceCompatibility: false`
- `canMutateCanonicalPetState: false`
- `canMakeSafetyDecision: false`
- `promotionEnabled: false`
- `promotionRequiresSeparateQualifiedRelease: true`

Meeting every evidence-readiness gate does **not** change those values. Promotion requires a separate release, explicit product review, replay evidence, and its own qualification graph.

## Evidence-readiness gates

The current research gates are intentionally conservative and are not product promises:

- at least 20 usable observations
- at least 10 owner-reviewed observations
- at least 80% owner confirmation among reviewed observations
- at least 3 observed contexts
- at least 5 paired baseline/intervention-or-recovery sessions

Passing these gates means only that the evidence stream is substantial enough to evaluate. It does not grant production authority.

## Safety and interpretation

Behavior Moments may describe visible pose, movement, orientation, interaction, recovery, and opted-in audio timing. They must not be presented as definitive emotion, intent, diagnosis, pain detection, aggression diagnosis, or permission for direct dog-to-dog greeting.

Owner corrections outrank unconfirmed model evidence for personalization evaluation. Rejected observations remain auditable but do not count as usable evidence.

## Release qualification

The dedicated Behavior Moments CI lane must prove:

1. release files are formatted and zero-warning lint clean;
2. API and web TypeScript checks pass;
3. shadow service unit contracts pass;
4. existing Behavior Vision privacy/safety contracts remain green;
5. production Behavior Vision code contains no canonical pet mutation;
6. compatibility code contains no Behavior Vision/Behavior Moments dependency;
7. the zero-authority policy flags remain literal and executable;
8. API and web production builds pass.

This lane is additive to inherited repository qualification. It does not replace root CI or earlier dogOS release lanes.
