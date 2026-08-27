# dogOS Behavior Vision Release Authority

## Purpose

Behavior Vision is shadow evidence, not autonomous behavior or safety authority. This contract adds a second boundary: model-derived evidence may influence a dog's longitudinal Behavior Vision baseline only when Woof can identify the exact qualified model release that produced it.

A plausible JSON response is not sufficient evidence of model identity.

## API active release pin

When `BEHAVIOR_VISION_SERVICE_URL` is configured, the API also requires a complete deployment pin:

- `BEHAVIOR_VISION_RELEASE_ID`
- `BEHAVIOR_VISION_MODEL_VERSION`
- `BEHAVIOR_VISION_FEATURE_VERSION`
- `BEHAVIOR_VISION_ARTIFACT_SHA256`

The artifact value is a 64-hex SHA-256 digest of the deployed model artifact or immutable model bundle. Woof does not hard-code a pretend production artifact into source control. The deployment must pin the artifact it actually serves.

Production additionally requires `BEHAVIOR_VISION_SERVICE_TOKEN`.

## Worker-owned release identity

The specialized Behavior Vision worker independently owns the identity of the release it is actually serving:

- `WOOF_BEHAVIOR_RELEASE_ID`
- `WOOF_BEHAVIOR_MODEL_VERSION`
- `WOOF_BEHAVIOR_FEATURE_VERSION`
- `WOOF_BEHAVIOR_ARTIFACT_SHA256`

A partially configured worker release fails closed. The worker does not derive its release identity from request metadata.

`/health` exposes non-secret release metadata and reports the worker unhealthy when release identity is missing or malformed. Credentials are never included in health output.

The pipeline's per-request adapter-composition diagnostic remains separate from release identity. Which adapters happened to contribute to one clip is not allowed to redefine the immutable deployed model release.

## Bilateral request contract

The API sends its `expectedRelease` with every analysis request, including the observation response-contract version.

The worker parses that expectation, independently loads its own release identity, and rejects the request when the two differ. Only then does it run the video pipeline. The worker serializes its own release identity into the response rather than echoing request values.

The specialized service returns:

- the canonical observation schema version;
- release ID;
- model version;
- feature version;
- artifact SHA-256.

These response values are still claims from the service, not authority by themselves.

## API-owned qualification

`BehaviorVisionModelService` compares the returned identity against the API deployment pin. Qualification fails closed when any field differs or when the artifact digest is malformed.

Only after all fields match does the API inject `releaseQualification` into the normalized analysis. Upstream `releaseQualification` content is ignored and overwritten. A model service cannot certify itself merely by placing `qualified: true` in JSON.

Qualification version:

`woof-behavior-release-qualification-v1`

The qualification records:

- release ID;
- model version;
- feature version;
- artifact SHA-256;
- observation response contract;
- API-owned qualified state.

## Persistence

The existing privacy contract is unchanged:

- raw images/video remain transient;
- raw media is not written to Woof object storage by Behavior Vision;
- the timeline persists derived observations and an irreversible media fingerprint;
- qualified model metadata is stored with the derived observation.

A mismatched model response is rejected before the observation can be persisted.

## Active-release-only longitudinal learning

An observation being valid under a previous release does not prove that its numeric dimensions are directly comparable with a newly deployed release. Therefore longitudinal learning is scoped to the API's **currently active qualified release**.

Only exact matches on qualification version, release ID, model version, feature version, artifact SHA-256, and response contract may contribute to:

- the prior individual profile sent into a new model request;
- individualized dimension baselines;
- paired intervention-effect estimates;
- personalization confidence;
- Shadow readiness gates;
- current reviewable Behavior Moments.

Older qualified releases remain in the timeline as auditable history but receive zero current learning weight. A future cross-release calibration bridge must be separately evaluated and explicit; release mixing is never inferred automatically.

If the model service is not configured, there is no active qualified release, so prior model observations do not silently continue driving the current profile.

## Legacy observations

Older observations may not contain `releaseQualification`. They remain available as historical timeline records, but they are unqualified model evidence and also receive zero current learning weight.

Owner confirmation does not retroactively qualify an unknown model artifact. Human feedback can strengthen or reject a measurement from the active qualified release, but it cannot prove which software produced an old one.

## Shadow readiness

Behavior Shadow separates stored history into three buckets:

1. observations from the active qualified release;
2. observations from older qualified releases;
3. legacy/unqualified observations.

Only bucket 1 contributes to usable evidence, owner-review rates, paired sessions, personalization confidence, readiness, and current Behavior Moments.

The evaluation reports total history, all qualified history, active-release observations, inactive qualified observations, legacy unqualified observations, the active release ID, and the qualified release IDs represented in history.

The original zero-authority policy remains literal:

- cannot influence Compatibility;
- cannot mutate canonical pet state;
- cannot make safety decisions;
- promotion remains disabled;
- any promotion requires a separate qualified release.

## Why a SHA-256 is not called a signature

The artifact digest gives a cryptographic identity for the pinned artifact. It does not prove publisher identity by itself and is not described as a digital signature.

Trust comes from the combination of:

1. API deployment-owned release pin;
2. worker deployment-owned release identity;
3. authenticated specialized-service boundary in production;
4. bilateral expectation/worker comparison;
5. exact response/API-pin comparison;
6. API-owned qualification after comparison;
7. active-release-only longitudinal learning;
8. release-specific evaluation and future promotion gates.

If signed artifact manifests are introduced later, they should strengthen this contract rather than replace exact artifact pinning.

## Qualification

`dogOS Behavior Moments Shadow CI` enforces this boundary by checking:

- zero downstream authority remains literal;
- canonical pet mutation remains forbidden;
- Compatibility cannot import Behavior Vision authority;
- individual profiles require qualified evidence;
- active-release-only learning remains explicit;
- the API model request carries `expectedRelease`;
- environment validation keeps the API artifact pin;
- the worker owns and compares its own release environment identity;
- exact release/mismatch/missing-pin/malformed-metadata model tests;
- older-qualified and legacy-unqualified profile/readiness tests;
- Python worker compilation and release contract tests;
- existing Behavior Vision, Web transport, browser, API build, and Web build contracts.
