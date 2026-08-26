# dogOS Intelligence: Evidence Projection v1

`evidence-normalization-v1` and the `dogos_intelligence.observations` projection form the persistence boundary between canonical dogOS product events and the pure `baseline-policy-v1` oracle.

This release foundation is intentionally **derived and replayable**. It is not a second source-of-truth event ledger.

## Authority boundaries

Canonical product records remain authoritative:

- owner Daily Signals become canonical private `CareEvent` records before projection;
- Activities remain canonical Activity records;
- Coach remains authoritative for its own session/outcome records.

The intelligence projection stores only bounded normalized evidence derived from those sources. Projection rows may be rebuilt, corrected, retracted, or repaired without rewriting canonical source history.

## Release 1 source authority

Only three source classes are qualified:

| Canonical evidence | Projection source | Release 1 authority                                               |
| ------------------ | ----------------- | ----------------------------------------------------------------- |
| `SELF_REPORT`      | `OWNER_CHECKIN`   | `BASELINE_ELIGIBLE` on the six qualified Daily Signals dimensions |
| `ACTIVITY`         | `ACTIVITY`        | `CONTEXT_ONLY`                                                    |
| `COACH`            | `COACHING`        | `CONTEXT_ONLY`                                                    |

`BEHAVIOR_VISION`, `LOCATION`, `MEDIA`, `CLINIC`, and `WEARABLE` fail closed in `evidence-normalization-v1`. They do not receive projection authority merely because an EvidenceType exists elsewhere in dogOS.

### Why Activity and Coach are context-only

Release 1 does not infer that 45 minutes of activity means a dog's energy is `HIGHER`, or that a coaching result directly means mobility is `LOWER`.

That would manufacture health-like relative evidence from heterogeneous raw measurements without a qualified normalization model.

Instead:

- Activity may preserve `ACTIVITY_LOAD` or `RECOVERY_REST_PROXY` numeric measurements;
- Coach may preserve `TRAINING_COMFORT_SUCCESS`;
- those rows keep provenance and can support later context or separately qualified normalization policies;
- they cannot directly enter the six user-facing baseline dimensions in v1.

## Owner Daily Signals normalization

Owner semantic choices are deliberately simple:

- `LESS` → `-1`
- `USUAL` → `0`
- `MORE` → `1`
- `UNSURE` → **no observation**

`UNSURE` is missing evidence, never normal evidence.

The six baseline-eligible dimensions are:

- appetite
- energy
- bathroom/routine
- mobility/comfort
- engagement/social comfort
- sleep/rest

## Projection row contract

`dogos_intelligence.observations` stores bounded operational evidence:

- canonical text ID;
- actor user ID and pet ID;
- dimension and qualified source class;
- canonical source event/record identity when available;
- stable logical source identity;
- `observed_at` separate from `ingested_at`;
- upstream-normalized local date;
- bounded `delta_bucket` only where semantically valid;
- numeric value and explicit unit for measurements;
- confidence and reliability;
- `BASELINE_ELIGIBLE` versus `CONTEXT_ONLY` authority;
- normalization version and bounded reason;
- SHA-256 canonical payload hash;
- bounded structured context;
- supersession and explicit retraction lineage.

It stores no raw camera/video, access or refresh token, IP address, device fingerprint, precise location trail, unbounded note, or model trace.

## Replay identity

A projection identity is unique over:

`pet + dimension + source type + source identity + normalization version`

Replay behavior is strict:

1. same identity + same canonical payload hash → idempotent duplicate receipt;
2. same identity + different canonical payload → hard conflict;
3. concurrent identical inserts converge through the database uniqueness boundary;
4. projection replay may not silently reinterpret history under an existing identity.

This gives replay jobs a deterministic repair contract rather than `ON CONFLICT DO UPDATE` last-write-wins behavior.

## Correction, retraction, and explicit reversion semantics

Corrections are append-only successors.

An observation may have at most one active successor. Once an observation has been superseded, the predecessor remains historical truth but never silently regains effective authority merely because its successor is later retracted.

A correction is allowed only when it preserves pet and dimension identity.

Retraction is explicit and reasoned. Retracting an active correction removes that correction from effective evidence, but **does not resurrect the predecessor**. If an authorized user intentionally wants to return to an earlier semantic value, that decision must be represented by a new explicit successor with its own source identity and provenance.

The effective evidence query excludes:

- retracted rows; and
- every row that has a successor in the append-only lineage, regardless of whether that successor was later retracted.

This makes authority changes auditable. A disappearing correction cannot accidentally reactivate stale evidence.

Canonical source deletion/redaction will later invoke an explicit repair/retraction path rather than relying on physical projection deletion.

## Household authorization

Pet access is resolved through `HouseholdsService.assertPetAccessible` for projection writes, baseline reads, history reads, and retractions.

The same authority is now used by `CareEventsService`, replacing its older owner-only pet assertion. This closes an existing coherence bug where an active household member could legitimately record a shared-pet Activity but the resulting canonical CareEvent emission could fail because a deeper layer required ownership.

Authorization failures retain the canonical `Pet not found` behavior so unrelated users cannot infer pet existence through this surface.

## Bounded reads

Baseline evidence reads are bounded to the `baseline-policy-v1` retained horizon and capped at 512 rows per pet/dimension query.

Effective projection history requests may span at most the same retained horizon and are capped at 512 rows.

Rows are ordered deterministically by `observed_at`, then canonical observation ID.

## Privacy and operational limits

The database and service both enforce bounded structure where practical:

- context ≤ 4096 serialized bytes;
- normalization reason ≤ 512 characters;
- source identity ≤ 256 characters;
- confidence in `(0, 1]`;
- delta bucket from `-2..2`;
- numeric measurements must be finite and carry a unit;
- payload hash is a lowercase SHA-256 hex digest.

## Qualification

The dedicated Evidence Projection CI lane must prove on real PostgreSQL:

- exactly one additive intelligence migration is owned by this release;
- the full historical migration chain still deploys;
- privacy-thin schema constraints remain present;
- only Owner Check-in has Release 1 baseline authority;
- `baseline-policy-v1` remains green;
- normalization receipt and fail-closed source mapping remain green;
- 20 concurrent identical writes create one logical projection observation;
- same identity with changed semantics is rejected;
- competing corrections serialize to one active successor;
- retracting a correction cannot implicitly resurrect its predecessor;
- an intentional return to an earlier semantic value requires a new explicit successor;
- Activity remains context-only;
- household members may operate on shared pets while unrelated users fail closed;
- canonical CareEvent household authority remains race-safe and zero-reward Daily Signals evidence stays private;
- API type-check and production build remain green.

## Not yet included

This foundation does **not** yet complete issue #33.

Still required in later commits/releases before 1B is complete:

- canonical Daily Signals capture service and DTO validation;
- household-local-day idempotency across multiple household members;
- timezone/DST fixtures and one documented local-day policy;
- projection replay from canonical CareEvents after partial failure;
- source repair/redaction orchestration;
- Activity and Coach canonical-record adapters wired into replay;
- public `/api/v1/intelligence/*` endpoints;
- production HTTP retry/isolation proof;
- Docker production qualification.

Those layers should depend on this replay and authority contract rather than redefining it.
