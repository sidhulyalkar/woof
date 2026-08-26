# dogOS Intelligence: Daily Signals Capture v1

`daily-signals-capture-v1` is the canonical write boundary for the first user-authored longitudinal dogOS intelligence signal.

It depends on the qualified `baseline-policy-v1` and Evidence Projection v1 releases. It does not own the Today UI, diagnosis, notification authority, or downstream Concierge/Health Lens behavior.

## Product contract

A Daily Signals check-in records how one accessible dog is doing across a small set of owner-observable dimensions:

- appetite
- energy
- bathroom/routine
- mobility/comfort
- engagement/social comfort
- sleep/rest

Each answered dimension is one of `LESS`, `USUAL`, `MORE`, or `UNSURE`. `UNSURE` is missing evidence and never becomes a baseline-normal observation.

A check-in may also include one optional private note of at most 500 characters. The note remains on the canonical private CareEvent and is never copied into the intelligence projection.

## API

Authenticated capture is exposed as:

`POST /api/v1/intelligence/daily-signals`

The request supplies:

- `householdId`
- `petId`
- optional ISO `observedAt`
- `signals`
- optional `note`

The request does **not** supply a trusted local date or timezone.

## Household and pet authority

Capture resolves the explicit `(user, household, pet)` triple through `HouseholdsService.assertHouseholdPetAccessible`.

This is intentionally stricter than selecting the first household that happens to expose a pet. A pet may be visible in more than one household, and those households may use different clocks. Daily identity therefore belongs to one explicitly authorized household context.

Authorization failures retain non-disclosing not-found behavior.

## Local-day authority

`daily-signals-local-day-v1` owns time normalization.

The household's persisted timezone must be a valid canonical IANA timezone. Newly updated household timezones pass the same validator.

The server derives local day as:

`trusted observed instant -> canonical household IANA zone -> YYYY-MM-DD`

Rules:

1. no client-provided `localDate` is trusted;
2. a missing `observedAt` uses server now;
3. a future client timestamp is clamped to server now **before** local-day derivation;
4. a historical/offline timestamp remains historical and maps to the household-local day on which it occurred;
5. DST gaps and repeated hours are ordinary instants and therefore deterministic;
6. an absent or invalid household timezone fails closed rather than falling back to UTC.

The executable fixture suite includes America/Los_Angeles spring-forward and fall-back boundaries plus Asia/Tokyo and local-midnight cases.

## One logical check-in per household + pet + local day

The logical identity is:

`daily-signals-v1:<householdId>:<petId>:<YYYY-MM-DD>`

Daily Signals uses `CareEventDedupeScope = PET`.

The canonical CareEvent write acquires a PostgreSQL transaction-scoped advisory lock over the pet/event/dedupe identity before lookup or insertion. This is a database authority boundary, not an in-memory mutex, so concurrent requests arriving under different household user IDs or different API processes serialize on the same logical day identity.

The first accepted payload becomes canonical for that day.

- same logical identity + same canonical payload hash -> duplicate success receipt;
- same logical identity + different payload hash -> `409 Conflict`;
- no last-write-wins update is performed.

A later correction must use the explicit correction/supersession path rather than mutating the day's source record.

## Canonical truth before projection

Capture writes one canonical CareEvent first:

- event type `DAILY_SIGNALS_CHECKIN`
- pathway `CARE`
- source `INTELLIGENCE`
- evidence type `SELF_REPORT`
- bounded evidence confidence `0.8`
- `PRIVATE` visibility
- `safetyEligible = false`
- zero Bond XP
- household/timezone/local-day/capture-version/payload-hash context
- structured signal answers and optional private note in outcome

The projection is derived only after the CareEvent transaction commits.

This preserves the authority chain:

`owner capture -> canonical private CareEvent -> normalization receipt -> replayable projection -> baseline-policy-v1`

## Partial-failure repair

Projection failure does not create a second source event on retry.

A retry first resolves the existing pet-scoped CareEvent. If its canonical payload hash matches, the service replays the canonical structured answers through `evidence-normalization-v1` and repairs any missing derived observations idempotently.

Projection identity remains source-event based, so already-created dimensions are duplicates and missing dimensions are inserted.

An orphan CareEvent ID or non-Daily-Signals source fails closed.

## Provenance versus repair authorization

Projection rows currently preserve the original CareEvent actor as their provenance user.

The qualified v1 repair path assumes that source actor remains authorized for the pet. A future release should split **current repair requester authority** from **historical source actor provenance** so a current owner can repair an old check-in after the original household member has left without rewriting source identity.

This release does not weaken pet authorization to paper over that distinction.

## Privacy boundary

The canonical CareEvent may hold the bounded private note because it is the source record.

The intelligence projection must not contain:

- the free-form note;
- raw image/video;
- access or refresh tokens;
- IP address or device fingerprint;
- precise location trail;
- model trace.

Projection context remains bounded and normalization-specific.

## Reward integrity

Daily Signals is data capture, not an XP-farming mechanic.

The canonical CareEvent is recorded with `safetyEligible = false`; the release contract requires zero Bond XP and zero effect on subsequent legitimate reward decay semantics.

## PostgreSQL qualification

The dedicated capture CI must prove against the real database:

- the complete historical migration chain still deploys;
- this slice owns no migration or Prisma schema change;
- IANA/timezone/DST fixtures pass;
- 20 concurrent identical submissions across two authorized household members converge to one canonical CareEvent;
- divergent same-day submissions serialize into one winner plus conflict, never last-write-wins;
- retry after partial projection deletion repairs from the same source event;
- `UNSURE` creates no baseline-authoritative projection row;
- the private note never appears in projection context;
- unrelated users and wrong household/pet combinations fail closed;
- missing/invalid timezone, invalid signals, empty signals, and oversized note fail deterministically;
- canonical event is private and zero reward;
- API type-check and production build pass.

## Not yet owned

This slice does not complete issue #33. Still separate:

- `GET /api/v1/intelligence/daily-signals/today`
- baseline summary endpoint
- dimension trends endpoint
- public correction UI/API
- current-requester repair after historical source actor loses access
- Activity and Coach replay orchestration
- source redaction orchestration
- Today UI integration
- Concierge or Health Lens authority
- learned anomaly models
- diagnosis or disease probability
