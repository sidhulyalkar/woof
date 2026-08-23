# dogOS Concierge v1

Concierge is a read-through reasoning and presentation layer for **what matters today**. It does not become another dog database, another reward engine, or another autonomous agent.

## Product contract

Concierge composes existing dogOS evidence from:

- Adventure: current pet, quest ranking, and quest rationale
- CareEvents: recent explicit dog/owner outcome feedback
- Autopilot: scheduled care reminders and conservative non-diagnostic check-ins
- Connectors: provider connection readiness only

The result is a short Today briefing with transparent reasons and source evidence.

## No new source of truth

v1 adds no database migration and no Concierge persistence table.

`GET /concierge/today` is read-only. Concierge does not own endpoints for create, update, delete, completion, acknowledgement, connection, ordering, medication changes, or preference mutation. A user follows an action into the owning feature when they want to change something.

## Pacing, not mood inference

Concierge does not infer mood, fatigue, illness, pain, stress, or any other medical/psychological state.

The only v1 pacing adaptation is deterministic and temporary:

- if a recent explicit owner outcome says `a_lot_today`, or
- if a recent explicit dog outcome says `not_their_thing`, or
- if a recent Adventure used the safe opt-out,

then Concierge marks the next briefing `GENTLE` for at most 72 hours and shows the exact CareEvent evidence responsible for the choice.

Older feedback does not permanently suppress future suggestions.

## Care preparation

A scheduled Autopilot reminder due within 72 hours may appear as a `CARE_PREP` suggestion.

For medication reminders Concierge may repeat the user-authored reminder title and due time, but it explicitly defers to veterinarian or medication-label instructions. It does not calculate or transform dose, frequency, contraindication, or prescription information.

## Tracker check-ins

Autopilot signals may be passed through as `CHECK_IN` suggestions only when the signal belongs to the selected pet. Their source remains Autopilot and the reason explicitly identifies them as non-diagnostic tracker context.

Concierge does not reinterpret wearable metrics into a diagnosis or health score.

## Connector context

Concierge can surface `REAUTH_REQUIRED` as low-priority connection context. It never treats stale or unavailable provider access as current evidence.

Provider data still enters dogOS only through Connectors and the owning domain importer. Concierge cannot import provider payloads itself.

## Weather

No verified live-weather transport is configured in v1.

The API therefore returns:

```text
weather.status = NOT_CONFIGURED
weather.live = false
```

The UI says that live weather is not connected. It must not synthesize, cache, guess, or imply current weather conditions.

A future weather transport should be added behind its own provenance, freshness, location-consent, retention, failure, and deletion contract before Concierge may use it.

## Explainability

Every surfaced Concierge suggestion contains:

- a stable kind
- priority
- title/body
- a plain-language `reason`
- one or more `evidence` entries naming the owning dogOS source
- an optional navigation action
- `suggestionOnly=true`

The top Adventure quest also retains Adventure's existing rationale rather than inventing a second ranking explanation.

## Today UI

Concierge is embedded above the existing Adventure deck on `/`.

It does not add another bottom-navigation item. The card shows:

- briefing summary
- normal vs gentle pace
- why the current top quest leads the deck
- at most three suggestions
- expandable “Why this is here” evidence
- explicit no-live-weather / no-autonomous-action disclosure

The component shares the existing `['adventure', 'me']` React Query cache. When Adventure regenerates after explicit outcome feedback, the Concierge query key changes with the new Adventure `generatedAt`, causing the briefing to refresh without polling.

## Feature flag

`ENABLE_DOGOS_CONCIERGE`

- defaults on outside production unless explicitly `false`
- defaults off in production unless explicitly `true`
- disabled API requests fail closed with 404

## Hard boundaries

Concierge v1 must not:

- write canonical Pet, Activity, CareEvent, MediaAsset, RewardLedger, reminder, Story, or connector state
- expose POST/PUT/PATCH/DELETE Concierge routes
- import provider payloads
- diagnose or score health
- calculate medication doses or prescriptions
- claim live weather without a verified provider
- purchase products or autonomously contact a provider
- persist inferred life changes

Any future persistent life-change adaptation requires an explicit user confirmation flow in the owning domain.

## Release qualification

Phase D is eligible to leave draft only when one exact head passes all seven lanes:

1. dogOS Concierge CI
2. dogOS Connectors CI
3. dogOS Our Story CI
4. dogOS Autopilot CI
5. dogOS Foundation CI
6. Adventure System CI
7. root CI

The dedicated Concierge lane owns its formatter/lint/type/test/build surface and has tripwires for write endpoints and persistence access. Root CI independently validates full monorepo static quality, backend tests, Chromium contracts, and production builds.
