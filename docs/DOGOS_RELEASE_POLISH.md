# dogOS Integration & Release Polish v1

## Purpose

The dogOS foundation is now layered and qualified. This release is not another subsystem. It is a product-integration pass that makes the existing system behave like one coherent consumer product under realistic household conditions.

The target user loop is intentionally short:

1. open Today,
2. immediately know which dog the context belongs to,
3. understand what matters now,
4. take or log one useful action,
5. see that action become canonical history that can inform Adventure, Concierge, and Our Story.

## Release principles

### The active dog is never implicit

Multi-dog households must not silently inherit whichever database record happens to sort first.

The web client now uses a deterministic active-dog context:

- explicit `?pet=<id>` URL context wins,
- otherwise the last user-selected dog is restored from local storage,
- otherwise the existing server-owned fallback is used,
- switching dogs updates the URL and the shared dogOS query surfaces.

Today exposes the switcher only when the user owns more than one dog. The same selection drives Adventure and Concierge. Activity uses the same active-dog context and lets the user switch without entering a separate settings flow.

### No production-facing demo history

The old `/activity` page displayed hard-coded walks, coordinates, notes, and a playdate. That is unacceptable in a deployable product because it makes a real account appear to have events that never happened.

The Activity surface now reads only authenticated, household-visible canonical Activities from the API. If that read fails, the UI shows a failure state. It never substitutes mock history.

### Manual logging records only what the user actually supplied

Quick Log is deliberately narrow. The user chooses:

- the dog,
- an activity type,
- an approximate duration.

The client writes a completed Activity with those values. It does not invent:

- a route,
- location points,
- distance,
- pace,
- physiological measurements,
- Bond XP or any other reward.

Reward policy remains server-owned. A completed Activity can emit canonical CareEvents through the already-qualified Activity service, so downstream dogOS layers can react without creating a parallel history.

### Day 1 has a real activation path

`/pets/new` creates the first owned dog with the minimum useful identity fields. Name is required; breed, birthday, and sex are optional. The client does not seed temperament, vaccination, health, or behavioral claims.

After creation, the new dog becomes the active context and the user returns to Today with the pet ID in the URL.

### Long-lived accounts stay bounded

Activity history is paginated in 20-record pages and loaded incrementally. The API already caps each page at 100 records.

Our Story retains its existing bounded historical scan and coverage disclosure. This release must not convert either surface into an unbounded load of a multi-year account.

### Degraded states remain truthful

- pet-list failure never guesses a dog,
- Activity failure never substitutes demo data,
- zero-dog state routes to real onboarding,
- Concierge continues to fail closed when unavailable,
- weather remains explicitly not configured until a verified transport exists,
- connector degradation remains local to the connector.

## Product hierarchy

### Today

Today remains the primary surface. Concierge explains current context, then Adventure offers the ranked activity deck. The dog switcher lives inside the daily context so the user can immediately see whose day they are viewing.

### Activity

Activity is the canonical recent-history and low-friction manual-entry surface. It is no longer a demo dashboard or pseudo-live tracker.

A future live-walk experience must use a real tracking transport and explicit location consent. It must not revive the old simulated route behavior.

### Our Story

Our Story remains the emotional chronology. Activity does not duplicate Story curation; it supplies canonical events that Story can compose.

## Release invariants

The dedicated release-polish CI lane must prove all of the following on the exact release candidate:

1. No database schema or migration changes are introduced by this integration pass.
2. Release-owned files are canonically formatted and pass zero-warning web lint/type-check.
3. Production Activity code contains none of the retired mock-history signatures.
4. Quick Log does not send route/location/distance or client-controlled reward fields.
5. Active-dog URL context overrides stale remembered context.
6. Adventure falls back to the active dog when no explicit pet is passed.
7. Owned-pet reads use `/pets/me` rather than arbitrary owner IDs.
8. First-dog creation does not invent temperament or vaccination state.
9. Activity reads are pet-scoped and paginated.
10. Chromium proves one household can switch between two dogs without Concierge/Adventure disagreement.
11. Chromium proves canonical Activity history changes with the selected dog.
12. Chromium proves a manual Activity write targets the selected dog and contains no fake route/reward fields.
13. Chromium proves first-dog creation produces a usable Today state for that newly created dog.
14. The web production build succeeds.

The release-polish lane is additive. Foundation, Adventure, Autopilot, Our Story, Connectors, Concierge, and root CI remain mandatory inherited qualification lanes before merge.
