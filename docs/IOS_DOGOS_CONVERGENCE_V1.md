# iOS dogOS Convergence v1

## Purpose

Woof already has a React Native / Expo application. The launch problem is therefore not "build an iOS app from scratch." It is to converge the native client onto the same relationship-first product and authority model already qualified across the modern Web and API surfaces.

This release establishes that convergence boundary without claiming App Store readiness or physical-device qualification.

## Product spine

The native primary navigation is now deliberately small:

1. **Today** — one useful shared action and a tiny outcome loop.
2. **Compass** — recent relationship context, never a universal dog score.
3. **Story** — durable relationship memory assembled from canonical activity/care/media sources.
4. **Community** — supporting people and local context rather than the product's central engagement loop.

Legacy capabilities such as pets, goals, media, events, map, and account/profile remain reachable as contextual tools rather than competing as seven permanent primary tabs.

This mirrors the modern dogOS navigation hierarchy already used by the Web client.

## Native Today

Native Today consumes the canonical Adventure authority:

- `GET /api/v1/adventure/me`
- `POST /api/v1/adventure/quests/:questId/select`
- `POST /api/v1/adventure/quests/:questId/complete`

The screen leads with one recommendation, explains why it surfaced, permits the user to start without remaining on the phone, supports a safe-stop path, and records dog and owner outcomes separately.

Selection persistence is intentionally non-blocking. A transient failure to save the selection must not prevent the real-world activity. Completion remains authoritative because the server owns the resulting CareEvent / Reward Ledger semantics.

Bond XP stays subordinate to relationship feedback and is not treated as recommendation truth.

## Native Compass

Compass reads the canonical Adventure dashboard projection and presents:

- Bond XP as game context;
- recent rhythm;
- pathway coverage;
- current learning summary;
- the server-provided disclaimer.

Coverage is explicitly presented as recent relationship context rather than dog quality, health, obedience, or universal wellness.

## Native Story

Story consumes `GET /api/v1/story` and presents bounded relationship memory from:

- activities;
- canonical CareEvents;
- media memories;
- server-derived milestones.

It does not create a new native event ledger. The Story API remains the read authority.

## Native Daily Signals

Daily Signals is now available as an in-the-moment native capture surface.

The client first resolves current authorized household/pet context through:

- `GET /api/v1/households/me`

and then records through:

- `POST /api/v1/intelligence/daily-signals`

Important semantics remain server-owned:

- explicit household + pet identity;
- household IANA timezone;
- one logical check-in per household + pet + local day;
- replay-safe duplicate handling;
- divergent same-day payloads fail closed with `409 Conflict` rather than last-write-wins;
- `UNSURE` is missing evidence;
- private note remains on the canonical CareEvent and is not projected into the longitudinal intelligence read model;
- zero Bond XP for data-entry volume.

The native screen does not claim baseline/trend/correction behavior that the backend has not yet exposed as public APIs.

## Community truth cleanup

The historical mobile Feed contained navigation to unregistered `PostDetail` and `CreatePost` routes. Those ghost actions are removed in this tranche rather than presenting controls that cannot reach owned product behavior.

Community currently supports the maintained feed/reaction path plus contextual entry to Events and Nearby/Map. A richer relationship-first social release remains separate.

## Authority rules

This release preserves the current dogOS laws:

- server authority beats presentation state;
- client state never manufactures pet/household authority;
- game reward is not model outcome;
- safe stop is valid outcome evidence;
- missing/unknown data remains missing/unknown;
- mobile does not invent refresh-token, offline mutation, medical, or social protocols the server does not own;
- Web/API/mobile may project the same canonical truth differently but do not create competing truth ledgers.

## Qualification in this release

`dogOS Mobile Convergence CI` owns:

- fail-closed source assertions for the four-part native spine;
- canonical API path assertions;
- absence of the retired phantom Community routes;
- relationship-first Today markers;
- Daily Signals safety/authority markers;
- committed formatting;
- full mobile TypeScript checking;
- zero-warning Expo lint.

This is repository qualification only.

## Explicit non-claims

This release does **not** claim:

- physical iPhone execution;
- TestFlight upload;
- App Store approval;
- final EAS project/signing authority;
- APNs/native Push delivery;
- background/offline mutation replay;
- native Health Lens parity;
- native caregiver parity;
- full First Adventure onboarding parity;
- broad accessibility/device matrix evidence;
- production API connectivity from a signed iOS binary;
- App Privacy answers or privacy-manifest completion;
- pilot validation.

## Next release train

The recommended launch sequence after this convergence boundary is:

1. **Native session reality** — exact server session/logout/logout-all behavior, cold-start restore, expired/revoked-session tests.
2. **First Adventure parity** — create/choose pet and reach a useful Today recommendation without legacy quiz pressure.
3. **Native core-loop E2E** — Today -> Adventure -> dog/owner outcome -> Story on simulator, then physical TestFlight.
4. **iOS production configuration** — real EAS project, production API environment, signing, release identity, permission copy, deep-link decision.
5. **Native notification authority** — only after the multi-device/native notification lifecycle is explicitly designed and qualified.
6. **App Store package** — current Apple SDK/toolchain requirement, privacy disclosures, account deletion, support/privacy URLs, review credentials, metadata, screenshots, age rating, export compliance.
7. **External pilot** — small owner cohort plus trainer/veterinary advisor feedback using the same measurement contract as public-beta gate #79.

## Exit condition

This convergence release is complete when the repository can truthfully say:

> The native Woof client now expresses the same Today -> context -> Story relationship loop as the modern dogOS platform and consumes canonical server authority rather than the historical mobile product hierarchy.

It is a prerequisite for iOS launch, not the launch itself.
