# Native First Adventure v1

Native First Adventure brings the existing relationship-first onboarding contract to the iOS/Expo client without creating a second source of truth.

## Product loop

The native entry path is:

`authenticate -> resolve server Companion state -> choose presentation mode if needed -> create or resolve a real dog relationship -> optional First Adventure context -> Today`

Authentication alone never grants pet-specific dogOS.

## Server-owned routing

The mobile router resolves `GET /companion/state` before opening authenticated product surfaces. The server landing state controls the client:

- `NEEDS_MODE` -> role chooser
- `NEEDS_PET_SETUP` -> First Adventure
- `PET_TODAY` -> Today / Compass / Story / Community
- `COMPANION_TODAY` -> petless-safe Companion experience

If Companion state cannot be verified, pet-specific surfaces stay closed. A locally remembered mode or pet is never sufficient authority.

## Replay-safe account creation

Native registration uses the existing server `registrationKey` contract. The replay identity is stored in SecureStore and retained until both server registration and local access-token persistence succeed.

A lost registration response can therefore retry the exact account transaction and converge on the same canonical user. The client does not invent a refresh-token protocol or resurrect a lost session token.

## Replay-safe first dog

The first durable dog create uses the server `creationKey` contract. The minimal transaction contains:

- dog name,
- optional breed,
- species `DOG`,
- replay identity.

Photos, temperament, mutable profile JSON, and other enrichment are deliberately outside this replay transaction.

### Ambiguous-write airlock

If the client cannot tell whether a pet-create request reached the server, that transaction becomes explicitly ambiguous.

While ambiguous:

- name and breed are frozen,
- the replay identity is retained,
- presentation-mode switching is disabled,
- the user may retry the exact create,
- or re-resolve server Companion state.

The client does not interpret a timeout as proof that no dog exists. Server `PET_TODAY` authority clears stale local retry metadata.

## Optional First Adventure evidence

Once the dog relationship exists, native asks the same bounded First Adventure questions as Web:

- up to three current owner goals,
- realistic time budget,
- realistic effort,
- one dog social-comfort observation.

Native and Web share the same closed answer vocabulary and question IDs.

`Not sure` is explicit uncertainty. Skipping is explicit missing evidence. Neither is treated as dislike, failure, or negative relationship evidence.

Adaptive Profile writes are optional and non-blocking. They cannot:

- gate Today,
- award Bond XP,
- create streak authority,
- manufacture mastery,
- reduce access or relationship status when skipped.

## Petless Companion path

Users are not forced to fabricate a dog. Animal Ally and Foster/Caregiver modes open a petless-safe experience with community, events, account, privacy, and deletion controls while pet-specific Today, Compass, and Story remain closed.

Changing presentation mode never grants access to a pet.

## Qualification

The dedicated Native First Adventure CI lane verifies:

- server-owned landing-state routing,
- fail-closed pet surfaces,
- registration replay wiring,
- pet replay wiring,
- ambiguous-write persistence and airlock behavior,
- native/Web First Adventure ontology parity,
- explicit `NOT_SURE` and `SKIPPED` semantics,
- non-blocking Adaptive Profile writes,
- petless Companion routing,
- canonical formatting,
- native TypeScript,
- zero-warning native lint.

The tranche also remains subject to the repository's broader security, foundation, mobile-convergence, production-config, generated-native/privacy, Expo compatibility, CodeQL, and general CI lanes when their path triggers apply.

## Explicit non-claims

This release does **not** claim:

- a signed iOS archive or IPA,
- TestFlight distribution,
- App Store Connect validation,
- physical-device qualification,
- production APNs delivery,
- production API/Web deployment,
- owner-pilot evidence.

Those are separate evidence boundaries and remain future release gates.
