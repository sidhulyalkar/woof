# dogOS Adaptive Profile Persistence v1

Adaptive Profile is the durable personalization substrate for Adaptive Adventure. It stores what Woof has evidence for about a specific dog-human household pair without turning game rewards, temporary context, or opaque legacy profile JSON into personalization authority.

## Product boundary

The learning unit is a `(householdId, petId)` pair. Profile state helps Woof choose and adapt safe Adventures for that pair. It does not award Bond XP, set levels, unlock chapters, or otherwise own the game economy.

The profile is intentionally sparse. Unknown is a valid state. Missing information does not block play, and the application should ask a question only when resolving that uncertainty is worth the interruption.

## Storage model

Two append-only tables keep durable meaning separate from interaction history:

- `adaptive_profile_evidence` stores versioned profile statements with dimension, subject, state, bounded value, confidence, provenance, source user, and occurrence time.
- `adaptive_profile_question_responses` stores replay-safe question outcomes and cooldown history. A skip can therefore suppress nagging without becoming a negative preference label.

Both tables carry `household_id` and `pet_id` and reference the composite `HouseholdPet` identity. A row cannot claim a dog belongs to a household where that pair does not exist.

`Pet.temperament` remains legacy metadata. It is not Adaptive Profile authority.

## Authorization

Every public profile operation is addressed by household and pet and must call `HouseholdsService.assertHouseholdPetAccessible(userId, householdId, petId)` before reading or writing profile data.

This preserves shared-household behavior without silently mutating or falling back to an unrelated personal household.

## Evidence authority

Profile evidence is append-only. Current state is a deterministic projection.

Authority order is:

1. `OWNER_CORRECTION`
2. `OWNER_EXPLICIT`
3. `HOUSEHOLD_EXPLICIT`
4. `HISTORICAL_QUIZ`
5. `OUTCOME_INFERENCE`

A higher-authority statement outranks a newer lower-authority inference. Within the same authority class, the newest statement wins before confidence or state. This allows an owner to correct or clear an earlier statement without deleting audit history.

The v1 subject classes are `DOG`, `OWNER`, and `PAIR`. A dimension with the wrong subject is ignored by the projection rather than being repaired implicitly.

## Question semantics

Static progressive-profile questions have fixture-locked answer vocabularies and bounded selection counts.

- `ANSWERED` records question history and explicit `KNOWN` evidence for a recognized static profile question.
- `NOT_SURE` records question history and explicit `UNKNOWN` evidence.
- `SKIPPED` records question history only.
- Dynamic gameplay questions may be logged for cooldown and analysis, but they do not automatically write durable profile evidence.

Temporary context such as a difficult day, bad timing, or one tiring session must not silently become a permanent dog preference.

## Owner corrections

Owner corrections are explicit Adaptive Profile mutations. They use `OWNER_CORRECTION` provenance and immediately become the highest-authority evidence for their dimension.

A correction may set a dimension to `KNOWN` with bounded values or to `UNKNOWN` with no value. `SKIP` and `NOT_SURE` are question outcomes, not durable correction values.

## Replay and concurrency behavior

Question responses and corrections use client-supplied mutation identities that are namespaced on the server.

An exact replay is treated as a duplicate. Reusing the same identity with divergent household, pet, user, question, dimension, outcome, state, or value fails closed with a conflict. Unique-constraint races are re-read and checked against the same semantic identity before being accepted as duplicates.

No retry can mint Bond XP because Adaptive Profile has no reward authority.

## Schema version

The initial schema version is `adaptive-profile-v1`. Reads project only evidence from the active schema version. Database dimensions are stored as strings so later versions can add dimensions without destructive migrations, while the v1 API accepts only fixture-locked dimensions.

## Current v1 dimensions

- owner goals
- owner time budget
- owner effort preference
- available environments
- dog energy pattern
- dog social comfort
- dog novelty comfort
- dog reinforcers
- dog obvious dislikes
- pair training experience

Canonical pet facts such as name, species, breed, and birthdate remain on the `Pet` model instead of being copied into profile evidence.

## Progressive-question adapter

The service can return a policy snapshot containing current projected profile evidence and recent question history. `profile-question-policy-v1` remains the sole deterministic authority for deciding which optional question, if any, is worth asking next.

The persistence layer does not rank questions and does not infer that more questions are better.

## Safety and reward separation

Adaptive Profile runtime code must not read or write `Bond XP`, `RewardLedger`, `totalPoints`, or pathway XP. A user can skip, answer `not sure`, or correct Woof without changing progression rewards.

Dog comfort, owner burden, safe stops, and later training outcomes belong in outcome and decision ledgers as evidence. They are not game scores.

## Qualification

The dedicated `dogOS Adaptive Profile Persistence CI` lane:

- restricts database ownership to the profile schema and migration slice;
- rejects game-currency authority in profile runtime code;
- validates and checks canonical Prisma formatting;
- generates Prisma Client;
- checks source formatting;
- lints profile runtime and contracts with zero warnings;
- type-checks the full API; and
- runs focused projection, authorization, replay, skip, unknown, correction, and bounded-payload contracts.

The normal Adventure, root, Foundation, and Action Runtime qualification lanes remain required before merge.
