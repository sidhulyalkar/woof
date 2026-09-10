# Expedition Authority v1

## North star

Expeditions make safe, useful participation feel like a shared world gradually revealing itself.

The human gets the cooperative game. The dog keeps the right to have an ordinary day.

This release is intentionally **server authority only**. It does not ship a native map, world animation, confetti loop, or client-side progress button. Those presentation layers can come later without changing what counts.

## Product taxonomy

Woof keeps its game systems separate on purpose:

- **Adventure Trail** is personal relationship discovery and chapter presentation.
- **Skillcraft** is human learning practice with server-scored exercises.
- **Story** archives meaningful memories and discoveries.
- **Social Adventure League** is optional bounded human-side comparison.
- **Expeditions** are cooperative, non-ranking shared progress.

Expedition progress never becomes Bond XP, pet proficiency, health state, or League rank.

## Scope authority

There are two explicit scopes.

### Global

`GET /api/v1/expeditions/global`

Global Expeditions aggregate bounded receipts across canonical eligible evidence in Woof. They do not depend on Pack membership or geography.

### Pack

`GET /api/v1/expeditions/packs/:packId`

Pack Expeditions require a current server-confirmed `ACTIVE` row in `dogos_social.pack_memberships` for the requesting user. Each contributing user is independently joined through that same membership authority.

Only evidence at or after the member's current `joined_at` may first become Pack progress. A public Pack listing, region label, cached client `joined` value, or client-submitted Pack ID is not membership authority.

Leaving a Pack later does not rewrite already-issued historical contribution receipts. A later rejoin establishes a new `joined_at` boundary for newly materialized evidence.

Pack Expedition authority deliberately does **not** depend on `regionKey`. Structured coarse-locality authority is a separate pre-public-beta problem tracked in #139. Expedition work must not turn the current free-form locality field into trusted geography by implication.

## No client contribution endpoint

Authority v1 exposes read endpoints only.

There is no API equivalent of:

```text
POST /expeditions/progress { points: 1 }
```

A read reconciles bounded immutable receipts from already-existing canonical server evidence and then projects aggregates from those receipts. The client cannot submit a point amount, pathway, pet performance value, score magnitude, or completion flag.

A later event-driven materializer may replace read-time reconciliation for scale. That optimization must preserve the same receipt identity and eligibility contract.

## Canonical evidence

Initial evidence is narrowly whitelisted.

### Adventure evidence

Eligible CareEvents must:

- exist in canonical `public.care_events`;
- come from `QUEST_ENGINE`;
- have an event type beginning with `QUEST_`;
- fall inside the explicit current Expedition season;
- use exactly one eligible pathway: `EXPLORE`, `ENRICH`, or `RECOVER`.

`CARE` is structurally excluded from receipt schema and materialization SQL.

### Human Skill evidence

Eligible Human Skill evidence must be a completed server-issued row in `dogos_social.human_skill_attempts` for one of the four canonical rooms:

- `MAKE_IT_EASIER`
- `CATCH_THE_GOOD`
- `PAIRING_LAB`
- `MARKER_TIMING`

Only completion breadth matters. Practice score magnitude, correctness magnitude, response details, and Marker Timing milliseconds do not enter Expedition arithmetic or receipt identity.

This also gives Companion / Animal Ally users a pet-independent participation path without fabricating pet ownership or pet state.

## Season authority

Authority version: `expedition-authority-v1`.

Expedition key: `shared-world`.

Expedition version: `v1`.

The first season boundary is an explicit Monday 00:00 UTC to next-Monday 00:00 UTC window, represented as `week:YYYY-MM-DD`.

This is not a daily streak. Missing a day does not erase progress, freeze a streak, create a comeback penalty, or make recovery a failure state.

## Objectives and anti-grind caps

### Sniff & Explore

Categories: `EXPLORE`, `ENRICH`.

Each contributor can add at most two receipts per category and four receipts total per season/scope.

Repeated exploration volume cannot dominate collective progress.

### Recovery Counts

Category: `RECOVER`.

Each contributor can add at most two receipts per season/scope.

Recovery is legitimate participation, not evidence of a broken streak.

### Read the Room

Categories are the four Human Skill rooms.

Each contributor can add at most one receipt per distinct room and four total per season/scope. Retrying a room or improving a practice score adds no further Expedition progress.

## Immutable receipts

`dogos_social.expedition_receipts` is additive operational state backed by canonical evidence.

Each receipt binds:

- expedition key/version;
- explicit season;
- policy version;
- `GLOBAL` or `PACK` scope;
- Pack ID only for Pack scope;
- contributor user ID;
- canonical source type and source ID;
- objective and bounded category/pathway;
- deterministic source fingerprint;
- evidence timestamp;
- authorization timestamp.

The database rejects receipt updates with a trigger. Duplicate evidence identities are prevented with a unique index and materialization uses `ON CONFLICT DO NOTHING`, making retries/concurrent reads idempotent.

Account and Pack deletion cascades may delete associated operational receipts according to existing lifecycle policy. Immutability means application code cannot silently rewrite an issued contribution into a different contribution.

## Calibration is deliberately not fake authority

The new Expedition API returns:

- `target: null`
- `status: CALIBRATING`

for the initial objectives.

We do not yet know the right collective completion targets. Authority v1 should collect bounded participation evidence in a pilot, then calibrate targets against active-user breadth and the desired pace of world reveal.

Arbitrary historical numbers such as 250 raw events are not promoted into new game truth merely because they already exist.

## Legacy compatibility

`GET /api/v1/pack/challenges` is deprecated.

Despite its historical name, that endpoint was a database-wide cooperative aggregate with no Social Pack membership predicate. Authority v1 preserves its broad/global meaning as a compatibility adapter over the new **Global Expedition receipt projection**.

The legacy response keeps its old challenge IDs/targets so existing clients do not break immediately. Those legacy targets are compatibility presentation only and are not the calibrated target authority of the new Expedition API.

There is one cooperative truth engine. The old route does not maintain a second raw-event aggregate.

## Hard exclusions

The following contribute zero Expedition progress:

- `CARE`, clinical state, symptoms, diagnoses, medications or veterinary behavior;
- raw steps, distance, mileage, duration, speed, calories or exercise intensity;
- pet obedience, physical capability, fear threshold or behavioral performance;
- likes, comments, followers, reactions or posting frequency;
- streaks, freezes, missed days or comeback penalties;
- Human Skill score magnitude, correctness magnitude or timing precision;
- free-form Pack locality or device location.

Safe stopping, adapting the setup, decompression, and an ordinary day must never be framed as losing the Expedition.

## Presentation later

Once authority is qualified and pilot evidence supports the mechanic, native Woof can render the projection as a cooperative world rather than a progress spreadsheet:

- regions of a world reveal as diverse categories accumulate;
- each objective can reveal a different ecological or narrative layer;
- Pack worlds can feel intimate without displaying individual raw performance;
- completed worlds can archive into Story as seasonal artifacts;
- League ranking remains visually secondary and separate.

The animation may become delightful. The counting rules stay boring, bounded, inspectable, and server-owned.
