# dogOS event and legacy reward authority

Issue: #68

## Canonical authority

Woof has two maintained incentive systems with separate semantics:

- **Bond XP / Adventure:** issued only from trusted canonical CareEvents into the immutable `reward_ledger`, with policy-owned caps, evidence rules, dedupe, and pathway semantics.
- **Social Adventure Score:** a separate competition score derived from qualified social/adventure evidence. It is not legacy points and does not receive event-attendance or feedback bonuses.

Community event attendance and event feedback are **acknowledgement-only participation signals**. They do not award Bond XP, Social Adventure Score, legacy points, badges, streaks, or another hidden currency.

## Event participation semantics

### Check-in

`EventsService.checkIn` claims the unchecked RSVP with one conditional database transition:

- the update matches `eventId`, `userId`, and `checkedInAt: null`;
- exactly one concurrent request can perform the state transition;
- a retry or concurrent loser reads the canonical RSVP and returns an acknowledged no-op;
- a user with no RSVP still receives a bounded validation error;
- the response contains no reward amount or reward copy.

This is intentionally retry-safe. A repeated check-in must not become a second semantic event merely because the client retried.

### Feedback

`EventsService.submitFeedback` requires an RSVP and uses the existing `(eventId, userId)` identity through one database upsert. Re-submission updates the same feedback record. First submission and later edits have the same reward authority: **none**. Omitting optional tags retains the previous replacement behavior by storing an empty tag list rather than silently preserving older tags.

## Legacy gamification compatibility

`/api/v1/gamification/me/summary` remains an authenticated, deprecated compatibility read. It may expose historical legacy totals, badge rows, and streak rows that predate this authority boundary.

The legacy compatibility service is read-only:

- no `awardPoints`;
- no `awardBadge`;
- no `updateStreak`;
- no point/badge/streak create, update, upsert, or delete authority;
- no orphaned legacy leaderboard or point-transaction service reads;
- stale streak display is normalized in memory rather than written during a read;
- `GamificationModule` does not export the service to other domains;
- mutation DTOs have been removed.

Historical rows are not deleted or migrated by this release. That would be a separate data-retention/product-compatibility migration with its own rollback and evidence requirements.

## Legacy total-points freeze

Bond XP is no longer mirrored into `users.totalPoints`. Canonical reward truth comes from `reward_ledger`; the old aggregate is not an Adventure ledger or an incentive authority.

A fresh user receiving positive Bond XP must therefore retain the default legacy `totalPoints` value. Existing non-zero totals remain historical compatibility data until a later explicit archive/removal migration.

The Web profile labels this field **Legacy points** so the product does not present the frozen counter as an active economy.

## Client authority cleanup

The retirement boundary includes maintained client source, not only the API service. The release removes phantom Web and Mobile gamification clients that referenced routes the maintained server does not expose, including legacy point mutation, leaderboard, stats, badge, and arbitrary gamification profile routes.

The Mobile profile no longer depends on points, levels, ranks, badges, or an imaginary leaderboard. It also no longer navigates to unregistered `Settings`, `EditProfile`, `PetsList`, `PetDetail`, `Activities`, or `Leaderboard` routes. It uses only current account identity, real pet data, the registered `Pets`, `Goals`, `Events`, and `Library` tabs, and logout. Pet-loading failure is isolated so it does not make the account profile itself unavailable.

The deprecated `/gamification/me/summary` read remains available only as historical compatibility for the Web profile. It is not a general client reward API.

## Non-authority invariants

The following must remain true:

- Events never import or inject `GamificationService`.
- Event check-in and feedback responses never contain `pointsAwarded` or “earned points” copy.
- No API runtime calls `awardPoints`, `awardBadge`, or `updateStreak`.
- No runtime creates `pointTransaction` or `badgeAward` rows through the legacy service.
- No runtime mutates `weeklyStreak` through the legacy service.
- CareEvents do not increment `users.totalPoints`.
- Social Adventure remains independent of legacy points.
- Community participation is not silently converted into Bond XP.
- Historical compatibility reads must not mutate their backing rows.
- Web and Mobile source must not reintroduce `/gamification/points`, `/gamification/leaderboard`, `/gamification/stats`, `/gamification/badges`, or `/gamification/profile/*` clients.
- Mobile Profile navigation must remain within routes actually owned by the maintained navigator.

## Deliberate non-goals

This release does not:

- delete historical point transactions, badge awards, streaks, or user totals;
- reinterpret old point values as Bond XP;
- migrate legacy totals into Social Adventure Score;
- add event attendance or feedback to the Bond XP reward policy;
- change Adventure reward caps, evidence eligibility, or ledger semantics;
- change Social Adventure competition policy;
- add a new Mobile Adventure or Social Adventure client surface;
- claim that historical legacy data has been purged from a deployed database.

## Qualification boundary

Repository qualification proves source ownership, retry-safe event behavior, read-only legacy compatibility, Bond XP ledger independence, Web/Mobile client truth, and API/Web/Mobile type-safety ownership.

It does **not** prove production database contents, deployed UI revision, physical-device Mobile behavior, or live traffic behavior. Those require deployment/device authority and black-box evidence.
