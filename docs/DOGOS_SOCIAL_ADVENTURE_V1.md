# dogOS Social Adventure v1

Social Adventure turns Woof's relationship-first daily loop into an opt-in social game without making the animal's obedience, health, exercise volume, fear threshold, or symptom burden the competition.

## Product rule

**The human can compete. The pet does not have to.**

The social loop is:

`useful moment together -> private canonical outcome -> optional share -> community reaction / Pack participation -> human learning breadth + bounded Adventure variety`

The scientific loop remains separate:

`canonical outcome -> pair learning -> safer/better next recommendation`

Social score is never recommendation truth.

## Three separate economies

### Canonical evidence

Adventure, Coach and CareEvent outcomes remain the authority for what happened. Dog experience, owner burden, safe opt-outs and corrections stay distinct and private unless an owner explicitly publishes a bounded social card.

### Bond / Story progression

Bond XP, chapters, discoveries and memories describe the dog-human relationship. Social Adventure does not rewrite those ledgers.

### Social Adventure score

`social-adventure-score-v1` is a weekly presentation/competition economy. It is computed by the server and may be discarded/recomputed without changing canonical dog state.

The v1 score deliberately measures **breadth, not claimed proficiency**:

- 50 points for each distinct Human Skill Arcade game completed during the week, max 200;
- 25 points per distinct eligible Adventure pathway, max 175.

Maximum: 375.

The numeric 0-100 Arcade result is personal practice feedback. A lower score, a perfect score and later retries produce the same league value once that game has been completed. This is intentional: the scenario bank is open source and Marker Timing uses browser-reported timing, so neither should be treated as tamper-resistant evidence of real-world handling proficiency.

Eligible Adventure variety pathways are `MOVE`, `EXPLORE`, `ENRICH`, `LEARN`, `CONNECT`, `RECOVER`, and `BOND`.

The following produce **zero additional Social Adventure points**:

- higher Arcade practice scores;
- replaying an Arcade game already completed that week;
- distance;
- steps or exercise intensity;
- repetition/session volume beyond distinct variety;
- post count;
- likes/reactions/comments/followers;
- legacy total points;
- Bond XP;
- CARE / symptom / Health Lens / Daily Signals state;
- safe-stop count;
- missed days or streak length.

This keeps competitive pressure from rewarding overexercise, deliberate overexposure, popularity, repetition farming, or spoofed practice telemetry.

## Human Skill Arcade

v1 contains four server-scored practice games:

### Make It Easier

Choose a smaller, more learnable next step when a setup is too difficult. The first scenario teaches lowering environmental distraction rather than repeating a cue louder or escalating difficulty.

### Catch the Good

Identify useful voluntary behavior worth reinforcing instead of waiting for an error.

### Pairing Lab

Practice basic positive-association timing. A mild, tolerable stimulus predicts the positive event. The exercise explicitly does not authorize DIY escalation for serious fear/aggression cases.

### Marker Timing

Practice marking the target behavior with temporal precision. The browser reports its tap timing so the learner can receive immediate feedback. That millisecond result is **practice telemetry only** and never determines public rank.

Attempts are server-issued, expire, are scored against the canonical scenario, and become immutable after completion. Choice answers receive 100 for the encoded preferred answer and 0 otherwise. Explanations teach the principle after the round. A completed distinct game contributes one fixed weekly breadth unit regardless of its practice result; retries add no league value.

This separation is important. A short synthetic game can teach a useful concept, but it is not a credential for real-world animal handling. Future proficiency competition requires a stronger evidence channel and should be promoted only after its measurement validity is demonstrated.

## Sharing contract

A canonical Adventure outcome remains private. `GET /social-adventure/share-candidates` returns a bounded preview of recent eligible outcomes that have not already been shared.

The preview may include:

- quest title;
- pet name;
- broad pathway;
- a bounded derived summary;
- a social kind such as `ADVENTURE_MEMORY`, `DISCOVERY`, or `GOOD_READ`.

It does not copy free-form notes, route traces, Daily Signals, Health Lens details, model confidence or arbitrary private context into the social layer.

A second explicit write creates a standard Post plus a `dogos_social.shares` source receipt. `CARE_EVENT` and `HUMAN_SKILL_ATTEMPT` are the only source types in v1. Client-supplied social kinds, pet outcomes, scores and ranking values are not accepted.

Adventure safe opt-outs are represented as `GOOD_READ`. A completed quest that produced `not_their_thing` evidence can become a `DISCOVERY`. This is intentional culture design: noticing, adapting and learning are socially legible wins.

A Human Skill share may include the user's practice result, but that result remains presentation-only and does not increase league score.

## Reactions

Social Adventure reactions are bounded to:

- `NICE_READ`;
- `GOOD_CALL`;
- `TRYING_THIS`;
- `ADVENTURE_INSPIRATION`;
- `CHEER`.

Reactions are social presentation only. They do not become pet/model labels and do not increase league score.

## Feed privacy

The new feed shows:

- PUBLIC Social Adventure shares;
- the viewer's own PRIVATE shares.

Blocking is enforced in both directions.

Legacy `FRIENDS_ONLY` posts intentionally remain hidden from non-owners until Woof has a modern, explicit friend-authority contract. Guessing that an old social edge means permission would fail open.

The legacy Social service is hardened by the same release:

- no `+2 points` reward for posting;
- household-authorized pets/activities instead of owner-only pet checks;
- PUBLIC-or-owner visibility on feed and post reads;
- bilateral block filtering;
- visibility checks before like/comment reads or writes.

## Global league privacy

Global league participation defaults to **off**.

An opted-in user appears only while their profile is PUBLIC. A viewer never sees users blocked in either direction. Turning league visibility off does not delete canonical history or earned private game progress.

## Local Packs

A local Pack stores only a user-chosen coarse `region_key`, such as `south-bay-ca`.

Woof does not derive local Pack membership/ranking from:

- home coordinates;
- route endpoints;
- live GPS;
- device pings;
- inferred neighborhood.

Joining a Pack is explicit. Local standings are withheld until at least five active members exist. This is a first privacy floor, not a claim that five-person aggregation makes all locality risk disappear.

Existing cooperative `/pack/challenges` remain non-ranking and now count only canonical `QUEST_ENGINE` Adventure events, preventing unrelated CareEvents from becoming community progress.

## Database authority

Social Adventure data lives in `dogos_social`, separate from canonical Prisma models.

Tables:

- `preferences`;
- `shares`;
- `reactions`;
- `human_skill_attempts`;
- `packs`;
- `pack_memberships`;
- `competition_receipts`.

Completed Human Skill attempts are immutable at PostgreSQL. Competition receipts are append-only/immutable at PostgreSQL. Competition receipt identity includes user, weekly season, policy version and a SHA-256 hash of the bounded source evidence used for that snapshot.

The v1 score function deduplicates Human Skill games and Adventure pathways before arithmetic. A future ledger hardening slice should also canonicalize receipt source identity to one deterministic first qualifying row per breadth category so irrelevant replay volume cannot create redundant zero-delta snapshots.

## Server authority

Clients may submit:

- a share source ID they own;
- an optional caption;
- PUBLIC/PRIVATE share intent;
- one Arcade answer/tap;
- a bounded reaction;
- an explicit global-league preference;
- a Pack name/coarse region or join intent.

Clients do **not** submit authoritative:

- rank;
- Social Adventure score;
- league proficiency;
- correct-answer truth;
- pet outcome;
- social share kind;
- health/safety status;
- source propensity/model feature;
- exact locality.

Arcade practice scores are calculated by the server from the canonical scenario, but v1 intentionally does not elevate those values into competitive evidence. In particular, browser timing is untrusted practice telemetry.

## UI surfaces

- `/community`: score summary, explicit global opt-in, global league, recent private share candidates, social feed and welfare-positive reactions.
- `/arcade`: four Human Skill practice games, personal best-score feedback, fixed breadth contribution and optional result sharing.
- `/community/packs`: coarse local Pack creation/joining and privacy-thresholded local standings.
- `/pack`: cooperative non-ranking Pack challenges remain available.
- `/leaderboard`: redirects to `/community`; the old mock distance/walk/friend-count leaderboard is retired.
- bottom navigation now uses **Community** as the social destination.

## Release invariants

Social Adventure v1 must fail qualification if:

- legacy post creation awards points;
- score policy starts counting CARE, posting/popularity, raw Arcade score magnitude, or repeated pathway volume;
- global leaderboard default becomes opt-out instead of opt-in;
- local cohort threshold disappears;
- completed Arcade attempts become mutable;
- competition receipts become mutable;
- Social feed loses bilateral block or visibility authority;
- Pack challenges count arbitrary CareEvents;
- migration/schema drift leaks the operational schema into canonical Prisma state;
- API/web type checks, tests or production builds fail.

## Next releases

This release intentionally does not finish all of #63 or #64.

Next Social Adventure slices should add:

1. friend-to-friend challenge templates that are re-resolved through recipient eligibility;
2. curated seasonal/cooperative expeditions;
3. stronger moderation/reporting around shared content;
4. richer Human Skill scenario banks and trainer-reviewed calibration;
5. a validated evidence channel before any true proficiency-based public competition is promoted;
6. canonical breadth-only competition receipt source identity so zero-value repetition cannot churn snapshots;
7. friend-authority semantics for `FRIENDS_ONLY`;
8. `ANIMAL_ALLY`, foster, volunteer and authorized-caregiver capabilities from #64;
9. partner-authorized shelter/foster opportunities and responsible adoption-readiness flows.

The release succeeds if social energy makes useful dog-human interaction feel more playful while the data and incentive architecture makes unsafe volume, popularity optimization and gameable pseudo-skill ranking harder, not easier.
