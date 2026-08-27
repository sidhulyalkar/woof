# dogOS Social Adventure v1

Social Adventure turns Woof's relationship-first daily loop into an opt-in social game without making the animal's obedience, health, exercise volume, fear threshold, or symptom burden the competition.

## Product rule

**The human can compete. The pet does not have to.**

The social loop is:

`useful moment together -> private canonical outcome -> optional share -> community reaction / Pack participation -> human skill + bounded variety progression`

The scientific loop remains separate:

`canonical outcome -> pair learning -> safer/better next recommendation`

Social score is never recommendation truth.

## Three separate economies

### Canonical evidence

Adventure, Coach and CareEvent outcomes remain the authority for what happened. Dog experience, owner burden, safe opt-outs and corrections stay distinct and private unless an owner explicitly publishes a bounded social card.

### Bond / Story progression

Bond XP, chapters, discoveries and memories describe the dog-human relationship. Social Adventure does not rewrite those ledgers.

### Social Adventure score

`social-adventure-score-v1` is a weekly presentation/competition economy. It is computed from trusted receipts and may be discarded/recomputed without changing canonical dog state.

The v1 score is:

- best weekly score for each of four Human Skill Arcade games, max 400;
- 25 points per distinct eligible Adventure pathway, max 175.

Maximum: 575.

Eligible Adventure variety pathways are `MOVE`, `EXPLORE`, `ENRICH`, `LEARN`, `CONNECT`, `RECOVER`, and `BOND`.

The following produce **zero Social Adventure points**:

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

This keeps competitive pressure from rewarding overexercise, deliberate overexposure, popularity or data entry.

## Human Skill Arcade

v1 contains four server-scored games:

### Make It Easier

Choose a smaller, more learnable next step when a setup is too difficult. The first scenario teaches lowering environmental distraction rather than repeating a cue louder or escalating difficulty.

### Catch the Good

Identify useful voluntary behavior worth reinforcing instead of waiting for an error.

### Pairing Lab

Practice basic positive-association timing. A mild, tolerable stimulus predicts the positive event. The exercise explicitly does not authorize DIY escalation for serious fear/aggression cases.

### Marker Timing

Practice marking the target behavior with temporal precision. The game score is a teaching aid, not a claim that millisecond precision is itself the training goal.

Attempts are server-issued, expire, are scored server-side, and become immutable after completion. Only the best weekly score per game contributes to competition, so replay can serve mastery without repetition farming.

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

Completed Human Skill attempts are immutable at PostgreSQL. Competition receipts are append-only/immutable at PostgreSQL. Competition receipt identity includes user, weekly season, policy version and a SHA-256 hash of the exact bounded source evidence used for that snapshot.

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
- Arcade score;
- correct answer;
- pet outcome;
- social share kind;
- health/safety status;
- source propensity/model feature;
- exact locality.

## UI surfaces

- `/community`: score summary, explicit global opt-in, global league, recent private share candidates, social feed and welfare-positive reactions.
- `/arcade`: four Human Skill games, best-score semantics and optional result sharing.
- `/community/packs`: coarse local Pack creation/joining and privacy-thresholded local standings.
- `/pack`: cooperative non-ranking Pack challenges remain available.
- `/leaderboard`: redirects to `/community`; the old mock distance/walk/friend-count leaderboard is retired.
- bottom navigation now uses **Community** as the social destination.

## Release invariants

Social Adventure v1 must fail qualification if:

- legacy post creation awards points;
- score policy starts counting CARE, posting/popularity, or repeated pathway volume;
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
5. friend-authority semantics for `FRIENDS_ONLY`;
6. `ANIMAL_ALLY`, foster, volunteer and authorized-caregiver modes from #64;
7. partner-authorized shelter/foster opportunities and responsible adoption-readiness flows.

The release succeeds if social energy makes useful dog-human interaction feel more playful while the data and incentive architecture makes unsafe volume/popularity optimization harder, not easier.
