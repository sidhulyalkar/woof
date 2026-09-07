# Native Skillcraft v1

## Purpose

Skillcraft brings the existing server-authoritative Human Skill Arcade into native Woof.

The game is intentionally aimed at the human side of the relationship:

```text
notice -> choose a setup -> time the useful moment -> learn from feedback -> try again
```

A dog does not need to perform for the human to practice. Petless Companion users and Pet Guardian users receive the same Skillcraft surface and the same server-authored game rules.

## Product contract

Native Skillcraft is a client for the existing `social-adventure` authority. It does not invent questions, correct answers, score formulas, weekly seasons, league points, or proficiency claims.

The client uses:

- `GET /social-adventure/arcade` for the public challenge catalog and weekly practice state;
- `POST /social-adventure/arcade/:challengeKey/attempts` for a server-issued attempt;
- `POST /social-adventure/arcade/attempts/:attemptId/complete` for server scoring;
- `POST /social-adventure/shares` only after an explicit user action to share a completed Human Skill attempt.

The server remains authoritative for challenge version, scenario identity, attempt expiry, scoring, correctness, timing error, explanations, and the Social Adventure breadth model.

## Four rooms

v1 exposes the four server-owned Human Skill challenges already qualified by dogOS Social Adventure:

1. **Make It Easier** — difficulty and setup selection;
2. **Catch the Good** — noticing useful behavior worth reinforcing;
3. **Pairing Lab** — positive association ordering and timing;
4. **Marker Timing** — temporal precision practice.

Native code does not contain correct option IDs or a duplicate score formula.

## Weekly breadth, not grind

The catalog's `bestScore` is scoped by the server to the current UTC weekly season.

Native Skillcraft turns that into a simple room counter:

```text
rooms explored this week / 4
```

A room is explored when the server reports a current-season practice result for that challenge.

The important asymmetry is deliberate:

- completing a distinct room can contribute one fixed Human Skill breadth unit to Social Adventure;
- replaying the same room does not add another breadth unit;
- raising a personal score does not increase public rank;
- millisecond timing magnitude does not increase public rank;
- posting, reactions, comments, or popularity do not increase public rank.

The 0–100 result remains useful personal feedback. It is not evidence of professional training proficiency.

## Timing game

Marker Timing uses the server-issued public timing scenario.

The native client measures the user's tap relative to the local round start and sends only `tapMs` as the challenge response. The server decides the practice score and timing error.

The native client may animate the public behavior track and describe the current cue window, but it must not reproduce the server score formula or turn timing precision into competitive authority.

If the local display reaches the end of the playable window before a tap, the client stops the round and asks for a fresh attempt. It does not manufacture an outcome.

## Sharing

Sharing is always optional and post-completion.

Native Skillcraft can publish a completed Human Skill attempt only when the user presses **Share skill moment**. The request uses:

- `sourceType = HUMAN_SKILL_ATTEMPT`;
- the server-owned attempt ID;
- explicit `PUBLIC` visibility.

The server resolves what is safe to publish from the authorized attempt. The client does not copy private dog history into the share payload.

A shared practice result is social expression, not recommendation evidence. Likes, comments, reactions, follower count, and posting frequency do not add Skillcraft or Social Adventure rank.

## Companion boundary

Skillcraft is pet-independent by design.

`COMPANION_TODAY` users can open the same native Skillcraft route as `PET_TODAY` users without receiving pet-only Today, Compass, Story, CareEvent, Health, or household authority.

The route therefore belongs in both authenticated native navigators, but Skillcraft itself does not import pet, Adventure, CareEvent, Health, or household mutation APIs.

Account mode still controls presentation. Pet relationships still control pet authority.

## Safety boundary

Skillcraft teaches general reward-based mechanics. It must never imply that an Arcade score authorizes a user to handle significant fear, aggression, pain, sudden behavior change, or other cases that may require qualified trainer, behavior, or veterinary support.

The game must not:

- rank a dog;
- award Bond XP;
- create or modify pet state;
- create CareEvents or recommendation evidence;
- unlock care or health access;
- claim professional proficiency;
- reward repeated attempts with more public points;
- reward higher practice scores with more public points;
- automatically share a result;
- make social reactions part of the score.

## Native surfaces

### Skillcraft

The native screen provides:

- a four-room weekly breadth panel;
- server-authored challenge cards;
- server-issued rounds;
- multiple-choice interaction for three rooms;
- a local timing track for Marker Timing;
- server-scored receipts and explanations;
- personal practice-best context;
- an explicit optional share action;
- an explicit professional-support boundary.

### Companion Home

Skillcraft is the first useful pet-independent action shown in Companion mode.

### Community

Community links to Skillcraft so guardians can move from social browsing into human-skill practice rather than making scrolling the engagement destination.

## Qualification

`Native Skillcraft CI` cross-checks native and server source contracts. It must fail if:

- the four Human Skill challenge keys drift unexpectedly;
- the native API stops using the existing server-authoritative Arcade endpoints;
- correct answer IDs or the server score formula are copied into native code;
- Skillcraft gains pet/Adventure/CareEvent/Health mutation authority;
- weekly breadth, no-grind, optional-sharing, or safety-boundary copy disappears;
- the route is removed from either Pet Guardian or Companion native navigation;
- Companion Home or Community loses its Skillcraft entry point;
- touched source or documentation is not formatted;
- the full native client fails type-check or lint.

Existing Social Adventure, Companion, mobile convergence, security, Xcode, root CI, and CodeQL lanes remain broader authorities.

## What comes next

Skillcraft makes Woof more game-like without needing another currency. The next game releases should build on that restraint:

- cooperative **Expeditions** with varied eligible contributions instead of volume races;
- Story-derived collectible art that never pressures uploads;
- friend challenges that are re-resolved against the recipient's eligibility rather than copying one dog's task;
- cosmetic Trail/camp evolution from already-earned discovery, never care compliance;
- richer trainer- and shelter-reviewed Human Skill scenarios;
- Companion learning paths that make fostering, volunteering, and responsible guardianship easier to understand without pretending Woof grants placement eligibility.

The design target remains: **the human gets the game; the dog keeps the right to have an ordinary day.**
