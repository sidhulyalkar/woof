# Native Expedition World v1

## Purpose

The native Expedition surface gives Woof's cooperative weekly system a game-shaped place to live without creating a second reward authority on the phone.

The visual north star is a shared landscape that feels inhabited by useful human choices. The human gets a world to explore; the dog does not become the game piece.

## Authority boundary

The native client reads only the server projections:

- `GET /api/v1/expeditions/global`
- `GET /api/v1/expeditions/packs/:packId`

There is no native Expedition contribution mutation. Canonical Adventure and Human Skill evidence is reconciled by the server into immutable, bounded receipts. The phone never submits Expedition points, receipt counts, completion state, target values, or ranking state.

The first native world recognizes the three v1 server objective keys:

- `SNIFF_EXPLORE` → **Wandering Grove**
- `RECOVERY_COUNTS` → **Resting Hollow**
- `READ_THE_ROOM` → **Signal Observatory**

Those place names, icons, hills, and other scenery are presentation. They do not change what the server counts.

## Calibration means no meter

Expedition Authority v1 deliberately returns `target: null` and `status: CALIBRATING` for the canonical projection.

Native therefore does **not** invent a fallback target, percentage, completion bar, level threshold, unlock threshold, or aggregate "three of three" meter. Every landmark exists from the beginning. The surface shows the server's bounded contribution total, contributor count, and the viewer's contribution as descriptive evidence only.

A viewer with `myContribution > 0` may see a small **Your mark** treatment. This is binary presentation that a server-issued contribution exists. It is not a proficiency claim, completion claim, reward multiplier, or hidden score.

The scene geometry is intentionally static with respect to totals. Contribution counts do not enlarge landmarks, brighten the sky, unlock terrain, increase opacity, or otherwise encode an unofficial target.

## Global and Pack scope

Everyone can read the Global Expedition.

Pack Expedition requests are stricter. The native screen first reads the server Pack catalog and exposes Pack selectors only where the server returned `joined: true`. A Pack projection is fetched only after that server-confirmed membership check. The response must identify the same Pack before it is displayed.

If Pack membership cannot refresh, Woof keeps any previously confirmed projection clearly stale rather than inferring membership from location or local state. If a refreshed catalog says the user is no longer joined, the Pack projection is hidden and the view returns to Global.

No coordinate, route trace, nearby inference, or Social Adventure leaderboard result can grant Pack Expedition access.

## What can count

The server remains the sole source of truth. In v1 it can issue bounded Expedition receipts from:

- suitable `EXPLORE` and `ENRICH` Adventure evidence;
- `RECOVER` Adventure evidence;
- completion breadth across distinct Human Skill rooms.

The native client does not reproduce these eligibility rules to award progress.

CARE, medical state, health signals, symptoms, medication, distance, duration, intensity, missed days, streaks, likes, reactions, follower counts, leaderboard rank, and repeated grinding do not become Expedition progress. Human Skill practice-score magnitude does not become Expedition progress either.

Recovery remains visible because choosing decompression or stopping can be the useful decision. The cooperative game must never imply that doing more is always better.

## Failure behavior

Global projection, Pack membership, and Pack projection reads fail independently.

When authority is unavailable, native shows the last server-confirmed state where appropriate and explicit unavailable copy. It does not reconstruct missing Expedition state from Adventure Trail XP, Social Adventure score, Story, raw activity history, cached route metrics, or client arithmetic.

## Relationship to Social Adventure

Social Adventure and Expedition intentionally coexist:

- **Social Adventure** is an optional, bounded human-side league.
- **Expedition** is cooperative world-building without a podium.
- **Adventure Trail** is private relationship presentation.

Their numbers are not interchangeable.

Community links to Expedition as a separate destination rather than embedding cooperative totals into the league card.

## Companion participation

Companion-mode users can open the same Expedition surface. They can contribute through eligible Human Skill breadth without Woof pretending they completed dog-specific Adventure evidence. This keeps the social learning loop useful before pet ownership and leaves room for future foster, shelter, and responsible adoption pathways.

## Deliberately deferred

This first native tranche does not persist completed Expedition seasons into Story, award cosmetics, mint badges, add loot, create streaks, or set community targets.

A later server-authorized archive can turn past seasons into Story artifacts once calibration establishes useful targets and retention evidence shows the mechanic improves learning and connection rather than volume-chasing.

## Qualification contract

`Native Expedition World CI` must prove that:

1. the mobile Expedition API is GET-only;
2. Global and Pack projections remain server-authored;
3. Pack requests require a server-confirmed joined Pack and response-Pack identity check;
4. `CALIBRATING` renders as **no target yet** rather than a client meter;
5. server totals are displayed but not transformed into progress geometry or completion arithmetic;
6. Community links to Expedition;
7. both guardian and Companion navigation graphs expose the same Expedition screen;
8. full native type-check and lint stay green;
9. the server Expedition policy contract is rerun alongside the native qualification.
