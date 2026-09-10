# Native Expedition Field Journal v1

## Purpose

The Expedition Field Journal makes Woof more collectible without creating another reward economy.

A field note says **this happened**. It does not say **do more**, **finish the set**, or **beat somebody else**.

The journal is a personal memory projection over the same immutable Expedition receipts that already power the cooperative world. It does not mint badges, award points, create streaks, or write a second archive.

## Canonical authority

The journal reads only:

- the signed-in user's own receipts;
- `GLOBAL` Expedition scope;
- the current `EXPEDITION_KEY`, `EXPEDITION_VERSION`, and `EXPEDITION_POLICY_VERSION`;
- canonical Expedition objective keys already accepted by the server.

`GET /api/v1/expeditions/journal` is read-only.

Before reading journal history, the journal service asks the canonical `ExpeditionsService` to materialize the current Global season. That existing materialization is idempotent and remains the only writer. The journal never inserts, updates, claims, completes, or awards anything.

No new database table or migration is required. The existing immutable `dogos_social.expedition_receipts` table is the evidence source.

## Presence, never volume

For a given season, repeated authorized receipts for one objective collapse to one journal landmark.

Two bounded `EXPLORE` receipts therefore create the same **Wandering Grove** stamp as one receipt. The journal response intentionally omits receipt count, contribution count, score, rank, target, rarity, and completion fields.

The three possible journal landmarks remain:

- `SNIFF_EXPLORE` → **Wandering Grove**
- `RECOVERY_COUNTS` → **Resting Hollow**
- `READ_THE_ROOM` → **Signal Observatory**

The UI renders only landmarks that actually occurred. It does not render empty slots for the other landmarks. Empty slots would turn a memory page into a checklist.

A page with only **Resting Hollow** is not less complete than a page with several kinds of moments. It simply records a different week.

## Active and past pages

A journal entry can be:

- `ACTIVE` for the current Monday-UTC Expedition season;
- `PAST` for an earlier valid Monday-UTC season.

`ACTIVE` does not mean incomplete. The native copy explicitly says the page reflects server-issued receipts so far and that there is nothing the user needs to fill.

`PAST` does not mean completed. The page simply records the landmark kinds that had authorized evidence in that season.

No state called `COMPLETE`, `FAILED`, `MISSED`, or equivalent belongs in the journal.

## Historical validation

Storage is not treated as presentation authority by itself.

Journal projection accepts only normalized `week:YYYY-MM-DD` keys that parse to real Monday UTC dates and are not after the current Expedition season. A malformed, non-Monday, or future receipt must not surface as a journal page.

This validation is useful even though canonical Expedition writers already create valid season keys. Historical projections should fail closed if storage is ever repaired, imported, or manually altered.

## Privacy and scope

Field Journal v1 is personal and Global-only.

It does not expose another user's receipts, community totals, contributor counts, Global rank, Pack rank, or Pack history. Pack receipts stay out of the personal Global journal even when the viewer currently belongs to that Pack.

Pack history is deliberately deferred. Historical Pack membership changes, ownership transfer, and local-community privacy deserve their own authority contract rather than being inferred from current membership.

## Bounded coverage

The first journal returns up to the **26 most recent participated seasons** under the current Expedition authority version.

This is intentionally described as recent coverage, not complete lifetime history. A user with more than 26 participated seasons may have older valid receipts that are not returned.

Native copy must say that missing older pages are not presented as non-participation.

The API returns the coverage contract explicitly:

- `kind: RECENT_PARTICIPATED_SEASONS`
- `maxSeasons: 26`

The client does not infer a lifetime participation count from the number of returned pages.

## Native presentation

The journal lives inside the Expedition surface so both Guardian and Companion modes receive the same human-side history.

Each field note is postcard-like rather than achievement-like:

- week label;
- one stamp per landmark that actually occurred;
- no empty stamp placeholders;
- no page score;
- no progress meter;
- no rarity color;
- no completion burst;
- no reward claim button.

Season dates are formatted in UTC so the Monday-UTC authority does not appear as Sunday on clients west of UTC.

The current live Expedition world may show server-authored communal totals. Historical journal pages do not. The present can say **we are here together**; the past only says **you were here**.

## Failure behavior

Current Global Expedition, Pack catalog, Pack projection, and Journal are independent read slices.

A Journal failure does not erase a healthy current-world projection. A current-world failure does not authorize the client to rebuild journal history from Adventure Trail XP, Social Adventure score, Story, route data, local caches, or raw activity records.

No local fallback journal is permitted.

## Companion mode

Companion users can accumulate legitimate journal history through human-side Expedition evidence such as eligible Human Skill breadth. Woof does not synthesize dog-specific Adventure stamps for them.

This keeps the journal useful before pet ownership without making acquiring a pet an unlock condition or reward.

## Relationship to Story

Field Journal v1 is **not** Story persistence and does not create Story moments or milestones.

That separation is useful. Expedition history is human-side cooperative evidence, while Story remains relationship memory tied to authorized pet context. A later product tranche may create an explicit guardian-side Story link or server-authored seasonal artifact, but it must not silently reinterpret Journal participation as a pet achievement.

## Qualification contract

`Expedition Field Journal CI` must prove that:

1. Journal is a GET-only authenticated projection;
2. current-season receipts are materialized only through canonical `ExpeditionsService` authority;
3. only the viewer's current-policy Global receipts enter Journal history;
4. Pack receipts and other users' receipts do not enter the journal;
5. future, malformed, and non-Monday season keys do not surface;
6. repeated receipts collapse to one landmark presence per objective per season;
7. the response contains no contribution counts, scores, ranks, targets, rarity, or completion state;
8. native pages render only returned landmarks and no empty completion slots;
9. native copy explicitly rejects catch-up and fill-the-page pressure;
10. coverage is clearly bounded to recent participated seasons;
11. UTC season formatting is preserved;
12. API and mobile type-check/lint remain green alongside the existing Expedition integration contracts.

## Future visual evolution

The journal can become more delightful without becoming more coercive.

Safe future directions include server-authored seasonal paper textures, weather motifs, non-scarce illustrated postcards, or short narrative captions derived from the objective mix. Those cosmetics should not encode rarity, comparative status, health, pet performance, or hidden completion thresholds.

The useful design question is not **how do we make people fill every page?** It is **how do we make real weeks worth remembering?**
