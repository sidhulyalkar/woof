# dogOS Adventure Learning v2

Adventure learning v2 separates three authorities that must not collapse into one another:

1. **Reward authority** decides whether a trusted action earns Bond XP and which reward pathway receives it.
2. **Learning authority** derives bounded recommendation-fit signals from canonical Adventure outcomes.
3. **Temporary context** adapts the next few recommendations to handler load and welfare-respecting stops without rewriting durable dog preference.

The purpose of this release is not to make personalization more aggressive. It is to make the evidence semantics correct enough that stronger ranking methods can be introduced later without learning the wrong target.

## Canonical evidence boundary

Adventure learning reads from `care_events`, but through a dedicated projection rather than the general Care Summary.

`CareEventsService.getAdventureLearningEvents()` is:

- authorized for the exact `userId + petId` handling pair;
- restricted to `source = QUEST_ENGINE`;
- restricted to `QUEST_*` and `SAFE_OPT_OUT` outcomes;
- limited to the last 28 days and at most 24 events;
- independent of `reward_ledger`;
- intentionally free of Bond XP or reward values.

Frequent activity logs, Daily Signals, health events, or unrelated CareEvents therefore cannot crowd the relevant Adventure outcomes out of the learning window.

Interaction-derived recommendation fit remains handler-pair scoped. Shared durable facts about the dog belong in Adaptive Profile rather than being inferred by pooling every household member's handling outcomes.

## Original pathway versus reward pathway

A welfare-respecting safe opt-out or an explicit mismatch can intentionally earn a BOND reward even when the attempted Adventure was LEARN or CONNECT.

That reward pathway is not the learning target.

Completion persists both concepts separately:

- `originalPathway`: what Woof actually recommended and the pair attempted;
- `rewardPathway`: where the reward policy placed the resulting XP.

`originalPathway` is the only pathway provenance used for recommendation-fit learning when it is available. Ambiguous legacy BOND mismatch events without original-pathway provenance are ignored rather than guessed.

## Durable dog fit

Durable fit is based only on explicit dog-experience outcomes:

- `loved_it`: positive evidence;
- `comfortable`: smaller positive evidence;
- `not_their_thing`: negative evidence when it was not a safe opt-out.

Signals decay with age across a 28-day window and remain bounded to a small `[-0.08, 0.08]` pathway modifier. A single mismatch cannot erase several recent positive outcomes.

Owner effort does not suppress dog evidence. If the dog loved an activity while the handler reports that it was a lot today, Woof can learn both truths simultaneously: durable positive dog fit and temporary lower-effort context.

## Temporary handling context

Owner load and safe opt-outs are deliberately not durable dog preference labels.

`ownerExperience = a_lot_today` temporarily:

- reduces the relative weight of higher-effort MOVE, EXPLORE, ENRICH, LEARN, and CONNECT options;
- increases RECOVER and BOND relevance;
- marks the immediate pace as `easy` for roughly the next day.

A safe opt-out temporarily lowers immediate repetition of the original attempted pathway and raises RECOVER/BOND options. It never creates durable dislike evidence.

Temporary modifiers decay across three days and disappear completely after the window.

## Ranking influence

Adventure keeps the existing conservative relevance envelope. Durable and temporary modifiers are combined and clamped so `personalRelevance` stays between `0.90` and `1.08`.

When temporary pace is `easy`, the explicit Recovery template receives only a small additional score nudge. The system becomes gentler for the moment rather than rewriting the entire recommendation order.

This release intentionally does not introduce a learned bandit, neural ranker, or unconstrained optimization loop. It establishes the evidence contract those systems would need to respect.

## Policy provenance

The current policy identifier is:

`adventure-learning-v2`

Selection and completion observability record the policy version alongside original/reward pathway provenance so later audits can reconstruct which learning policy produced and interpreted a recommendation.

## Hard prohibitions

Adventure learning must never use the following as recommendation targets or features:

- Bond XP;
- reward-ledger values;
- total points;
- streak pressure;
- repetition volume by itself;
- safe stopping as negative preference evidence;
- temporary owner load as permanent dog preference;
- unrelated CareEvent JSON fields that merely resemble Adventure outcomes.

The dedicated `dogOS Adventure Learning Authority CI` lane enforces these architecture boundaries in addition to running deterministic policy contracts, reward-regression tests, the Postgres evidence-projection integration test, TypeScript, formatting, and API build.

## Product interpretation

The intended loop is:

**Recommend → practice/play → observe the dog → report the pair's experience → preserve welfare-respecting choices → adapt tomorrow without overreacting.**

The learning system should become more useful because it understands the difference between *the dog did not enjoy this*, *this was too much for me today*, and *we correctly chose to stop*. Those are different facts, and dogOS now stores and learns from them as different facts.
