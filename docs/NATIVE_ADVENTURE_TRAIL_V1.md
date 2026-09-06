# Native Adventure Trail v1

## Purpose

Adventure Trail makes Woof feel more playful without turning the dog into a scorecard.

The real-world loop remains authoritative:

```text
notice -> choose -> do together -> read the response -> adapt -> remember
```

Adventure Trail is a presentation layer over that loop. It gives the human a longer arc to uncover while preserving the existing rule that recommendation truth, reward truth, health truth, and social competition are separate authorities.

## Product contract

Adventure Trail reads only fields already returned by the canonical Adventure dashboard:

- server-earned Bond XP;
- rolling Rhythm;
- canonical pathway history exposed through the Pawprint Compass.

It does not write CareEvents, RewardLedger rows, recommendation evidence, pet state, health state, or social score. The client cannot submit XP or award itself a chapter.

Policy marker: `adventure-trail-presentation-v1`.

## Chapters

v1 contains six presentation-only relationship chapters:

1. **First Pawprints** at 0 Bond XP;
2. **Finding Rhythm** at 100 Bond XP;
3. **Reading Each Other** at 250 Bond XP;
4. **A Wider World** at 500 Bond XP;
5. **The Familiar Trail** at 900 Bond XP;
6. **Shared Language** at 1500 Bond XP.

These thresholds are product pacing constants, not claims about dog quality, owner quality, training mastery, attachment, welfare, or clinical status. Reaching a chapter does not unlock care, increase recommendation authority, or make an action safer.

The final chapter is intentionally open-ended. Woof should not create an infinite treadmill of escalating requirements simply to preserve engagement.

## Discovery stamps

Adventure Trail treats breadth as discovery rather than perfection.

The collectible presentation pathways are:

- `MOVE`;
- `EXPLORE`;
- `ENRICH`;
- `LEARN`;
- `CONNECT`;
- `RECOVER`;
- `BOND`.

A stamp appears after the canonical Compass reports positive earned XP for that pathway. Repetition does not create additional stamps.

`CARE` is intentionally excluded from the collection layer. Preventive or clinical behavior may remain visible in the normal Compass where appropriate, but health-related actions must not become a collectible obligation.

Recovery and safe listening remain legitimate progress. A game that only lights up for exercise, exposure, or obedience would push the product toward exactly the wrong incentives.

## Rhythm, not streaks

The native Trail visualizes the existing rolling multi-week Rhythm count with paw markers.

It does not create a daily streak, freeze, loss counter, missed-day warning, or reset penalty. Missing a day never erases progress. This preserves the existing Adventure contract that real life, recovery, illness, weather, travel, and lower-capacity days are not failures.

## Authority boundaries

Adventure Trail must never:

- calculate or submit authoritative Bond XP;
- alter Adventure ranking or recommendation evidence;
- gate Today, Story, Health Lens, account controls, or any other product access;
- add health, symptoms, medication, weight, veterinary spending, or CARE actions to the collection game;
- rank dogs or owners by exercise volume, distance, repetitions, or physiological data;
- use post count, reactions, followers, or popularity as Trail progress;
- convert Social Adventure score into relationship truth;
- punish missed days or require a perfect routine.

The human gets a game-shaped sense of unfolding progress. The dog keeps the right to have an ordinary day.

## Native surface

Compass now leads with an **Adventure Trail** card containing:

- the current chapter and chapter-to-chapter progress;
- server-earned Bond XP context;
- seven discovery stamps;
- a clear CARE collection boundary;
- rolling Rhythm paws with explicit anti-reset copy.

The ordinary Compass remains below it and continues to show all canonical pathways as recent opportunity coverage, not a health score.

## Qualification

`Native Adventure Trail CI` verifies on the exact change:

- the presentation policy is deterministic and read-only;
- chapter thresholds remain ordered and start at zero;
- the seven discovery pathways remain the intended set and exclude CARE;
- the Trail derives from `dashboard.bondXp`, `dashboard.compass`, and `dashboard.rhythm`;
- the client contains the explicit no-unlock, no-daily-reset, and CARE-boundary copy;
- the touched TypeScript/TSX, workflow, and documentation are formatted;
- the full native client type-checks;
- native lint remains zero-warning.

Existing repository workflows remain authoritative for broader mobile, security, Xcode, privacy, and dogOS contracts.

## Next game slices

Adventure Trail is deliberately the quiet meta-game, not the whole game design. The next high-value slices are:

- **Expeditions:** bounded seasonal cooperative objectives where the community completes varied safe actions together rather than racing for volume;
- **Friend quests:** opt-in challenge templates that are re-resolved against each recipient's eligibility instead of copying one dog's task to another;
- **Skillcraft on native:** short Human Skill Arcade scenarios that teach timing, reinforcement, setup, and reading behavior without claiming professional proficiency;
- **Story collectibles:** chapter art, memory constellations, and discovery cards derived from already-authorized Story moments rather than upload pressure;
- **Companion progression:** petless Animal Ally and Foster/Caregiver paths with useful human-skill and community goals that never fabricate dog state;
- **responsible adoption readiness:** partner-authorized shelter/foster opportunities and learning quests that make taking the step toward pet guardianship clearer without turning adoption into a reward unlock.

The north star is not more taps. It is making useful dog-human life feel like a world that gradually reveals itself.
