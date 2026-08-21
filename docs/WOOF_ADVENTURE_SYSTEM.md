# Woof Adventure System

Woof Adventure turns the product from a collection of pet features into one longitudinal game for the dog-owner pair.

> The real dog is the character. The real world is the game world. The relationship is the progression.

The system is intentionally designed around **individual fit**, not maximum exercise, maximum social exposure, screen time, or raw engagement.

## Product surfaces

| Surface     | Role                                                                                           |
| ----------- | ---------------------------------------------------------------------------------------------- |
| **Today**   | Personalized three-card Quest Deck with one recommended quest, an alternative, and a wildcard  |
| **Compass** | Eight-pathway recent opportunity coverage, explicitly not a health score                       |
| **Journey** | Adventure Book built on the private Media Library                                              |
| **Coach**   | Reward-based Learn progression and dog-literacy feedback                                       |
| **Pack**    | Compatibility-first social discovery and cooperative challenges                                |
| **Health**  | Separate high-stakes Health Lens; emergency/illness flows are outside competitive gamification |

### Pawprint Compass

The eight pathways are:

- `MOVE` — movement and conditioning
- `EXPLORE` — nature, novelty, sniffing, sensory exploration
- `ENRICH` — scent work, searching, puzzles, foraging
- `LEARN` — reward-based training and communication
- `CONNECT` — comfortable social experiences
- `CARE` — private preventive-care support
- `RECOVER` — rest and decompression
- `BOND` — shared experiences that fit both halves of the pair

Compass bars are **recent opportunity coverage**. They must never be described as health percentages, ownership quality, or veterinary assessment.

## Daily Quest Engine

`GET /api/v1/adventure/me` composes the existing Insights engine with Adventure-specific pathway coverage and recent outcome learning.

The first version is deliberately rule-based and explainable:

1. generate safe candidate quests from current Insights plus evergreen templates;
2. map each candidate onto one primary wellbeing pathway;
3. reward under-covered pathways without treating a full Compass as a universal prescription;
4. apply a tightly capped recent dog-owner preference signal;
5. prefer three different primary pathways when possible;
6. generate deterministic daily quest IDs so clients cannot invent reward-bearing quests;
7. let the owner choose.

A future contextual bandit can replace the ranking layer only after sufficient real outcomes exist. It should optimize downstream fit and safety rather than app engagement.

## Five-second outcome loop

Every quest can close with two tiny questions:

**Dog**

- Loved it
- Comfortable
- Not their thing

**Owner**

- Great
- Fine
- A lot today

For eligible social/training quests, `I listened and stopped` is a successful completion. It records a `SAFE_OPT_OUT` event on the `BOND` pathway rather than rewarding forced completion.

A normal `not_their_thing` result is treated as preference evidence and is also credited to `BOND`, not to the original pathway. This prevents a stressful social interaction from inflating `CONNECT` progress simply because the owner pressed “complete.”

## Trusted reward architecture

The old client-controlled `/gamification/points`, `/badges`, `/streaks`, and raw leaderboard mutation/read surfaces are removed from the controller.

The authoritative flow is:

```text
trusted domain action
      ↓
CareEvent
      ↓
safety eligibility
      ↓
server RewardPolicy
      ↓
immutable RewardLedger
      ↓
Bond XP + pathway history
```

The client never submits an XP amount.

### Database

Migration:

`packages/database/prisma/migrations/20260821223000_add_woof_adventure_system/migration.sql`

Tables:

- `care_events`
- `reward_ledger`
- `quest_interactions`

Important properties:

- `(user_id, dedupe_key)` is unique for idempotency;
- each reward ledger row maps one-to-one to a CareEvent;
- evidence confidence is constrained to `[0,1]`;
- pathway and visibility values are database-constrained;
- high-value calculations happen in server code, not DTOs;
- new Adventure reads are ledger-derived;
- `users.total_points` is updated only as a temporary compatibility aggregate for old surfaces.

## Reward policy

Policy version: `bond-xp-v1`.

The starting policy uses fixed event values plus conservative modifiers:

- evidence confidence can only make a very small difference;
- optional media is a tiny bonus;
- novelty is capped;
- repeated same-pathway actions decay;
- repeated event types decay over seven days;
- per-pathway and whole-day XP caps prevent volume farming;
- safety-ineligible events issue zero XP;
- recovery earns full legitimate credit;
- safe opt-outs earn meaningful Bond XP.

The system deliberately does **not** multiply reward by distance, duration, calories, money spent, or number of photos.

## Domain integrations

### Activities

Completed activities with an owned pet emit trusted CareEvents. Current semantics include walking, running, hiking, enrichment, training, social outings, parallel walks, and recovery/decompression.

Activity completion remains reliable during a rolling deployment even if reward emission fails. The activity is saved first and reward emission is additive.

### Woof Coach

Coach sessions now feed the same reward ledger:

- comfortable training session → `TRAINING_SESSION / LEARN`;
- stress signal + stopped early → `SAFE_OPT_OUT / BOND`;
- concern signals without a safe stop → the session is stored but reward eligibility fails closed.

Coach therefore rewards communication and judgment rather than repetitions at any cost.

### Media Library / Journey

Journey reads the existing private Media Library and albums. Photos remain optional. The experience is valuable even with no media.

The reward policy caps any memory bonus so the system cannot become a photo farming game.

### Pack

`GET /api/v1/pack/challenges` exposes aggregate cooperative challenges. It returns community totals, contributor count, and the current user's contribution without publishing individual raw rankings.

The first challenges are:

- Sniff & Explore Week
- Recovery Counts

The social layer keeps compatibility-first discovery and shared events available while avoiding a “most miles wins” leaderboard.

## Rhythm replaces streak pressure

Adventure summary looks at meaningful activity across a rolling five-week window and describes the result as **Rhythm**.

Missing a day never zeroes a streak. Recovery and other legitimate pathways can preserve rhythm.

## Health boundary

Health Lens stays structurally separate from the reward engine.

Never gamify:

- emergency severity;
- diagnosis or symptom seriousness;
- medications;
- weight loss rankings;
- veterinary spending;
- forced proof photos;
- fear/aggression as failure.

Emergency Health Lens UX should remain free of XP, confetti, “quest completed” copy, or competitive surfaces.

Preventive-care logging can later emit private, narrowly eligible `CARE` events, but clinical urgency must never enter the game economy.

## Analytics north star

The redesign should shift platform-level success toward **Weekly Meaningful Dyad Actions**.

A meaningful dyad action has:

1. a real action with or for the dog;
2. a legitimate wellbeing pathway;
3. passed safety eligibility;
4. meaningful completion or outcome context.

Recommended supporting metrics:

- quest acceptance → completion;
- comfortable/positive dog outcomes;
- owner enjoyment;
- repeated preferred experiences;
- pathway diversity without universal quotas;
- training progression with low stress;
- recovery adoption;
- safe-stop behavior;
- positive social repeat rate;
- nature/exploration frequency;
- 30/90-day retained dyads.

Do not optimize the quest policy for screen time, posts, likes, or notification opens.

## Rollout

1. deploy database migration before enabling the Adventure UI;
2. deploy API with Adventure + CareEvent modules;
3. verify the old client-controlled gamification mutation routes are absent;
4. smoke `GET /adventure/me` with an existing pet;
5. complete one quest and verify exactly one CareEvent + ledger row;
6. retry the completion and confirm idempotent duplicate behavior;
7. log a completed walk and verify trusted Activity emission;
8. record a comfortable Coach session and verify Learn credit;
9. record a Coach stress + stopped-early session and verify Bond safe-stop credit;
10. verify Health Lens emergency UX contains no game reward treatment;
11. validate Compass copy says opportunity coverage rather than health score;
12. verify Pack exposes aggregate cooperation rather than raw individual rankings.

## Future work after real-world outcome collection

- richer owner preference learning and availability context;
- route semantics, terrain, temperature, place novelty, and favorite-place inference;
- private preventive Care Journey milestones;
- household / multi-dog quests;
- friend-scoped personalized leagues based on quest completion percentage rather than raw exercise volume;
- configurable seasonal cooperative Pack challenges;
- contextual quest ranking evaluated against the explainable rule-based baseline;
- veterinary handoff built from structured longitudinal history, with explicit owner control.
