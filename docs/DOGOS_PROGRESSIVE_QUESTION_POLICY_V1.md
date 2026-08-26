# dogOS Progressive Question Policy v1

`profile-question-policy-v1` is the first executable policy under Adaptive Adventure issue #47 / parent #42.

Its job is intentionally narrow:

> Given bounded profile evidence, recent quest interactions, question history, and an explicit clock, decide whether **one optional question** is worth asking now.

It does not persist profile state, rank quests, award Bond XP, call ML, or decide medical/safety advice.

## Why this exists

The current onboarding quiz is already small, but its three questions are broad: schedule, general activity level, and preferred activities. Adaptive Adventure needs higher-value state such as goals, realistic time/effort, social and novelty comfort, reinforcers, obvious dislikes, available environments, and training experience without expanding onboarding into a long form.

The intended product loop is therefore:

`small First Adventure profile -> useful quest -> observed outcome -> optional high-value micro-question -> stronger pair model`

A missing fact does not automatically justify a question. User attention has a cost.

## Inputs

The policy receives only explicit data:

- profile evidence by dimension;
- provenance and confidence;
- recent bounded quest interactions;
- previous question outcomes;
- explicit `now`.

There is no implicit clock, database, environment variable, network call, random number, or ML inference.

### Durable dimensions

- owner goals
- owner time budget
- owner effort preference
- available environments
- dog energy pattern
- dog social comfort
- dog novelty comfort
- dog reinforcers
- dog obvious dislikes
- training experience

Known restrictions and safety exclusions are deliberately outside this casual question policy. They belong to explicit safety/profile capture and hard eligibility gates.

## Authority rules

1. Invalid or future-dated evidence cannot suppress a question.
2. Explicit owner corrections outrank inferred behavioral confidence.
3. High-confidence known dimensions are not re-asked.
4. Repeated positive social/training outcomes may eventually provide enough evidence to suppress a redundant generic question.
5. A single dismissal never becomes a durable dislike.
6. Two related dismissals may trigger a preference-versus-timing clarification.
7. Safe opt-outs are never counted as dislike evidence.
8. A recent positive training completion may trigger one low-burden difficulty question.
9. Recent `SKIPPED` and `NOT_SURE` answers enter cooldowns so Woof does not nag.
10. The policy returns at most one question.

## Value-of-information proxy

Eligible candidates receive a deterministic score from:

- decision value;
- remaining uncertainty;
- contextual trigger strength;
- eligibility relevance;
- burden penalty.

The score is a ranking receipt, not a probability and not a user-facing quality grade. Equal scores tie-break by stable question ID.

## Two economies remain separate

This policy has no Bond XP, reward amount, streak, retention, click-through, or game-currency target.

Game rewards can make Adventure fun. Training labels must describe what actually happened for the dog and human. Feeding our own XP schedule back into personalization would create a self-referential objective.

## Version receipt

Policy version: `profile-question-policy-v1`

Canonical receipt SHA-256:

`dca2936d547338e58d5911006e575d8394d685e7679c2d52ad5cb24e07cd0483`

The Jest contract recomputes this hash and verifies that the canonical receipt and exported policy object remain equivalent.

## Non-goals

This release does not add profile persistence, a REST endpoint, First Adventure UI, learned active questioning, a skill graph, contextual-bandit exploration, or a deep personalization model.

Those layers should consume this oracle after it qualifies, not duplicate its decision semantics in UI code.
