# dogOS Community Events reward authority

Community event attendance and venue feedback are **acknowledgement-only** surfaces.

## Active authority

- `EventsService.checkIn` records `eventRSVP.checkedInAt` and returns a participation acknowledgement.
- `EventsService.submitFeedback` creates or updates `eventFeedback` rows for community signal quality.
- Duplicate check-ins are rejected before any write; feedback updates are idempotent per `(eventId, userId)`.

## Retired coupling

Community Events must **not** call `GamificationService.awardPoints`, mutate `users.totalPoints`, or return `pointsAwarded` / "You earned points" copy.

Legacy gamification remains **read-only** through `GET /gamification/me/summary` for historical profile compatibility. Adventure/Bond rewards continue through `CareEventsService` and the Adventure system.

## Remaining legacy writers (historical read path)

| Surface | Role |
| --- | --- |
| `GET /gamification/me/summary` | Read-only legacy totals, badges, streaks |
| `CareEventsService` / Adventure | Canonical Bond XP authority (not community events) |
| `GamificationService.awardPoints` | Retained for compatibility only; no live community callers |

New event or social code must not reintroduce universal point awards. CI enforces `awardPoints` absence under `apps/api/src/events`.
