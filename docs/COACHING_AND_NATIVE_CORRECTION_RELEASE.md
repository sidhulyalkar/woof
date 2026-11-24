# Coaching integrity and native observation correction

This tranche implements the first reliability slice of the Woof product evaluation.
It builds on #159 (execution program), #44 (Coach), and #162 (Daily Signals).

## Delivered behavior

- Coach increases difficulty only after two recent, comfortable successes at the
  current level and the same explicitly reported distraction level. Missing
  context cannot authorize an increase. This is a conservative v1 context check,
  not the full skill/environment mastery graph.
- New sessions use `woof-training-session-v2`. v1 history remains readable, but its
  old distraction value (formerly inferred from plan level) cannot authorize
  progression. Dashboard reads show the decision already applied to the plan;
  they do not apply the same progression again.
- Session requests require a UUID `sessionKey` and `expectedPlanUpdatedAt` from the
  plan the owner practiced. PostgreSQL locks the plan row, rechecks pet ownership,
  and writes activity, plan update, and response receipt in one transaction.
- Exact retries return the original receipt, including after pause. Different
  payloads using the same identity conflict. Different sessions using stale plan
  versions also conflict. Current access is required even for replay.
- Web freezes the submission, including duration, on first save and retries it
  without inventing another occurrence. Observation controls lock while a save is
  uncertain. Owners can report a stop before the first repetition and optionally
  choose distraction level instead of having it inferred by the client.
- Native Daily Signals loads current observations, offers explicit correction,
  supports retraction of all structured answers, and keeps original notes private.
  Stale corrections require an explicit reload; delayed responses from a previous
  dog cannot overwrite the current editor.

The post-detail page also reads its route id through `useParams`, resolving a
pre-existing Next.js generated-page type mismatch exposed by build validation.

## Storage and rollout

Apply `20261123000100_coaching_session_receipts` before deploying the API. It adds
`dogos_coaching.session_receipts`, keyed by plan plus client session UUID, with
unique activity linkage and cascading deletion via user, plan, and activity.
The stored response contains no free-form session note. Payload hashes are used
for conflict detection; the Activity remains the session occurrence record.

Deploy API and Web together and require older Web clients to reload: the new
required fields intentionally reject unidentifiable legacy session writes.
Native correction uses already-existing backend endpoints. No production
migration or deployment is performed by this branch.

CareEvent/reward emission retains its existing best-effort controller behavior
and stable activity-based deduplication. Exact retries can repair an emission
failure; a durable reward outbox is not introduced here. Browser/native draft
recovery after process termination is also not claimed.

## Qualification

- Coaching unit regressions: mixed levels, mixed/unknown context, safe stops.
- PostgreSQL integration tests: concurrent identical retries, conflicting payload,
  stale plan versions, progression across levels, replay after ownership change,
  rollback on receipt insertion failure, and zero-repetition stop.
- Web interaction tests: immutable lost-response retry, explicit distraction,
  zero-repetition stop.
- Native component interaction tests with DOM primitives: correction, retries,
  competing edits, full retraction, late response isolation, sparse capture.
- API/Web/mobile TypeScript, touched-code lint, API/Web production builds, and
  existing native source contracts.

Local database tests use PostgreSQL 18 and the Prisma-shaped test schema with
only the unrelated vector field represented as text, plus the actual new migration.
This proves the coaching transaction behavior, not the full historical pgvector
migration chain. Repository CI remains responsible for that chain on PostgreSQL 15.
Native tests prove component state/request behavior, not physical-device layout,
VoiceOver, signing, or App Store readiness.

## Next slices

1. Reviewed, varied Skillcraft scenarios and held-out learning examples.
2. One complete Learn -> practice -> reflection -> next recommendation journey.
3. Full decision receipts (#43) and context-specific mastery (#44).
4. Consistent reward policy and supported Story discoveries (#45/#163).

These remain follow-up work; this branch does not claim the full product roadmap
or a deployed dog-care app is complete.
