# dogOS Meetup Outcome Authority v1

## Product boundary

The social loop is only trustworthy if real-world outcomes have stronger authority than engagement telemetry.

This tranche makes the post-meetup reflection participant-scoped:

`discover -> chat -> propose -> accept -> meet -> private reflection -> coordinate again`

## Canonical model

`MeetupOutcome` is the canonical participant record. There can be exactly one row per
`(proposalId, participantId)`.

It stores the participant's own:

- whether the meetup occurred;
- dog experience;
- owner experience;
- meet-again intent;
- optional rating/tags;
- safety answer;
- optional private note.

The shared `MeetupProposal` remains coordination state. New outcome writes do not aggregate ratings,
feedback tags, safety answers, or private notes back onto that row.

## Concurrency and retry

- database uniqueness is the final duplicate boundary;
- exact retries converge to the existing outcome;
- divergent retries fail with conflict instead of overwriting;
- concurrent same-participant submissions cannot produce two canonical outcomes;
- acceptance uses a guarded `pending -> accepted|declined` transition.

## Shared-state semantics

A positive occurrence report may promote an accepted proposal to `completed`.

A single participant reporting `occurred=false` does **not** cancel the shared proposal. Cancellation
remains an explicit coordination action, not an inference from one person's private reflection.

## Privacy

Each participant can read only their own outcome through authenticated participant authority.
The other participant's `meetAgain`, safety answer, rating, tags, and private note are not returned.

Telemetry may record that an outcome was submitted, but telemetry is observational and is not the
canonical outcome ledger.

## Repeat planning

The completion response exposes `repeatPlanningEligible` only from the current participant's own
outcome. It never implies mutual interest.

## Evidence boundary

Repository qualification can prove migrations, uniqueness, authorization, retry/concurrency behavior,
shared-state transitions, and API/client contracts.

It does not prove that a meetup happened, that feedback is truthful, or that two people mutually want
another meetup. Those remain real-world/pilot evidence.
