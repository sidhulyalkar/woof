# dogOS Adventure History Integrity

Adventure ranking can only become trainable if its historical evidence cannot silently change after an interaction occurs.

## Problem

`QuestInteraction` is an event-like record keyed by:

`user + pet + quest + interaction`

The application intentionally uses `INSERT ... ON CONFLICT DO UPDATE` so network retries converge instead of producing duplicate rows. Before this contract, that retry path also meant a later request could replace `pathway`, replace `context`, and refresh `created_at` for an interaction that had already happened.

That is unacceptable as a substrate for counterfactual recommendation evaluation. Historical evidence must not acquire a different rank, receipt identity, pathway, or timestamp after downstream outcomes are known.

## Database authority

The database now owns the immutability boundary with a `BEFORE UPDATE` trigger on `quest_interactions`.

An attempted update has only two outcomes:

1. **Exact semantic retry.** Identity, pathway, and context match the stored row. PostgreSQL returns the existing row unchanged, preserving the original `id` and `created_at`.
2. **Divergent retry or mutation.** Any historical field differs. PostgreSQL rejects the update with SQLSTATE `23514`.

The trigger intentionally allows deletion. Account/pet cascade deletion and privacy workflows must remain able to remove the user's historical data.

## Why the boundary lives in PostgreSQL

Application-only checks are insufficient because:

- multiple API processes can race,
- a future maintenance script could bypass one service method,
- `ON CONFLICT DO UPDATE` is itself executed atomically by PostgreSQL,
- the history contract should survive refactors of Adventure service code.

The database therefore provides the final fail-closed boundary while the application keeps its existing idempotent insert interface.

## What this does not solve

This release is a prerequisite for the broader counterfactual decision/outcome ledger in issue #43. It does **not** yet persist the complete quest deck, candidate eligibility/reason codes, deterministic scores, rank order, feature-schema identity, exploration budget, or selection propensity.

The correct next layer is an immutable server-generated deck receipt created before any selection/outcome. Existing `QuestInteraction` rows can then reference that receipt without being able to rewrite it later.

## Qualification

`dogOS Adventure History Integrity CI` starts PostgreSQL, applies the complete canonical migration history, and proves:

- an initial interaction can be recorded,
- an exact conflict retry converges to one row,
- the original timestamp is preserved,
- a divergent retry fails closed,
- a direct historical mutation fails,
- user cascade deletion still removes the interaction,
- the migration retains its `RETURN OLD` no-op path and explicit integrity-error contract.

This is intentionally a database integration contract rather than a source-text-only assertion.
