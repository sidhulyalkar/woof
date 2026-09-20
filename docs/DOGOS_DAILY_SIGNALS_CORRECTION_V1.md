# Daily Signals Correction Authority v1

## Purpose

Daily Signals capture intentionally makes one structured check-in canonical per dog + household-local day. v1 capture rejects a later different payload rather than silently overwriting evidence.

Correction v1 adds an explicit, append-only way to repair that structured evidence.

## Canonical chain

The original remains immutable:

`DAILY_SIGNALS_CHECKIN`

Corrections append:

`DAILY_SIGNALS_CORRECTION`

Each correction records:

- root CareEvent id;
- immediate predecessor CareEvent id;
- monotonically increasing correction sequence;
- household id;
- local date and authoritative household timezone;
- correction policy version;
- deterministic structured-payload hash.

The client must send the CareEvent it believes is current. A stale client cannot silently create another correction.

## Idempotence and concurrency

Logical correction identity is:

`daily-signals-correction:<rootCareEventId>:v<sequence>`

CareEvents uses PET-scoped PostgreSQL serialization for this identity.

An exact retry of the already-created immediate successor is accepted as a duplicate and re-runs projection convergence. A different payload at the same sequence conflicts.

If the chain has advanced beyond that immediate successor, the request is stale and must reload.

## Full structured state

A correction is the complete intended structured signal state for the day, not a patch.

It may contain:

- LESS
- USUAL
- MORE
- UNSURE
- omitted dimensions

Unlike first capture, correction may intentionally contain zero reported dimensions. That means the user corrected the structured evidence to “nothing should currently contribute from this check-in.”

## Projection

CareEvents are canonical. Intelligence observations are derived.

Every correction reconciliation walks the entire canonical chain from the original event through the current tip.

For a reported LESS / USUAL / MORE dimension:

- a new OWNER_CHECKIN observation is normalized from the correction CareEvent;
- it supersedes the prior chain observation for that dimension when one exists;
- it keeps the original local-day chronology.

For UNSURE or an omitted dimension:

- no replacement baseline observation is created;
- prior active evidence for that dimension is retracted.

Replaying the same chain is idempotent and can repair missing derived projection rows.

## Privacy

Correction v1 owns **structured signal state only**.

The correction DTO has no note field.

A free-form note written on the original private CareEvent:

- is never returned by the effective-state endpoint;
- is never copied to a correction event;
- never enters baseline projection;
- cannot be replaced or cleared by another household member through correction authority.

Future note editing requires a separate author-scoped policy.

## Reward boundary

Correction events set `safetyEligible: false` and issue zero Bond XP.

Correcting evidence is data repair, not a game action.

## Evidence boundary

Repository qualification proves append-only chain, authorization, concurrency, retry, projection supersession/retraction, privacy and zero-reward contracts in the qualified test environment.

It does not prove deployed staging behavior, native correction UX, physical-device usability or pilot comprehension.
