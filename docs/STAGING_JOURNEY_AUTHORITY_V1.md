# Staging Journey Authority v1

## Purpose

A successful staging release must prove more than liveness, database readiness, and release provenance. It must prove that the deployed public API can execute a minimal real user lifecycle through the same authorization boundaries a normal client uses.

This contract is intentionally small. It is a release qualification, not a synthetic product benchmark.

## Qualified lifecycle

After Web/API provenance and the non-destructive live smoke pass, staging runs a disposable synthetic journey against the public API:

1. register a unique email account with a replay-safe `registrationKey`;
2. use the returned access token to load `/auth/me`;
3. create a minimal owned dog with a replay-safe `creationKey`;
4. mutate that dog through the normal owner-authorized update route;
5. delete the account through `DELETE /users/me`;
6. prove the deleted access token is rejected;
7. prove the deleted email/password credentials are rejected.

The staging release receipt is retained only after this job succeeds because receipt creation depends on the entire staging Web job.

## Privacy and cleanup

The verifier must not print the synthetic email, password, bearer token, response bodies, database URLs, or database records.

The account is disposable and the canonical account-deletion API is the cleanup authority. If any step fails after registration, a `finally` path attempts `DELETE /users/me` using the synthetic session. There is no direct database cleanup and no test-only API.

Synthetic identifiers are unique per run and exist only long enough to exercise the journey.

## Evidence boundary

Repository CI proves the runner is parseable, its success path works under a fake HTTP transport, failure after registration triggers cleanup, and the staging workflow orders the journey after live smoke and before release receipt authority.

Repository CI does **not** prove the deployed staging environment works. That evidence exists only when the real staging workflow executes the journey successfully against the deployed API.

A green staging receipt therefore means, transitively:

- exact-SHA canonical main release checks passed;
- API migration/deployment and release identity passed;
- immutable and stable Web provenance passed;
- non-destructive live release smoke passed;
- the disposable public-API lifecycle passed;
- the receipt was emitted only after those authorities succeeded.

## Non-goals

v1 does not create caregiver relationships, community posts, Packs, media, push subscriptions, or realtime conversations. Those surfaces have their own repository authorities and should only be added to deployed synthetic qualification when their cleanup semantics and operational value justify the extra mutable staging state.

Production promotion does not rerun this mutation directly against production. Instead, production requires a successful staging workflow for the exact same SHA and independently re-verifies canonical release authority before promotion.
