# Authentication and session incident runbook

Runbook ID: `auth-session`  
Repository qualification: `CODE_QUALIFIED`  
Live rehearsal status: `NOT_YET_PROVEN`

## Authority boundary

Woof authentication sessions are server-authoritative database records. The maintained user-facing revocation endpoints are:

- `POST /api/v1/auth/logout` to revoke the current authenticated session;
- `POST /api/v1/auth/logout-all` to revoke all active sessions for the current authenticated user.

There is **no maintained operator endpoint that revokes arbitrary users' sessions**. This runbook must not invent one. A broad JWT-secret rotation would invalidate authentication globally and is a high-impact credential/deployment action, not a first-line response.

## Detection

Use this runbook for:

- `auth5xx`
- authentication-related `http5xx`
- authentication-related `application5xxBurst`
- reports of sessions that should have been revoked but remain usable
- suspected JWT signing-secret or session-authority compromise

Use guarded metrics when available:

```bash
export WOOF_API_ORIGIN="https://woof-api-prod.fly.dev"
test -n "${OPS_METRICS_TOKEN:-}" || { echo "OPS_METRICS_TOKEN unavailable" >&2; exit 2; }
curl --fail-with-body --silent --show-error \
  -H "x-woof-ops-token: ${OPS_METRICS_TOKEN}" \
  "$WOOF_API_ORIGIN/api/v1/ops/metrics.json" | jq
```

The `auth5xx` policy covers maintained auth operations including register, login, logout, logout-all, and profile retrieval. A 401/403 increase is not automatically an availability incident; distinguish expected authentication rejection from server failure.

## Impact

Classify:

- **login availability:** valid users cannot authenticate;
- **session validation:** issued sessions cannot be checked or are incorrectly rejected;
- **revocation failure:** logout/logout-all does not invalidate expected sessions;
- **credential compromise:** signing secret, password credential, token, or session material may be exposed;
- **authorization confusion:** authentication succeeds but the wrong principal/authority is used.

Treat suspected credential compromise as a security incident even if availability metrics are green.

## Containment

1. Preserve auth error-rate and readiness evidence without copying tokens or passwords.
2. If only one user suspects compromise and can authenticate, direct that user through the maintained `logout-all` authority.
3. If an individual user cannot authenticate and arbitrary operator revocation is required, escalate to an authorized database/security procedure. Do not fabricate an admin API.
4. If the JWT signing secret itself may be compromised, stop ordinary release work and escalate to production secret-rotation authority.
5. Do not disable session-authority checks to restore login availability.
6. Do not change authentication failures into successful anonymous access.

## Diagnosis

Check general runtime health first:

```bash
curl --fail-with-body --silent --show-error \
  "$WOOF_API_ORIGIN/api/v1/ops/health/ready" | jq
```

Then determine whether the failure is:

- before credential verification;
- during password verification;
- during session-row creation;
- during JWT issuance;
- during `SessionAuthorityService.assertActive` / session lookup;
- during logout/revocation;
- a general database/runtime failure affecting auth incidentally.

For user-driven revocation, the real API boundary is:

```text
POST /api/v1/auth/logout-all
Authorization: Bearer <current-user-token>
```

Do not place a real token in a runbook transcript. Validate this flow only with an authorized synthetic/test account during rehearsal.

Record the active application `releaseSha`, auth alert window, and whether database readiness is healthy. If auth 5xx coincides with general 5xx/readiness failure, follow `deployment-readiness.md` and `database-migration.md` before making auth-specific changes.

## Recovery

### User-scoped compromise

Use the maintained authenticated `logout-all` flow when the user can still authenticate. Then require reauthentication and verify old sessions are rejected.

### Auth availability regression

Prefer a reviewed forward fix or exact previous-image recovery after database compatibility review. Session authority must remain enabled during recovery.

### JWT signing-secret compromise

Rotate the signing secret only through the authorized production secret-management/deployment path. Treat this as global session invalidation:

- expect all currently issued JWTs to stop authenticating;
- communicate the forced sign-in impact;
- redeploy/restart the runtime as required by the platform so the new secret is authoritative;
- never write the replacement secret into the repository, PR, issue, incident note, or shell transcript;
- verify both newly issued sessions and rejection of tokens signed under the previous key.

Because current production secret authority is an external blocker, this repository runbook does not claim that rotation can be executed from this connector session.

## Rollback

Rollback auth code through the same exact-image procedure as `deployment-readiness.md`, subject to database/schema compatibility.

Do not roll back a compromised JWT secret to its previous value. Secret rotation is a one-way security boundary even if the application code is rolled back.

Do not restore previously revoked session rows to recover availability.

## Verification

Use an authorized synthetic account and verify:

1. login succeeds with correct credentials;
2. `/api/v1/auth/me` succeeds for the new session;
3. `POST /api/v1/auth/logout` rejects subsequent use of that revoked session;
4. a newly logged-in session can call `POST /api/v1/auth/logout-all`;
5. every session for that synthetic user created before logout-all is subsequently rejected;
6. unrelated users are not revoked during user-scoped recovery;
7. auth 5xx alert windows clear;
8. readiness remains healthy and reports the intended `releaseSha`.

For a global secret rotation, additionally prove an old-key token is rejected and a new-key token is accepted.

## Evidence

Capture only:

- alert key/window;
- release SHA;
- synthetic account identifier or fixture label, not a real user identity;
- revocation action type (`logout`, `logout-all`, global key rotation);
- bounded HTTP status outcomes;
- recovery SHA/image;
- verification timestamps.

Never capture passwords, JWTs, session IDs, authorization headers, password hashes, signing secrets, or raw database session rows.

## Rehearsal

Live rehearsal status: `NOT_YET_PROVEN`

Promotion to `PROVEN` requires a production-like synthetic-user drill demonstrating login, current-session revocation, all-session revocation, and post-revocation rejection. Global JWT rotation requires a separately authorized drill because it intentionally invalidates all active authentication.
