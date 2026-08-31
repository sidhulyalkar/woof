# Connector ingestion incident runbook

Runbook ID: `connector-ingestion`  
Repository qualification: `CODE_QUALIFIED`  
Live rehearsal status: `NOT_YET_PROVEN`

## Authority boundary

Woof connector ingestion is a verified internal transport seam, not a browser-supplied provider-data API. Production connector routes are hidden unless `ENABLE_DOGOS_CONNECTORS=true`, and enabling connectors in production requires a valid 32-byte `CONNECTOR_CREDENTIALS_KEY`.

Current connector authority deliberately preserves these boundaries:

- undocumented OAuth is not allowed;
- browser provider impersonation is not allowed;
- raw provider payloads are not stored;
- provider observations must map to an owned Woof pet;
- verified imports are idempotent through persisted import receipts and canonical payload hashes;
- imported wearable observations are not reward eligible;
- local disconnect removes local credentials and records local revocation, but remote provider revocation is currently `NOT_CONFIGURED`.

Do not describe a local disconnect as remote OAuth/provider revocation.

## Detection

Use this runbook for:

- `connectorRejected`;
- connector-related `http5xx`;
- connector-related `application5xxBurst`;
- provider/kind-specific import drift;
- connector credentials becoming `REAUTH_REQUIRED`;
- verified device transport failures after an envelope has a trusted provider identity.

Use the guarded metrics boundary when available:

```bash
export WOOF_API_ORIGIN="https://woof-api-prod.fly.dev"
test -n "${OPS_METRICS_TOKEN:-}" || { echo "OPS_METRICS_TOKEN unavailable" >&2; exit 2; }
curl --fail-with-body --silent --show-error \
  -H "x-woof-ops-token: ${OPS_METRICS_TOKEN}" \
  "$WOOF_API_ORIGIN/api/v1/ops/metrics.json" | jq
```

Connector metrics intentionally expose only verified provider, observation kind, outcome, count, and bounded timing aggregates. Do not add external account IDs, external object IDs, user IDs, pet IDs, or raw payloads for diagnosis.

Pre-provider envelope rejection belongs to `deviceContractRejections`; use `api-degradation.md` if the failure occurs before provider identity is trusted.

## Impact

Classify:

- **transport contract drift:** verified provider envelopes no longer normalize under the maintained contract;
- **credential state failure:** encrypted local credential cannot be used and connection is moved to `REAUTH_REQUIRED`;
- **identity mapping failure:** external pet identity is not mapped to an owned Woof pet;
- **idempotency conflict:** the same provider external object ID arrives with changed canonical content;
- **persistence failure:** canonical observation succeeds or begins but the import receipt cannot be persisted consistently;
- **credential compromise:** connector encryption material or a stored provider credential may have been exposed;
- **provider outage:** an external transport is unavailable without a Woof contract regression.

A rejected import is not evidence that canonical pet data should be mutated manually.

## Containment

1. If connector ingestion cannot be trusted, prefer setting `ENABLE_DOGOS_CONNECTORS=false` through authorized production configuration and redeploying/restarting as required. The production guard then returns 404 rather than exposing a degraded pseudo-integration.
2. Do not enable an undocumented OAuth path to work around partner/provider failure.
3. Do not allow browser callers to submit provider-owned observations directly.
4. Do not bypass credential-state checks, pet identity mapping, import receipts, or payload-hash conflict detection.
5. Do not mark imported wearable data reward eligible to compensate for ingestion issues.
6. If credential compromise is suspected, also invoke `privacy-telemetry.md`. Do not log, export, or copy decrypted credential material for debugging.
7. Do not claim remote revocation after a local disconnect. Follow the external provider's authorized revocation path only when a verified runtime exists.

## Diagnosis

Start with API readiness and exact release identity. Then classify by trusted provider + kind only:

- `DAILY_ACTIVITY`;
- `DEVICE_STATUS`.

Inspect whether the connection is `CONNECTED`, `REAUTH_REQUIRED`, or locally `REVOKED` through an authorized synthetic account. Never paste real connection records or credentials into incident notes.

For an idempotency incident, use synthetic transport fixtures and verify these three cases:

1. first import of a new external object creates the canonical result and import receipt;
2. exact replay returns the existing semantic result rather than duplicating canonical state;
3. the same external object ID with different canonical content is rejected as `external_object_changed_after_import`.

For a credential-state incident, distinguish a missing/invalid encryption key from an expired/revoked provider credential. Production connector startup/configuration authority must remain fail closed when the connector encryption key is invalid.

For apparent provider failure, do not infer provider payload details from metrics. Use the provider's separately authorized operational surface when available.

## Recovery

### Contract or application regression

Ship a reviewed forward fix that preserves the versioned device contract, identity mapping, idempotent receipt semantics, privacy boundaries, and no-reward policy. Use exact-image recovery from `deployment-readiness.md` only after database compatibility is confirmed.

### Credential requires reauthorization

Leave the connection in `REAUTH_REQUIRED` until a verified provider transport can obtain valid credentials. Do not fabricate or manually edit provider credentials.

### Connector encryption key compromise

Treat the key as compromised secret material. Rotate it through authorized production secret authority and define an explicit credential migration/re-authentication plan. Existing encrypted envelopes cannot be assumed decryptable under a replacement key. Do not retain the old key merely to preserve convenience after compromise.

### Provider outage

Keep provider-owned ingestion degraded or disabled without altering canonical Woof data. Restore normal ingestion only after the verified transport succeeds again.

## Rollback

Rollback code through `deployment-readiness.md`. Do not roll back:

- a compromised credential-encryption key;
- an already recorded local revocation merely to make a connector look connected;
- payload hashes/import receipts to force a conflicting external object through;
- canonical pet ownership or identity mappings by hand;
- the rule that imported wearable observations are not reward eligible.

## Verification

Using synthetic provider/account/pet fixtures, require:

1. production-disabled connector routes remain unavailable when `ENABLE_DOGOS_CONNECTORS=false`;
2. a configured connector still requires usable encrypted credentials;
3. a verified provider pet must map to a pet owned by the authenticated Woof user;
4. a first valid import produces one canonical observation and one semantic import receipt;
5. an exact replay remains duplicate-safe;
6. changed content under the same external object ID is rejected;
7. provider/kind rejection metrics recover without adding identifying labels;
8. local disconnect removes local credentials, marks the connection locally revoked, and continues to report remote revocation as `NOT_CONFIGURED` unless a real remote path has since been implemented;
9. imported wearable observations remain non-reward-eligible.

## Evidence

Capture only:

- release SHA;
- provider enum and observation kind;
- outcome class (`IMPORTED`, `DUPLICATE`, `REJECTED`);
- alert threshold/window;
- synthetic fixture labels;
- bounded HTTP/result classification;
- recovery SHA and verification timestamps.

Never capture provider access/refresh tokens, connector encryption keys, external account IDs, external object IDs, external pet IDs, raw provider payloads, user IDs, pet IDs, or decrypted credential envelopes.

## Rehearsal

Live rehearsal status: `NOT_YET_PROVEN`

Promotion to `PROVEN` requires a production-like synthetic transport drill demonstrating valid import, exact replay, changed-object conflict rejection, credential degradation, local disconnect semantics, and safe connector disablement without changing reward authority.