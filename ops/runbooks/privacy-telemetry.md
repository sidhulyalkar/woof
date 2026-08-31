# Privacy and telemetry incident runbook

Runbook ID: `privacy-telemetry`  
Repository qualification: `CODE_QUALIFIED`  
Live rehearsal status: `NOT_YET_PROVEN`

## Authority boundary

This runbook covers suspected exposure of private user/pet content, credentials, provider diagnostics, object identifiers, push subscription material, or telemetry that violates Woof's privacy-minimized operational contracts.

Current code-qualified boundaries include:

- operational metrics are low-cardinality and declare that user IDs, pet IDs, provider external IDs, request URLs, and raw payloads are not collected;
- API Sentry is configured with `sendDefaultPii: false` and strips request, user, extra, breadcrumbs, span data, and free-form span descriptions before egress;
- Health Lens and Behavior Vision reduce provider failures to bounded classes rather than reading/logging private provider error bodies or arbitrary exception messages;
- private Storage suppresses object-key/filename/provider-detail telemetry and normalizes raw S3 errors before they escape upward;
- Web Push telemetry suppresses user IDs, notification content, endpoint/key material, and provider exception content;
- external integration configuration or CI evidence is not production qualification.

This repository does not itself establish live provider retention, routing, deletion, or secret-rotation authority. Those require the deployed provider/platform controls.

## Detection

Use this runbook for manual signal `privacy_or_secret_exposure`, including:

- a credential or token visible in logs, issue trackers, CI output, screenshots, chat, or telemetry;
- user/pet/request/provider payload content visible in operational metrics or Sentry;
- Health/Behavior provider bodies or arbitrary exception details escaping the bounded failure boundary;
- private object keys or filenames appearing in logs;
- push endpoint, `p256dh`, `auth`, title, body, or user identity appearing in telemetry;
- an integration being represented as production-qualified without live deployment evidence.

Do not reproduce the suspect sensitive value in the incident record to prove the incident. Record its **class**, location, time range, release SHA, and bounded count instead.

## Impact

Classify the incident:

- **secret exposure:** JWT, operational metrics token, connector credential-encryption key, provider API key/token, S3 credentials, VAPID private key, deployment token, or equivalent secret;
- **private payload exposure:** health concern/image context, behavior evidence, notification content, request body/query, caregiver note, chat content, or provider payload;
- **identifier exposure:** user/pet IDs, private object keys, push endpoints/keys, provider external IDs, session IDs;
- **provider diagnostic exposure:** arbitrary upstream error body/message/stack that can contain request/private context;
- **status inflation:** repository/configuration evidence is represented as live production evidence;
- **unknown scope:** evidence is incomplete. Treat scope as unresolved, not zero.

A privacy incident can be severe even when availability and error-rate metrics are green.

## Containment

1. Stop further exposure through the narrowest qualified boundary available.
2. If one optional integration is responsible, disable that integration through its maintained configuration boundary rather than weakening global security/privacy controls.
3. If a secret may be exposed, rotate or revoke it through the owning production/provider authority as soon as that authority is available. Do not wait for code cleanup before invalidating a usable exposed credential.
4. If the connector encryption key is compromised, invoke the migration/reauthorization implications in `connector-ingestion.md`; do not merely replace the key and assume old ciphertext remains usable.
5. If JWT signing material is compromised, follow `auth-session.md` and treat rotation as global session invalidation.
6. Do not paste raw private payloads or secrets into GitHub issues, PRs, Slack/chatops, email, or new debugging logs.
7. Do not disable Sentry scrubbers, operational metric privacy rules, Storage normalization, Health/Behavior failure normalization, or Push telemetry suppression to gain diagnostic detail.
8. Preserve low-sensitivity metadata before deleting/rotating provider-side evidence when policy and authority permit.

## Diagnosis

Record:

- exact application release SHA;
- first/last observed timestamps;
- exposure class from the Impact section;
- code path or telemetry destination;
- whether the exposed value was usable as a credential or only private content/identifier;
- whether the source was application-owned, CI, provider-owned, or manually copied;
- bounded count of affected events when available without querying raw private payloads.

Use synthetic marker tests to reproduce the **class** of exposure. Never replay a real secret or private user payload.

For Sentry-related incidents, inspect source authority first: `scrubSentryEvent` removes request/user/extra/breadcrumbs, while transaction scrubbing removes span data and free-form descriptions. A live-provider verification still requires authorized Sentry access and is separate evidence.

For integration telemetry, compare against `docs/EXTERNAL_INTEGRATION_INVENTORY.json`. Every current external runtime remains `productionQualified: false` unless a later release explicitly changes that classification with live evidence.

For Web Push data at rest, note that migration of persisted subscription material to encrypted envelopes is a distinct data-migration concern. Treat exposed endpoint/key material as sensitive regardless of whether that migration has completed.

## Recovery

### Code-owned telemetry regression

Remove the sensitive field or raw diagnostic at the earliest owning boundary, add a synthetic privacy test, and add/strengthen a fail-closed source contract so the same pattern cannot quietly return.

### Exposed credential

Rotate/revoke through the authoritative external secret/provider path. Then update deployment/runtime configuration without committing the replacement secret. Verify the old credential is rejected when the provider supports a safe check and the new credential works only from the intended target.

### Provider-side retained data

Use the provider's authorized deletion/retention controls when available. Repository changes cannot prove provider-side deletion. Record provider receipt/confirmation without copying the deleted content back into Woof evidence.

### Status inflation

Correct documentation/product claims to distinguish repository-qualified, configured, deployed, and live-black-box-proven states. Do not fix inflated claims by lowering technical gates.

## Rollback

Rollback privacy/security code only through `deployment-readiness.md` and only if the previous image preserves or strengthens the relevant privacy boundary.

Never roll back:

- to a known-compromised secret;
- to a release that logs raw provider bodies/messages/stacks;
- to a release that exposes private object keys or Push subscription/content data;
- to Sentry configuration with weaker PII scrubbing;
- a credential revocation merely because rotation caused operational friction.

If an older application image requires a compromised credential to function, prefer a forward recovery.

## Verification

Use synthetic markers and require:

1. operational metrics contain no user ID, pet ID, provider external ID, request URL/query/body, or raw payload marker;
2. Sentry scrubber tests prove request/user/extra/breadcrumb/span data is removed before egress;
3. Health Lens and Behavior Vision synthetic private provider error bodies/messages do not appear in logs or surfaced exceptions;
4. Storage synthetic filenames/object keys/provider messages do not appear in telemetry and raw provider failures are normalized;
5. Push synthetic user/content/endpoint/key/provider-message markers do not appear in telemetry;
6. any rotated credential is absent from repository history introduced by the recovery change and is no longer accepted where provider verification is safely available;
7. external integration inventory still refuses production qualification without live evidence;
8. the triggering exposure path is absent for at least one representative synthetic execution after recovery.

## Evidence

Store only privacy-minimized incident metadata:

- exposure class;
- exact release SHA;
- affected component/provider name;
- first/last observed time;
- bounded event count;
- rotation/revocation receipt identifier when it is itself non-secret;
- recovery SHA;
- synthetic verification results.

Do not store the exposed secret/private value, raw request/provider body, user/pet identity, object key, push endpoint/key, chat text, caregiver note, or health/behavior content in the incident artifact.

## Rehearsal

Live rehearsal status: `NOT_YET_PROVEN`

Promotion to `PROVEN` requires an authorized production-like drill using synthetic canary markers that demonstrates containment, credential rotation or bounded feature disablement when applicable, privacy-safe evidence capture, and post-recovery verification without placing real private data into the incident workflow.