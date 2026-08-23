# dogOS Connectors

Connectors is the external-system boundary for dogOS. It is deliberately split from the canonical dog, activity, care, media, Story, and Autopilot domains so vendor APIs cannot become hidden sources of truth.

## Phase C v1 rule

**A provider transport may authenticate, fetch, normalize, deduplicate, and prove provenance. It may not directly rewrite canonical dogOS records.**

The intended pipeline is:

`provider transport → verified connector identity → normalized import envelope → domain importer → canonical source or immutable observation`

For wearables, the domain importer already exists: Fi/Tractive summaries pass through Autopilot's adapter and immutable zero-reward CareEvent path.

## Three persistence zones

Phase C intentionally separates three different kinds of state:

1. **Canonical dogOS truth** remains in the existing `public` schema. `Pet`, `Activity`, `CareEvent`, `MediaAsset`, Story, Adventure, and household records retain their existing owners and mutation rules.
2. **Connector operational metadata** lives in the isolated PostgreSQL schema `dogos_connectors`. It stores only queryable transport metadata: connection lifecycle, provider-to-pet identity mapping, cursors, hash-only import receipts, and revocation receipts.
3. **Provider secrets** remain in `IntegrationToken`, but only after authenticated encryption. No OAuth access/refresh token is stored in the operational schema.

This prevents provider bookkeeping from becoming a parallel dog database while keeping sync state relational and auditable.

## Provider registry

Phase C begins with these provider classes:

| Provider | Domain | v1 availability | Permitted capability |
| --- | --- | --- | --- |
| Fi | Wearable | `PARTNER_REQUIRED` | Daily activity, device status |
| Tractive | Wearable | `PARTNER_REQUIRED` | Daily activity, device status |
| Veterinary partner | Vet | `PARTNER_REQUIRED` | Appointment/vaccination import, medication/document references |
| Chewy | Retail | `PARTNER_REQUIRED` | Catalog references, user-approved handoff |
| Petco | Retail | `PARTNER_REQUIRED` | Catalog references, user-approved handoff |

`PARTNER_REQUIRED` is intentional. dogOS does not turn a consumer login page, reverse-engineered endpoint, or undocumented mobile API into a claimed OAuth integration.

A future provider can move out of this state only after its supported API contract, credentials, scopes, callback policy, revocation behavior, and signing requirements are verified.

## Credential storage

The pre-existing `IntegrationToken.data` column is ordinary JSON. Existing push-subscription usage demonstrates that the field itself does not provide encryption.

Phase C therefore wraps connector credential payloads in an authenticated encryption envelope before persistence:

- AES-256-GCM
- fresh 96-bit IV for every write
- authenticated tag
- user + provider binding through additional authenticated data (AAD)
- versioned envelope (`v=1`, `alg=A256GCM`)
- no plaintext access or refresh token in the database JSON

`CONNECTOR_CREDENTIALS_KEY` must decode to exactly 32 bytes. Production startup fails when Connectors is enabled without a valid key.

A connection is not considered healthy merely because a credential row exists. `CONNECTED` requires both:

- an operational connection in `dogos_connectors.connections` with `status=CONNECTED`
- a decryptable, unexpired authenticated credential envelope

Expired, malformed, or authentication-failing credentials degrade the operational connection to `REAUTH_REQUIRED`.

## Operational relational schema

The additive migration creates five tables under `dogos_connectors`:

- `connections`
  - user/provider lifecycle
  - external account ID/display label
  - granted scopes
  - connected, sync, and revoked timestamps
- `pet_identities`
  - provider animal ID ↔ dogOS pet ID
  - unique per connection
  - a database trigger rejects any mapping where the pet is not owned by the connection user
- `sync_cursors`
  - per-connection, per-resource cursor and watermark
- `import_receipts`
  - provider external object ID
  - SHA-256 of the normalized canonical observation
  - imported/skipped/failed disposition
  - optional canonical reference ID/type
  - no raw provider payload
- `revocation_receipts`
  - local credential deletion vs future remote revocation
  - success/unavailable/failure evidence
  - no raw provider response body

The migration is additive. It does not alter canonical dogOS tables.

## Wearables

Fi and Tractive retain the Phase A normalization policy:

- daily activity and device-status summaries only
- unrecognized vendor fields are discarded
- location/GPS-shaped payloads are rejected by Autopilot
- imported wearable observations are private and `safetyEligible=false`
- wearable observations cannot earn Bond XP
- connector transport cannot mutate `Pet` or `Activity`

There is intentionally **no public browser wearable-ingestion endpoint**. A browser must not be able to impersonate Fi or Tractive by posting a provider-shaped JSON body.

The internal verified-transport seam requires all of the following before delegation to Autopilot:

1. operational connection status is `CONNECTED`
2. encrypted provider credential authenticates and is not expired
3. external provider pet ID maps to an owned dogOS pet
4. provider is a wearable provider
5. provider payload passes the existing Autopilot summary/location normalization rules

The connector then records a hash-only import receipt referencing the resulting immutable CareEvent.

### Lost-response repair

Autopilot/CareEvent remains the exactly-once source boundary. The connector receipt is downstream evidence, not a second dedupe authority.

If a process dies after CareEvent persistence but before the connector receipt is written, replay re-enters Autopilot, receives the existing canonical observation, and repairs the missing receipt from that persisted observation. If the retry payload differs from the canonical observation, the canonical receipt is repaired first and the altered retry is rejected.

## Veterinary records

Veterinary transport is provider-specific. Phase C does not pretend generic human FHIR support is equivalent to a veterinary EHR integration.

The registry defines the allowed v1 intent:

- appointments may be imported as sourced records/observations
- vaccination data may be imported with provenance
- medication information is a reference to veterinarian-provided instructions
- documents remain source references
- no connector computes dosage, prescribes treatment, or directly rewrites canonical pet health fields

A real vet transport must use the same connection, identity, cursor, and import-receipt boundary before it can be enabled.

## Retail

Retail is approval-first:

- catalog/product references may inform suggestions
- handoff may occur only after the user chooses to proceed
- no autonomous cart additions
- no autonomous order placement
- no autonomous payment or charge

Chewy and Petco remain partner-gated until a supported integration contract is available.

## Location

Precise tracker location import is disabled in this slice. Provider definitions explicitly report `preciseLocationEnabled=false`.

Before location can be enabled, Connectors must add an explicit scope, retention duration, deletion/revocation behavior, and user-visible permission surface. Phase A's location-rejection behavior remains the default.

## Current public API

All routes are JWT-protected and feature-gated by `ENABLE_DOGOS_CONNECTORS`.

- `GET /connectors`
  - provider capabilities
  - truthful operational + credential connection state
  - credential-vault readiness
  - global safety boundaries
- `POST /connectors/:provider/oauth/start`
  - currently returns an explicit `partner_required` conflict for registry providers
  - never invents an undocumented authorization URL
- `DELETE /connectors/:provider`
  - deletes local encrypted credentials
  - records local revocation evidence when an operational connection exists
  - does not falsely claim remote revocation when no official revocation transport is configured

Provider ingestion, connection registration, identity binding, and cursor advancement are internal transport seams, not browser APIs.

## CI invariants

The dedicated Connectors lane must prove:

- credential plaintext is absent from persisted envelopes
- ciphertext is bound to user/provider AAD
- missing/malformed encryption key fails closed
- expired/corrupt credentials cannot remain `CONNECTED`
- unsupported/partner-gated providers cannot manufacture OAuth state
- browser provider impersonation is not exposed
- cross-user provider-pet mapping is rejected by the database
- sync cursors persist independently from canonical records
- import receipt uniqueness prevents provenance duplication
- lost receipt repair derives from immutable Autopilot/CareEvent truth
- altered external-object replays are rejected
- retail definitions prohibit autonomous purchase
- precise location remains disabled
- operational tables contain no raw provider payload column
- connector source contains no direct canonical `Pet`, `Activity`, `CareEvent`, or `MediaAsset` mutation calls
- strict lint/type-check and API build remain green
- inherited Autopilot, Our Story, Foundation, Adventure, and root CI remain green on the same exact head
