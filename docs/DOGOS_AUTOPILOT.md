# dogOS Autopilot v1

## Purpose

Autopilot is the first proactive dogOS layer. It turns a small amount of trusted context into reminders and check-ins without turning Woof into a medical monitor or giving external vendors authority over canonical dog state.

The pipeline is deliberately one-way:

`provider payload -> adapter -> normalized summary -> immutable CareEvent -> dedupe -> bounded signal/reminder policy -> user suggestion`

External providers never receive a code path that can directly mutate `Pet`, Activity, Health Lens, Adventure rewards, or household membership.

## Phase A provider boundary

The first adapters are Fi and Tractive stubs. They accept only:

- `DAILY_ACTIVITY`
  - activity minutes
  - distance meters
  - steps
- `DEVICE_STATUS`
  - battery percent
  - online/offline/unknown state

Autopilot v1 intentionally rejects location-shaped payloads, including nested latitude/longitude, coordinates, GPS traces, routes, and tracks. Continuous location ingestion belongs in the Connectors phase, where retention, revocation, export, and location-specific permissions can be explicit.

Provider-specific fields that are not recognized by an adapter are discarded rather than copied into storage.

## Observation integrity

Normalized tracker summaries reuse the Adventure `CareEvent` ledger because it already provides:

- immutable event identity
- per-user dedupe keys
- pet ownership checks
- occurrence-time normalization
- private visibility
- provenance fields

Wearable observations use `evidenceType=WEARABLE`, provider-specific `source`, and `safetyEligible=false`.

That last invariant is critical: tracker data is context, not proof of virtuous care. It cannot award Bond XP. The normal Adventure reward ledger still records the zero-reward decision so the policy remains auditable.

## Baseline-change signals

Autopilot signals are suggestions stored as user notifications, not diagnoses or pet-health fields.

### Lower activity check-in

A lower-activity signal requires all of the following:

1. a current daily tracker summary with activity minutes;
2. at least six prior summaries in the preceding 28 days;
3. a median recent baseline;
4. current activity at or below 55% of that baseline; and
5. an absolute drop of at least 20 minutes.

The message explicitly names benign context such as rest days, weather, routine changes, and tracker wear. It does not infer illness, pain, fatigue, or another medical state.

### Tracker battery

Battery at or below 15% can create an informational device-maintenance signal. This is operational, not a wellbeing inference.

Replayed provider events are deduped by the immutable CareEvent key and cannot create a second signal.

## Care reminders

Care reminders use the existing durable `Notification` store as scheduled notification envelopes. This keeps the first proactive layer additive without introducing another parallel notification database abstraction.

Supported reminder kinds:

- vet appointment
- medication
- grooming
- general care

A reminder may be one-time or repeat every 1-365 days. Household members who can access a pet may schedule its reminders.

The scheduler runs every ten minutes. Before attempting delivery it claims a reminder with a PostgreSQL advisory transaction lock and records `lastAttemptAt`. This prevents multiple API replicas from immediately delivering the same reminder. Failed delivery leaves the reminder scheduled and eligible for a bounded retry after six hours. A one-time reminder is completed only after successful push delivery. Recurring reminders advance to the next future due time only after successful delivery.

Medication reminder copy tells the user to follow veterinary instructions. The scheduler does not calculate or change medication dosage.

## API

All endpoints require JWT authentication and `ENABLE_DOGOS_AUTOPILOT`.

- `GET /autopilot`
  - active reminders
  - unacknowledged signals
  - provider capabilities
  - explicit safety/privacy boundaries
- `POST /autopilot/observations/:provider`
  - authenticated owner-side Fi/Tractive stub ingestion
- `POST /autopilot/reminders`
  - schedule a care reminder
- `DELETE /autopilot/reminders/:id`
  - cancel a reminder owned by the authenticated user
- `POST /autopilot/signals/:id/acknowledge`
  - acknowledge a signal owned by the authenticated user

Provider webhook authentication and OAuth/token lifecycle are intentionally deferred to the Connectors phase. Phase A does not expose an unauthenticated vendor webhook.

## Rollout

`ENABLE_DOGOS_AUTOPILOT` follows the Adventure rollout convention:

- development/test: enabled unless explicitly false
- production: disabled unless explicitly true

A disabled experimental surface returns 404 rather than exposing rollout configuration.

## Executable invariants

`.github/workflows/dogos-autopilot-ci.yml` verifies the release surface and also rejects direct Prisma `Pet` create/update/delete/upsert calls anywhere inside `src/autopilot`.

The focused contracts cover:

- Fi normalization and field minimization
- Tractive normalization
- nested location rejection
- unsupported provider rejection
- invalid battery rejection
- wearable CareEvents are private and zero-reward
- provider replay does not duplicate signals
- sparse baselines do not alert
- large, well-supported activity drops create non-diagnostic check-ins
- low battery creates only an informational signal
- household-authorized reminder creation
- failed push delivery does not complete a reminder

## Next phases

Autopilot v1 deliberately stops before provider OAuth/webhooks, continuous location retention, veterinary-record imports, retail actions, or autonomous purchasing. Those belong to the Connectors and Concierge phases where consent, provenance, revocation, and human approval can be designed as first-class contracts.
