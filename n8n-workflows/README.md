# n8n workflow prototypes

The files in this directory are **reference prototypes only**. Woof does not currently have a qualified n8n runtime integration.

## Authority boundary

Current repository truth:

- there is no maintained `N8nWebhooksController` in `apps/api/src`;
- there is no maintained n8n webhook authentication guard in `apps/api/src`;
- Woof does not configure an `N8N_WEBHOOK_SECRET` runtime setting;
- these workflow JSON files are not evidence that any n8n instance is deployed, connected, authenticated, or allowed to mutate Woof data;
- no product surface should describe n8n automation as available today.

The canonical machine-readable classification is `RESERVED` in `docs/EXTERNAL_INTEGRATION_INVENTORY.json`.

## What is preserved here

The prototype workflows capture product ideas that may be useful later, including service follow-ups, meetup feedback reminders, event reminders, and fitness-goal acknowledgements. They are retained as design references, not executable release authority.

Do not copy example endpoints, identifiers, payloads, or credentials from old documentation into a production deployment. There are intentionally no default n8n credentials in this repository.

## Requirements before promotion

n8n may move from `RESERVED` to `OPTIONAL_QUALIFIED` only after a dedicated release implements and tests all of the following:

1. **Explicit runtime ownership.** Maintained API services/controllers own every inbound and outbound automation path.
2. **Authentication.** Every webhook or API request is authenticated with a dedicated secret or stronger credential mechanism. Secrets must never be transported in query strings or logged.
3. **Replay safety and idempotency.** Repeated delivery cannot duplicate rewards, notifications, bookings, or other mutations.
4. **Minimal event schemas.** Automation payloads carry opaque IDs and the minimum data required for the workflow. Raw private media, health text, behavior observations, push credentials, access tokens, and unnecessary profile data stay out of n8n.
5. **Bounded failure behavior.** Network timeouts, retries, delivery failures, and provider errors have explicit limits and content-free telemetry.
6. **Authorization.** Internal endpoints called by automation are not made broadly public merely to support n8n.
7. **Deterministic tests.** Provider stubs prove authentication, replay rejection/idempotency, malformed payload rejection, privacy boundaries, and degraded behavior.
8. **Durable CI ownership.** A `main` pull-request workflow owns the runtime and workflow definitions.
9. **Deployment evidence.** A real target n8n deployment, target Woof API revision, live authentication, networking, and black-box workflow execution are independently verified before anyone calls the integration production-qualified.

## Prototype review

When changing a workflow JSON file while n8n remains reserved:

- treat it as a design artifact;
- do not add real credentials or copied production payloads;
- avoid embedding personal data in fixtures;
- keep mutations visibly non-authoritative until the runtime requirements above exist;
- update the integration inventory if the intended authority changes.

This boundary is deliberate: preserving useful workflow ideas is cheap, while pretending a prototype is a live integration is expensive later. 🧩
