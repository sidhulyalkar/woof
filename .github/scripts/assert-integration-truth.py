#!/usr/bin/env python3
"""Fail closed when external integration authority or privacy boundaries drift."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "docs/EXTERNAL_INTEGRATION_INVENTORY.json"
HEALTH = ROOT / "apps/api/src/health-lens/health-ai.service.ts"
HEALTH_TEST = ROOT / "apps/api/src/health-lens/health-ai.service.spec.ts"
STORAGE = ROOT / "apps/api/src/storage/storage.service.ts"
STORAGE_TEST = ROOT / "apps/api/src/storage/storage.service.spec.ts"
PUSH = ROOT / "apps/api/src/notifications/notifications.service.ts"
PUSH_TEST = ROOT / "apps/api/src/notifications/notifications.service.spec.ts"
PUSH_STORE = ROOT / "apps/api/src/notifications/push-subscription.store.ts"
PUSH_STORE_TEST = ROOT / "apps/api/src/notifications/push-subscription.store.spec.ts"
PUSH_CONTROLLER = ROOT / "apps/api/src/notifications/notifications.controller.ts"
PUSH_CONTROLLER_TEST = ROOT / "apps/api/src/notifications/notifications.controller.spec.ts"
ENV = ROOT / "apps/api/src/config/env.validation.ts"
ENV_EXAMPLE = ROOT / "apps/api/.env.example"
N8N_README = ROOT / "n8n-workflows/README.md"
MEDIA_WORKER = ROOT / "apps/api/src/media-library/media-derivative.worker.ts"
MEDIA_TEST = ROOT / "apps/api/src/media-library/media-derivative.worker.spec.ts"
WORKFLOW = ROOT / ".github/workflows/integration-truth-ci.yml"

required_files = [
    INVENTORY,
    HEALTH,
    HEALTH_TEST,
    STORAGE,
    STORAGE_TEST,
    PUSH,
    PUSH_TEST,
    PUSH_STORE,
    PUSH_STORE_TEST,
    PUSH_CONTROLLER,
    PUSH_CONTROLLER_TEST,
    ENV,
    ENV_EXAMPLE,
    N8N_README,
    MEDIA_WORKER,
    MEDIA_TEST,
    WORKFLOW,
]
for path in required_files:
    if not path.is_file():
        raise SystemExit(f"required integration-truth source missing: {path.relative_to(ROOT)}")

inventory = json.loads(INVENTORY.read_text())
allowed_statuses = {"ACTIVE", "OPTIONAL_QUALIFIED", "RESERVED", "RETIRED"}
entries = {entry["id"]: entry for entry in inventory.get("integrations", [])}
expected = {
    "openai_health_lens": "OPTIONAL_QUALIFIED",
    "behavior_vision": "OPTIONAL_QUALIFIED",
    "s3_private_storage": "OPTIONAL_QUALIFIED",
    "media_derivatives": "OPTIONAL_QUALIFIED",
    "web_push": "OPTIONAL_QUALIFIED",
    "n8n": "RESERVED",
}
if set(expected) - set(entries):
    raise SystemExit(f"integration inventory is incomplete: {sorted(set(expected) - set(entries))}")
for integration_id, expected_status in expected.items():
    entry = entries[integration_id]
    if entry.get("status") not in allowed_statuses:
        raise SystemExit(f"invalid status for {integration_id}: {entry.get('status')!r}")
    if entry.get("status") != expected_status:
        raise SystemExit(
            f"integration authority drift for {integration_id}: expected {expected_status}, got {entry.get('status')}"
        )
    if entry.get("productionQualified") is not False:
        raise SystemExit(
            f"{integration_id} must not claim production qualification without deployment evidence"
        )

health = HEALTH.read_text()
for required in [
    "type HealthProviderFailureReason =",
    "'provider_http_error'",
    "'invalid_json'",
    "'timeout'",
    "'transport_error'",
    "Authorization: `Bearer ${this.apiKey}`",
    "store: false",
    "Health model provider failure reason=${reason}",
]:
    if required not in health:
        raise SystemExit(f"Health Lens privacy/authority marker missing: {required}")
for forbidden in [
    "response.text()",
    "await response.text",
    "error.message",
    "error.stack",
    "JSON.stringify(error)",
    "logger.warn(error",
    "logger.error(error",
]:
    if forbidden in health:
        raise SystemExit(f"Health Lens contains forbidden private diagnostic marker: {forbidden}")

health_test = HEALTH_TEST.read_text()
for required in [
    "never reads private provider error bodies into logs or the API error boundary",
    "classifies malformed provider JSON without logging response content",
    "classifies AbortError as timeout without logging the exception message",
    "classifies transport failures without logging arbitrary exception details",
    "fails closed before network access when the provider is unconfigured",
]:
    if required not in health_test:
        raise SystemExit(f"Health Lens defining privacy test missing: {required}")

storage = STORAGE.read_text()
for required in [
    "type StorageOperation =",
    "private async providerCall<T>",
    "Object storage provider failure operation=${operation}",
    "Media storage operation is temporarily unavailable",
]:
    if required not in storage:
        raise SystemExit(f"Storage provider boundary marker missing: {required}")
for forbidden in [
    "error.message",
    "error.stack",
    "Private file uploaded successfully:",
    "Private streamed file uploaded successfully:",
    "File deleted successfully:",
]:
    if forbidden in storage:
        raise SystemExit(f"Storage contains forbidden private diagnostic marker: {forbidden}")

storage_test = STORAGE_TEST.read_text()
for required in [
    "keeps generated private keys and filenames out of successful telemetry",
    "normalizes upload SDK failures without logging or rethrowing provider details",
    "normalizes delete SDK failures without logging the private object key",
    "fails closed before provider access when private storage is unconfigured",
]:
    if required not in storage_test:
        raise SystemExit(f"Storage defining privacy test missing: {required}")

push = PUSH.read_text()
for forbidden in [
    "PrismaService",
    "integrationToken",
    "pushError.message",
    "pushError.stack",
    "candidate.message",
    "candidate.stack",
]:
    if forbidden in push:
        raise SystemExit(f"Web Push contains forbidden storage/provider diagnostic marker: {forbidden}")
for line in push.splitlines():
    if "this.logger." in line and any(
        marker in line
        for marker in [
            "${userId}",
            "${title}",
            "${body}",
            "endpoint",
            "p256dh",
            "auth",
            "subscriptionFingerprint",
        ]
    ):
        raise SystemExit(f"Web Push logger contains a private identifier/content marker: {line.strip()}")
for required in [
    "PushSubscriptionStore",
    "pushSubscriptionFingerprint(stored.subscription)",
    "this.subscriptions.removeInvalidCurrent(userId)",
    "stored.state === 'LEGACY_MIGRATION_REQUIRED'",
    "reason: 'legacy_migration_required'",
    "Push notification delivered",
    "Push delivery failed${statusSuffix}",
    "Expired push subscription cleanup status=${pushError.statusCode}",
    "push_encryption_not_configured",
]:
    if required not in push:
        raise SystemExit(f"Web Push bounded/encrypted authority marker missing: {required}")

push_store = PUSH_STORE.read_text()
for required in [
    "dogos-push-subscription-v1",
    "ConnectorCryptoService",
    "PUSH_LEGACY_PLAINTEXT_READS_UNTIL",
    "state: 'LEGACY_MIGRATION_REQUIRED'",
    "private legacyPlaintextReadsEnabled()",
    "this.crypto.encrypt(",
    "this.crypto.decrypt(",
    "function canonicalSubscription(subscription: PushSubscriptionMaterial)",
    "pushSubscriptionFingerprint(subscription: PushSubscriptionMaterial)",
    "async removeIfFingerprint(userId: string, expectedFingerprint: string)",
    "async removeInvalidCurrent(userId: string)",
    "private async deleteExactRow(userId: string, row: StoredRow)",
    "data: { equals: expectedData as Prisma.InputJsonValue }",
    "async migrateLegacyRows(",
]:
    if required not in push_store:
        raise SystemExit(f"Web Push encrypted-store marker missing: {required}")
if "Logger" in push_store or "console." in push_store:
    raise SystemExit("Web Push encrypted credential store must remain free of credential telemetry")

push_test = PUSH_TEST.read_text()
for required in [
    "stores subscriptions through the encrypted store and exposes no credential row",
    "reports subscription status with a full-material fingerprint, not private Push material",
    "reports legacy migration required as unsubscribed without deleting plaintext state",
    "fails delivery closed when legacy plaintext compatibility has ended",
    "reduces arbitrary provider failures to status-only telemetry",
    "removes only the exact expired subscription on provider status %s without identifier leakage",
    "does not erase a replacement when provider-expiry cleanup loses the conditional race",
    "removes invalid encrypted rows only through exact invalid-row cleanup",
    "returns a truthful disabled state before storage/provider access when VAPID is unconfigured",
    "fails closed before storage/provider access when subscription encryption is unconfigured",
]:
    if required not in push_test:
        raise SystemExit(f"Web Push defining privacy test missing: {required}")

push_store_test = PUSH_STORE_TEST.read_text()
for required in [
    "writes only an authenticated encryption envelope for new subscriptions",
    "fingerprints the complete subscription so rotated keys at one endpoint are distinct",
    "rejects an encrypted subscription copied into the wrong user context",
    "rejects tampered ciphertext and never falls back to plaintext interpretation",
    "lazily migrates a valid legacy plaintext row only inside the configured compatibility window",
    "fails closed after the legacy plaintext compatibility window without deleting the row",
    "migrates legacy rows explicitly even after runtime compatibility is closed",
    "does not delete rotated keys at the same endpoint when the fingerprint is stale",
    "removes only the exact invalid row snapshot it inspected",
    "does not remove a concurrently replaced row after invalid-state verification",
    "does not resurrect stale legacy material when a concurrent subscription update wins",
]:
    if required not in push_store_test:
        raise SystemExit(f"Web Push encrypted-store defining test missing: {required}")

push_controller = PUSH_CONTROLLER.read_text()
if "req.user.sub" not in push_controller:
    raise SystemExit("Web Push subscription ownership must derive from authenticated session")
for forbidden in ["subscribeDto.userId", "@Post('send')", "sendPush(", "SendPushDto"]:
    if forbidden in push_controller:
        raise SystemExit(f"Web Push controller regained client-selected recipient authority: {forbidden}")

push_controller_test = PUSH_CONTROLLER_TEST.read_text()
for required in [
    "derives subscription ownership from the authenticated session",
    "binds current-browser conditional revocation to the authenticated session",
    "does not expose the retired arbitrary-target send method",
]:
    if required not in push_controller_test:
        raise SystemExit(f"Web Push controller defining test missing: {required}")

web_push = entries["web_push"]
for configuration in [
    "VAPID_PUBLIC_KEY",
    "VAPID_PRIVATE_KEY",
    "CONNECTOR_CREDENTIALS_KEY",
    "PUSH_LEGACY_PLAINTEXT_READS_UNTIL",
]:
    if configuration not in web_push.get("configuration", []):
        raise SystemExit(f"Web Push integration inventory missing configuration: {configuration}")
for evidence in [
    "authenticated encrypted subscription storage",
    "session-owned subscribe/unsubscribe authority",
    "bounded opt-in legacy plaintext runtime compatibility",
    "explicit migration remains available after runtime compatibility sunset",
    "compare-and-swap legacy plaintext migration",
    "full-material subscription fingerprint reconciliation",
    "atomic fingerprint-bound current-browser revocation",
    "exact invalid-row cleanup under concurrent replacement",
    "exact provider-expiry cleanup under concurrent replacement",
    "public arbitrary-recipient sender retired",
]:
    if evidence not in web_push.get("repositoryEvidence", []):
        raise SystemExit(f"Web Push integration inventory evidence missing: {evidence}")

env = ENV.read_text()
env_example = ENV_EXAMPLE.read_text()
if "N8N_WEBHOOK_SECRET" in env or "N8N_WEBHOOK_SECRET" in env_example:
    raise SystemExit("n8n is RESERVED and must not expose a runtime secret/configuration knob")
for marker in [
    "VAPID_PUBLIC_KEY and VAPID_PRIVATE_KEY must be configured together",
    "CONNECTOR_CREDENTIALS_KEY is required and must decode to 32 bytes when Web Push is configured in production",
    "PUSH_LEGACY_PLAINTEXT_READS_UNTIL may be at most 30 days in the future in production",
]:
    if marker not in env:
        raise SystemExit(f"production Web Push environment contract is missing: {marker}")
if "PUSH_LEGACY_PLAINTEXT_READS_UNTIL=" not in env_example:
    raise SystemExit("Web Push legacy compatibility cutoff is missing from operator configuration")

runtime_markers = ["N8N_WEBHOOK_SECRET", "webhooks/n8n", "localhost:5678"]
for path in (ROOT / "apps/api/src").rglob("*.ts"):
    text = path.read_text()
    for marker in runtime_markers:
        if marker in text:
            raise SystemExit(
                f"n8n is RESERVED but runtime marker {marker!r} exists in {path.relative_to(ROOT)}"
            )

n8n = N8N_README.read_text()
n8n_lower = n8n.lower()
for required in [
    "reference prototypes only",
    "no maintained `n8nwebhookscontroller`",
    "no default n8n credentials",
    "replay safety and idempotency",
    "deployment evidence",
]:
    if required not in n8n_lower:
        raise SystemExit(f"n8n reserved-authority documentation marker missing: {required}")
for forbidden in ["woofadmin", "Username: `admin`", "N8N_WEBHOOK_SECRET=", "docker logs woof-n8n"]:
    if forbidden in n8n:
        raise SystemExit(f"n8n prototype documentation still looks deployment-authoritative: {forbidden}")

workflow = WORKFLOW.read_text()
for required in [
    ".github/scripts/assert-integration-truth.py",
    "docs/EXTERNAL_INTEGRATION_INVENTORY.json",
    "apps/api/src/storage/storage.service.spec.ts",
    "apps/api/src/notifications/notifications.service.spec.ts",
    "apps/api/src/notifications/push-subscription.store.spec.ts",
    "apps/api/src/notifications/notifications.controller.spec.ts",
    "apps/api/src/media-library/media-derivative.worker.spec.ts",
    "python .github/scripts/assert-integration-truth.py",
]:
    if required not in workflow:
        raise SystemExit(f"integration-truth CI ownership marker missing: {required}")

print(
    "External integration truth is explicit: optional runtimes are repository-qualified, Web Push credentials are "
    "encrypted with bounded legacy compatibility and race-safe cleanup, n8n remains reserved, and production "
    "claims require live evidence."
)
