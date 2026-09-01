#!/usr/bin/env python3
"""Fail closed when Web Push credential, migration, or browser authority drifts."""

from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "apps/api/src/notifications/push-subscription.store.ts"
STORE_TEST = ROOT / "apps/api/src/notifications/push-subscription.store.spec.ts"
SERVICE = ROOT / "apps/api/src/notifications/notifications.service.ts"
SERVICE_TEST = ROOT / "apps/api/src/notifications/notifications.service.spec.ts"
CONTROLLER = ROOT / "apps/api/src/notifications/notifications.controller.ts"
CONTROLLER_TEST = ROOT / "apps/api/src/notifications/notifications.controller.spec.ts"
DTO = ROOT / "apps/api/src/notifications/dto/push-subscription.dto.ts"
MODULE = ROOT / "apps/api/src/notifications/notifications.module.ts"
MIGRATION = ROOT / "apps/api/scripts/migrate-push-subscriptions.ts"
PACKAGE = ROOT / "apps/api/package.json"
ENV = ROOT / "apps/api/src/config/env.validation.ts"
ENV_TEST = ROOT / "apps/api/src/config/env.validation.spec.ts"
ENV_EXAMPLE = ROOT / "apps/api/.env.example"
WEB_API = ROOT / "apps/web/src/lib/api.ts"
WEB_HOOK = ROOT / "apps/web/src/hooks/use-push-notifications.ts"
MOBILE_NOTIFICATIONS = ROOT / "apps/mobile/src/api/notifications.ts"
INVENTORY = ROOT / "docs/EXTERNAL_INTEGRATION_INVENTORY.json"
DOC = ROOT / "docs/DOGOS_WEB_PUSH_ENCRYPTION.md"
WORKFLOW = ROOT / ".github/workflows/dogos-push-encryption-ci.yml"
INTEGRATION_WORKFLOW = ROOT / ".github/workflows/integration-truth-ci.yml"
INTEGRATION_GUARD = ROOT / ".github/scripts/assert-integration-truth.py"

required_files = [
    STORE,
    STORE_TEST,
    SERVICE,
    SERVICE_TEST,
    CONTROLLER,
    CONTROLLER_TEST,
    DTO,
    MODULE,
    MIGRATION,
    PACKAGE,
    ENV,
    ENV_TEST,
    ENV_EXAMPLE,
    WEB_API,
    WEB_HOOK,
    INVENTORY,
    DOC,
    WORKFLOW,
    INTEGRATION_WORKFLOW,
    INTEGRATION_GUARD,
]
for path in required_files:
    if not path.is_file():
        raise SystemExit(f"required Web Push authority source missing: {path.relative_to(ROOT)}")
if MOBILE_NOTIFICATIONS.exists():
    raise SystemExit("phantom Mobile notifications/push-token API must remain retired")


def require(text: str, markers: list[str], label: str) -> None:
    for marker in markers:
        if marker not in text:
            raise SystemExit(f"{label} marker missing: {marker}")


store = STORE.read_text()
require(
    store,
    [
        "const PUSH_PROVIDER = 'push_subscription'",
        "const PUSH_CONTEXT_VERSION = 'dogos-push-subscription-v1'",
        "private readonly config: ConfigService",
        "this.crypto.encrypt(",
        "this.crypto.decrypt(",
        "function canonicalSubscription(subscription: PushSubscriptionMaterial)",
        "pushSubscriptionFingerprint(subscription: PushSubscriptionMaterial)",
        "state: 'LEGACY_MIGRATION_REQUIRED'",
        "PUSH_LEGACY_PLAINTEXT_READS_UNTIL",
        "private legacyPlaintextReadsEnabled()",
        "async removeIfFingerprint(userId: string, expectedFingerprint: string)",
        "async removeInvalidCurrent(userId: string)",
        "private async deleteExactRow(userId: string, row: StoredRow)",
        "data: { equals: row.data as Prisma.InputJsonValue }",
        "...(lastSeenId ? { id: { gt: lastSeenId } } : {})",
        "data: { equals: expectedData as Prisma.InputJsonValue }",
        "async migrateLegacyRows(",
        "return this.readCurrentRow(userId, false)",
    ],
    "Push encrypted store",
)
for forbidden in [
    "Logger",
    "console.",
    "error.message",
    "error.stack",
    "process.stdout",
    "process.stderr",
]:
    if forbidden in store:
        raise SystemExit(f"Push encrypted store contains forbidden diagnostic/output authority: {forbidden}")

put_match = re.search(r"async put\([\s\S]*?\n  async get\(", store)
if not put_match:
    raise SystemExit("Push subscription put() boundary could not be located")
if "this.crypto.encrypt(" not in put_match.group(0) or "integrationToken.upsert" not in put_match.group(0):
    raise SystemExit("Push subscription writes must encrypt before IntegrationToken upsert")
for plaintext_write in ["data: toPlainJson(", "data: subscription", "data: credentials"]:
    if plaintext_write in put_match.group(0):
        raise SystemExit(f"Push subscription put() contains plaintext persistence: {plaintext_write}")

store_test = STORE_TEST.read_text()
require(
    store_test,
    [
        "writes only an authenticated encryption envelope for new subscriptions",
        "fingerprints the complete subscription so rotated keys at one endpoint are distinct",
        "rejects an encrypted subscription copied into the wrong user context",
        "rejects tampered ciphertext and never falls back to plaintext interpretation",
        "rejects partial envelope-shaped data instead of downgrading to legacy plaintext",
        "lazily migrates a valid legacy plaintext row only inside the configured compatibility window",
        "fails closed after the legacy plaintext compatibility window without deleting the row",
        "treats an absent legacy plaintext cutoff as compatibility disabled",
        "migrates legacy rows explicitly even after runtime compatibility is closed",
        "does not resurrect stale legacy material when a concurrent subscription update wins",
        "conditionally deletes only the exact encrypted row matching the browser fingerprint",
        "does not delete rotated keys at the same endpoint when the fingerprint is stale",
        "does not delete a concurrently replaced row after fingerprint verification",
        "removes only the exact invalid row snapshot it inspected",
        "does not remove a concurrently replaced row after invalid-state verification",
        "advances by id range so a deleted previous page tail cannot invalidate the next scan",
        "counts compare-and-swap losses instead of overwriting concurrent changes",
    ],
    "Push encrypted-store defining test",
)

service = SERVICE.read_text()
require(
    service,
    [
        "type PushNotificationInput = {",
        "pushSubscriptionFingerprint(stored.subscription)",
        "async removeCurrentPushSubscription(userId: string, subscriptionFingerprint: string)",
        "this.subscriptions.removeIfFingerprint(userId, subscriptionFingerprint)",
        "this.subscriptions.removeInvalidCurrent(userId)",
        "stored.state === 'LEGACY_MIGRATION_REQUIRED'",
        "reason: 'legacy_migration_required'",
        "Legacy push subscription requires operator migration",
        "Expired push subscription cleanup status=${pushError.statusCode}",
        "push_encryption_not_configured",
    ],
    "Push service authority",
)
for forbidden in [
    "SendPushDto",
    "PrismaService",
    "integrationToken",
    "readStoredSubscription",
    "toStoredSubscription",
    "pushError.message",
    "pushError.stack",
    "candidate.message",
    "candidate.stack",
]:
    if forbidden in service:
        raise SystemExit(f"NotificationsService bypasses encrypted/private boundary: {forbidden}")
for line in service.splitlines():
    if "this.logger." in line and any(
        private in line
        for private in [
            "${userId}",
            "${title}",
            "${body}",
            "endpoint",
            "p256dh",
            "auth",
            "subscriptionFingerprint",
        ]
    ):
        raise SystemExit(f"Push logger exposes private/correlatable material: {line.strip()}")

service_test = SERVICE_TEST.read_text()
require(
    service_test,
    [
        "reports subscription status with a full-material fingerprint, not private Push material",
        "reports legacy migration required as unsubscribed without deleting plaintext state",
        "fails delivery closed when legacy plaintext compatibility has ended",
        "uses fingerprint-bound removal for current-browser revocation",
        "treats a fingerprint mismatch as a safe no-op instead of account-wide deletion",
        "removes only the exact expired subscription on provider status %s without identifier leakage",
        "does not erase a replacement when provider-expiry cleanup loses the conditional race",
        "removes invalid encrypted rows only through exact invalid-row cleanup",
        "does not claim invalid-row removal when a concurrent replacement wins",
        "keeps account-wide unsubscribe available when VAPID and encryption are unavailable",
    ],
    "Push service defining test",
)

controller = CONTROLLER.read_text()
require(
    controller,
    [
        "@Get('subscription')",
        "@Post('subscribe')",
        "@Post('subscription/revoke')",
        "@Delete('unsubscribe')",
        "req.user.sub",
        "current.subscriptionFingerprint",
    ],
    "Push controller session authority",
)
for forbidden in [
    "subscribeDto.userId",
    "@Post('send')",
    "sendPush(",
    "SendPushDto",
    "@Param(",
    "unsubscribe/:endpoint",
    "@Delete('subscription')",
]:
    if forbidden in controller:
        raise SystemExit(f"Push controller regained client-selected/fragile authority: {forbidden}")

controller_test = CONTROLLER_TEST.read_text()
require(
    controller_test,
    [
        "derives subscription status ownership from the authenticated session",
        "derives subscription ownership from the authenticated session",
        "binds current-browser conditional revocation to the authenticated session",
        "does not expose the retired arbitrary-target send method",
    ],
    "Push controller defining authority test",
)

dto = DTO.read_text()
require(
    dto,
    [
        "export class SubscribeDto",
        "export class CurrentPushSubscriptionDto",
        "@Matches(/^[A-Za-z0-9_-]{43}$/)",
    ],
    "Push DTO",
)
if "SendPushDto" in dto:
    raise SystemExit("public arbitrary-recipient Push DTO must remain retired")
subscribe_match = re.search(r"export class SubscribeDto \{([\s\S]*?)\n\}", dto)
if not subscribe_match or "userId" in subscribe_match.group(1):
    raise SystemExit("SubscribeDto must not accept client-selected userId")

module = MODULE.read_text()
require(module, ["ConnectorCryptoService", "PushSubscriptionStore"], "NotificationsModule")

for path in (ROOT / "apps/api/src").rglob("*.ts"):
    if path in {STORE, STORE_TEST}:
        continue
    text = path.read_text()
    if "provider: 'push_subscription'" in text and "integrationToken" in text:
        raise SystemExit(
            f"direct push_subscription persistence bypasses encrypted store: {path.relative_to(ROOT)}"
        )
    if "export class SendPushDto" in text:
        raise SystemExit(f"public arbitrary-recipient Push DTO returned: {path.relative_to(ROOT)}")

migration = MIGRATION.read_text()
require(
    migration,
    [
        "new PushSubscriptionStore(prisma as unknown as PrismaService, crypto, config)",
        "migrateLegacyRows(batchSize)",
        "schemaVersion: 1",
        "migration: 'web_push_subscription_encryption_v1'",
        "Push subscription encryption migration failed",
        "await prisma.$disconnect()",
    ],
    "Push migration command",
)
for forbidden in [
    "console.log",
    "console.error",
    "error.message",
    "error.stack",
    "row.id",
    "userId",
    "endpoint",
    "p256dh",
    "auth",
    "ciphertext",
    "iv:",
    "tag:",
]:
    if forbidden in migration:
        raise SystemExit(f"Push migration command may expose identifier/credential detail: {forbidden}")

package = json.loads(PACKAGE.read_text())
if package.get("scripts", {}).get("migrate:push-subscriptions") != "ts-node scripts/migrate-push-subscriptions.ts":
    raise SystemExit("Push migration package script missing or drifted")
if package.get("devDependencies", {}).get("jest") != "^29.7.0":
    raise SystemExit("Push migration tranche must not remove the API Jest qualification dependency")

env = ENV.read_text()
require(
    env,
    [
        "PUSH_LEGACY_MAX_FUTURE_WINDOW_MS = 30 * 24 * 60 * 60 * 1000",
        "PUSH_LEGACY_PLAINTEXT_READS_UNTIL: z.string().optional()",
        "PUSH_LEGACY_PLAINTEXT_READS_UNTIL must be an ISO-8601 timestamp with an explicit timezone",
        "PUSH_LEGACY_PLAINTEXT_READS_UNTIL may be at most 30 days in the future in production",
        "VAPID_PUBLIC_KEY and VAPID_PRIVATE_KEY must be configured together",
        "CONNECTOR_CREDENTIALS_KEY is required and must decode to 32 bytes when Web Push is configured in production",
    ],
    "production Push environment authority",
)
require(
    ENV_TEST.read_text(),
    [
        "rejects malformed Web Push legacy plaintext compatibility cutoffs",
        "rejects Web Push legacy plaintext compatibility windows beyond 30 production days",
        "accepts a bounded Web Push legacy plaintext compatibility window in production",
        "accepts an already-expired Push legacy cutoff as an explicit disabled state",
        "requires encrypted credential storage when Web Push is configured in production",
    ],
    "production Push environment test",
)
require(
    ENV_EXAMPLE.read_text(),
    [
        "PUSH_LEGACY_PLAINTEXT_READS_UNTIL=",
        "Empty means runtime plaintext reads are disabled",
        "no more than 30 days ahead in production",
        "explicit migration command remains usable",
    ],
    "Push operator configuration",
)

web_api = WEB_API.read_text()
require(
    web_api,
    [
        "subscriptionFingerprint?: string;",
        "apiClient.get<PushSubscriptionStatus>('/notifications/subscription')",
        "unsubscribeCurrent: (subscriptionFingerprint: string)",
        "apiClient.post<PushSubscriptionResult>('/notifications/subscription/revoke'",
        "unsubscribeAccount: () => apiClient.delete<PushSubscriptionResult>('/notifications/unsubscribe')",
    ],
    "Web Push browser API",
)
for forbidden in [
    "/notifications/send",
    "sendPush:",
    "/notifications/unsubscribe/",
    "delete<PushSubscriptionResult>('/notifications/subscription'",
]:
    if forbidden in web_api:
        raise SystemExit(f"Web API retained phantom or fragile Push authority: {forbidden}")

web_hook = WEB_HOOK.read_text()
require(
    web_hook,
    [
        "Boolean(window.crypto?.subtle)",
        "function canonicalPushSubscription(subscription: PushSubscription)",
        "serialized.keys?.p256dh",
        "serialized.keys?.auth",
        "browserFingerprint === serverStatus.subscriptionFingerprint",
        "notificationsApi.unsubscribeCurrent(browserFingerprint)",
        "const fingerprint = await subscriptionFingerprint(subscription)",
    ],
    "browser Push reconciliation/revocation",
)
status_effect = re.search(r"useEffect\(\(\) => \{([\s\S]*?)\n  \}, \[\]\);", web_hook)
if not status_effect:
    raise SystemExit("browser Push status reconciliation effect could not be located")
if "unsubscribe" in status_effect.group(1):
    raise SystemExit("passive browser Push status reconciliation must never revoke server state")
if "notificationsApi.unsubscribeAccount" in web_hook:
    raise SystemExit("ordinary browser Push lifecycle must not use account-wide revocation")

inventory = json.loads(INVENTORY.read_text())
web_push = next(
    (entry for entry in inventory.get("integrations", []) if entry.get("id") == "web_push"),
    None,
)
if not web_push or web_push.get("productionQualified") is not False:
    raise SystemExit("Web Push inventory must remain present and not production-qualified")
for config_key in [
    "VAPID_PUBLIC_KEY",
    "VAPID_PRIVATE_KEY",
    "CONNECTOR_CREDENTIALS_KEY",
    "PUSH_LEGACY_PLAINTEXT_READS_UNTIL",
]:
    if config_key not in web_push.get("configuration", []):
        raise SystemExit(f"Web Push inventory configuration missing: {config_key}")
for evidence in [
    "authenticated encrypted subscription storage",
    "bounded opt-in legacy plaintext runtime compatibility",
    "explicit migration remains available after runtime compatibility sunset",
    "compare-and-swap legacy plaintext migration",
    "full-material subscription fingerprint reconciliation",
    "atomic fingerprint-bound current-browser revocation",
    "exact invalid-row cleanup under concurrent replacement",
    "exact provider-expiry cleanup under concurrent replacement",
    "decrypt-free account-wide recovery revocation",
    "public arbitrary-recipient sender retired",
]:
    if evidence not in web_push.get("repositoryEvidence", []):
        raise SystemExit(f"Web Push inventory repository evidence missing: {evidence}")

require(
    DOC.read_text(),
    [
        "dogos-push-subscription-v1:<userId>",
        "PUSH_LEGACY_PLAINTEXT_READS_UNTIL",
        "one-way **and time-bounded**",
        "LEGACY_MIGRATION_REQUIRED",
        "explicit migration command is deliberately independent of the runtime compatibility cutoff",
        "Provider 404/410 cleanup is bound to the full subscription fingerprint",
        "The pre-encryption application revision is **not data-compatible with encrypted Push rows**",
        "Replacing the environment key in place is **not** a valid rotation procedure",
        "does **not** prove that production rows were migrated",
    ],
    "Web Push encryption documentation",
)

workflow = WORKFLOW.read_text()
require(
    workflow,
    [
        ".github/scripts/assert-push-subscription-encryption.py",
        "apps/api/src/notifications/**",
        "apps/api/scripts/migrate-push-subscriptions.ts",
        "apps/api/src/config/env.validation.ts",
        "apps/api/src/config/env.validation.spec.ts",
        "apps/web/src/hooks/use-push-notifications.ts",
        "push-subscription.store.spec.ts",
        "notifications.controller.spec.ts",
        "python .github/scripts/assert-push-subscription-encryption.py",
    ],
    "Push encryption CI ownership",
)
require(
    INTEGRATION_WORKFLOW.read_text(),
    [
        "apps/api/src/notifications/**",
        "apps/api/src/notifications/push-subscription.store.spec.ts",
        "apps/api/src/notifications/notifications.controller.spec.ts",
        "apps/mobile/src/api/notifications.ts",
    ],
    "broad integration CI Push ownership",
)
require(
    INTEGRATION_GUARD.read_text(),
    ["PUSH_STORE", "authenticated encrypted subscription storage"],
    "broad integration truth encrypted Push",
)

print(
    "Web Push authority is explicit: credentials are user-bound encrypted data, legacy plaintext runtime "
    "compatibility has a bounded sunset, explicit migration remains available afterward, cleanup is race-safe, "
    "and live production migration remains unproven."
)
