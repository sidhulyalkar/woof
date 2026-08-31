#!/usr/bin/env python3
"""Fail closed when operational runbook authority drifts from maintained Woof source."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "ops/runbooks/runbook-registry.v1.json"
ALERT_POLICY = ROOT / "ops/alerts/woof-api-alert-policy.v1.json"
AUTH = ROOT / "apps/api/src/auth/auth.controller.ts"
SESSION_AUTHORITY = ROOT / "apps/api/src/auth/session-authority.service.ts"
CHAT = ROOT / "apps/api/src/chat/chat.gateway.ts"
REALTIME_ADMISSION = ROOT / "apps/api/src/chat/realtime-admission.service.ts"
CAREGIVER = ROOT / "apps/api/src/caregiver/caregiver.service.ts"
CONNECTOR_GUARD = ROOT / "apps/api/src/connectors/connectors-enabled.guard.ts"
CONNECTORS = ROOT / "apps/api/src/connectors/connectors.service.ts"
ENV = ROOT / "apps/api/src/config/env.validation.ts"
METRICS = ROOT / "apps/api/src/observability/operational-metrics.service.ts"
SENTRY = ROOT / "apps/api/src/sentry.ts"
INTEGRATIONS = ROOT / "docs/EXTERNAL_INTEGRATION_INVENTORY.json"
WORKFLOW = ROOT / ".github/workflows/operational-runbooks-ci.yml"

required_files = [
    REGISTRY,
    ALERT_POLICY,
    AUTH,
    SESSION_AUTHORITY,
    CHAT,
    REALTIME_ADMISSION,
    CAREGIVER,
    CONNECTOR_GUARD,
    CONNECTORS,
    ENV,
    METRICS,
    SENTRY,
    INTEGRATIONS,
    WORKFLOW,
]
for path in required_files:
    if not path.is_file():
        raise SystemExit(f"required runbook authority source missing: {path.relative_to(ROOT)}")

registry = json.loads(REGISTRY.read_text())
policy = json.loads(ALERT_POLICY.read_text())
integrations = json.loads(INTEGRATIONS.read_text())

if registry.get("version") != "woof-operational-runbooks-v1":
    raise SystemExit("unexpected operational runbook registry version")
if registry.get("alertPolicy") != "ops/alerts/woof-api-alert-policy.v1.json":
    raise SystemExit("runbook registry must point at the canonical v1 alert policy")
if registry.get("liveRehearsalStatus") != "NOT_YET_PROVEN":
    raise SystemExit("repository runbooks must not claim live rehearsal evidence")

authority = registry.get("authority") or {}
if authority.get("productionQualified") is not False:
    raise SystemExit("operational runbooks must not claim production qualification")
if authority.get("repositoryQualified") not in {False, True}:
    raise SystemExit("repositoryQualified must be an explicit boolean")
repository_qualification = (
    "CODE_QUALIFIED" if authority.get("repositoryQualified") is True else "PENDING"
)

runbooks = registry.get("runbooks")
if not isinstance(runbooks, list) or not runbooks:
    raise SystemExit("runbook registry must contain at least one runbook")

ids = [entry.get("id") for entry in runbooks]
if any(not isinstance(value, str) or not value for value in ids):
    raise SystemExit("every runbook must have a non-empty id")
if len(ids) != len(set(ids)):
    raise SystemExit("runbook ids must be unique")

required_ids = {
    "deployment-readiness",
    "database-migration",
    "auth-session",
    "api-degradation",
    "connector-ingestion",
    "privacy-telemetry",
    "realtime",
}
if set(ids) != required_ids:
    raise SystemExit(f"runbook registry ids drifted: expected {sorted(required_ids)}, got {sorted(ids)}")

required_sections = [
    "## Authority boundary",
    "## Detection",
    "## Impact",
    "## Containment",
    "## Diagnosis",
    "## Recovery",
    "## Rollback",
    "## Verification",
    "## Evidence",
    "## Rehearsal",
]

allowed_manual_signals = {"privacy_or_secret_exposure"}
registered_alerts: set[str] = set()
registered_manual: set[str] = set()
runbook_by_id = {}

for entry in runbooks:
    runbook_id = entry["id"]
    runbook_by_id[runbook_id] = entry
    owner = entry.get("owner")
    if not isinstance(owner, str) or not owner.strip():
        raise SystemExit(f"runbook {runbook_id} must declare a non-empty owner role")
    if entry.get("liveRehearsalStatus") != "NOT_YET_PROVEN":
        raise SystemExit(f"runbook {runbook_id} must remain NOT_YET_PROVEN until a real drill")

    relative = entry.get("path")
    if not isinstance(relative, str) or not relative.startswith("ops/runbooks/") or not relative.endswith(".md"):
        raise SystemExit(f"runbook {runbook_id} has invalid path {relative!r}")
    path = ROOT / relative
    if not path.is_file():
        raise SystemExit(f"registered runbook is missing: {relative}")

    text = path.read_text()
    if f"Runbook ID: `{runbook_id}`" not in text:
        raise SystemExit(f"runbook id marker missing from {relative}")
    if f"Repository qualification: `{repository_qualification}`" not in text:
        raise SystemExit(
            f"runbook {runbook_id} must match repository qualification {repository_qualification}"
        )
    if "Live rehearsal status: `NOT_YET_PROVEN`" not in text:
        raise SystemExit(f"live rehearsal marker missing from {relative}")
    for section in required_sections:
        if section not in text:
            raise SystemExit(f"runbook {runbook_id} is missing required section {section}")

    alerts = entry.get("alerts")
    manual = entry.get("manualSignals")
    if not isinstance(alerts, list) or not all(isinstance(value, str) and value for value in alerts):
        raise SystemExit(f"runbook {runbook_id} alerts must be a string list")
    if not isinstance(manual, list) or not all(isinstance(value, str) and value for value in manual):
        raise SystemExit(f"runbook {runbook_id} manualSignals must be a string list")
    if len(alerts) != len(set(alerts)) or len(manual) != len(set(manual)):
        raise SystemExit(f"runbook {runbook_id} contains duplicate alert/manual signal mappings")
    registered_alerts.update(alerts)
    registered_manual.update(manual)

policy_meta = {"version", "service", "evaluationInterval", "deferredSignals"}
policy_alerts = set(policy) - policy_meta
unknown_alerts = registered_alerts - policy_alerts
missing_alerts = policy_alerts - registered_alerts
if unknown_alerts:
    raise SystemExit(f"runbooks reference unknown alert-policy keys: {sorted(unknown_alerts)}")
if missing_alerts:
    raise SystemExit(f"alert-policy keys have no incident runbook: {sorted(missing_alerts)}")

policy_deferred_entries = policy.get("deferredSignals")
if not isinstance(policy_deferred_entries, list):
    raise SystemExit("alert policy deferredSignals must be a list")
policy_deferred = {
    entry.get("name")
    for entry in policy_deferred_entries
    if isinstance(entry, dict) and isinstance(entry.get("name"), str)
}
registry_deferred = registry.get("deferredSignals")
if not isinstance(registry_deferred, dict):
    raise SystemExit("runbook registry deferredSignals must be an object")
if set(registry_deferred) != policy_deferred:
    raise SystemExit(
        f"deferred-signal registry drift: policy={sorted(policy_deferred)}, registry={sorted(registry_deferred)}"
    )

for signal, detail in registry_deferred.items():
    if not isinstance(detail, dict) or detail.get("detectionMode") != "manual_deferred":
        raise SystemExit(f"deferred signal {signal} must remain manual_deferred")
    runbook_id = detail.get("runbook")
    if runbook_id not in runbook_by_id:
        raise SystemExit(f"deferred signal {signal} points to unknown runbook {runbook_id!r}")
    if signal not in runbook_by_id[runbook_id].get("manualSignals", []):
        raise SystemExit(f"deferred signal {signal} is not declared by runbook {runbook_id}")

unknown_manual = registered_manual - policy_deferred - allowed_manual_signals
if unknown_manual:
    raise SystemExit(f"runbooks contain undeclared manual signals: {sorted(unknown_manual)}")
if "privacy_or_secret_exposure" not in registered_manual:
    raise SystemExit("privacy/security incidents need an explicit manual signal")

# Runtime authority markers: runbooks must fail when underlying recovery assumptions disappear.
def require(path: Path, markers: list[str], label: str):
    text = path.read_text()
    for marker in markers:
        if marker not in text:
            raise SystemExit(f"{label} authority marker missing from {path.relative_to(ROOT)}: {marker}")

require(AUTH, ["@Post('logout')", "@Post('logout-all')"], "auth revocation")
require(
    SESSION_AUTHORITY,
    ["async withActiveSession<", "async revokeSession(", "async revokeAllSessions("],
    "session authority",
)
require(
    CHAT,
    [
        "this.sessionAuthority.withActiveSession(",
        "session:ready",
        "session:revoked",
        "session:expired",
        "withAuthorizedRealtimeRecipients",
    ],
    "realtime session/recipient",
)
require(
    REALTIME_ADMISSION,
    ["message: [", "typing: [", "membership: [", "MAX_REALTIME_BUCKETS = 10_000"],
    "realtime admission",
)
require(
    CAREGIVER,
    [
        "async issueGrant(",
        "async acceptGrant(",
        "async declineGrant(",
        "async revokeGrant(",
        "authorityClass: 'CONTEXT_ONLY'",
        "bondXpAuthority: false",
        "recommendationEvidenceAuthority: false",
    ],
    "caregiver authority",
)
require(CONNECTOR_GUARD, ["ENABLE_DOGOS_CONNECTORS", "NotFoundException"], "connector feature guard")
require(
    CONNECTORS,
    [
        "undocumentedOAuthAllowed: false",
        "browserProviderImpersonationAllowed: false",
        "importedWearablesRewardEligible: false",
        "rawProviderPayloadStored: false",
        "remoteRevocation: 'NOT_CONFIGURED'",
        "external_object_changed_after_import",
    ],
    "connector ingestion",
)
require(
    ENV,
    [
        "CONNECTOR_CREDENTIALS_KEY is required and must decode to 32 bytes when connectors are enabled in production",
        "OPS_METRICS_TOKEN must be at least 32 characters when configured",
    ],
    "production configuration",
)
require(
    METRICS,
    [
        "userIdentifiersCollected: false",
        "petIdentifiersCollected: false",
        "providerExternalIdentifiersCollected: false",
        "rawPayloadsCollected: false",
        "requestUrlsCollected: false",
    ],
    "operational metric privacy",
)
require(
    SENTRY,
    [
        "sendDefaultPii: false",
        "delete event.request",
        "delete event.user",
        "delete event.extra",
        "delete event.breadcrumbs",
        "delete span.data",
    ],
    "Sentry privacy",
)

for entry in integrations.get("integrations", []):
    if entry.get("productionQualified") is not False:
        raise SystemExit(
            f"runbooks cannot coexist with unproven production integration claim: {entry.get('id')}"
        )

# Reject committed credential-like literals while allowing variable names and placeholders.
for entry in runbooks:
    path = ROOT / entry["path"]
    text = path.read_text()
    forbidden_patterns = [
        (r"Authorization:\s*Bearer\s+eyJ[A-Za-z0-9_-]{10,}", "literal JWT"),
        (r"AKIA[0-9A-Z]{16}", "AWS access key"),
        (r"sk-[A-Za-z0-9_-]{20,}", "provider API key"),
        (r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", "private key"),
    ]
    for pattern, label in forbidden_patterns:
        if re.search(pattern, text):
            raise SystemExit(f"runbook {entry['id']} contains a credential-like {label}")

workflow = WORKFLOW.read_text()
for marker in [
    ".github/scripts/assert-operational-runbooks.py",
    "ops/runbooks/**",
    "ops/alerts/woof-api-alert-policy.v1.json",
    "apps/api/src/auth/**",
    "apps/api/src/chat/**",
    "apps/api/src/connectors/**",
    "apps/api/src/caregiver/**",
    "apps/api/src/observability/**",
    "docs/EXTERNAL_INTEGRATION_INVENTORY.json",
    "python .github/scripts/assert-operational-runbooks.py",
]:
    if marker not in workflow:
        raise SystemExit(f"operational runbook CI ownership marker missing: {marker}")

print(
    "Operational runbook authority is coherent: every alert is routed, deferred signals remain explicit, "
    "runtime assumptions are source-bound, private evidence stays minimized, and live production rehearsal is unclaimed."
)
