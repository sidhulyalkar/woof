#!/usr/bin/env python3
"""Fail closed if bounded operational load qualification loses authority or privacy."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ops/load/woof-bounded-load.v1.json"
HARNESS_FILES = [
    ROOT / "ops/load/run-bounded-load.mjs",
    ROOT / "ops/load/load-support.mjs",
    ROOT / "ops/load/load-scenarios.mjs",
    ROOT / "ops/load/load-telemetry.mjs",
]
WORKFLOW = ROOT / ".github/workflows/operational-load-ci.yml"
DOC = ROOT / "docs/operations/BOUNDED_LOAD_QUALIFICATION_V1.md"
ALERT_POLICY = ROOT / "ops/alerts/woof-api-alert-policy.v1.json"

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
URL_RE = re.compile(r"https?://", re.IGNORECASE)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid JSON at {path.relative_to(ROOT)}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"expected object JSON at {path.relative_to(ROOT)}")
    return value


def require_markers(text: str, markers: list[str], label: str) -> None:
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise SystemExit(f"{label} missing required markers: {missing}")


def reject_markers(text: str, markers: list[str], label: str) -> None:
    present = [marker for marker in markers if marker in text]
    if present:
        raise SystemExit(f"{label} contains forbidden markers: {present}")


def validate_profiles(config: dict[str, Any], alerts: dict[str, Any]) -> None:
    limits = config.get("resourceLimits")
    if not isinstance(limits, dict):
        raise SystemExit("bounded-load resource limits missing")
    memory = limits.get("memoryBytes")
    cpus = limits.get("nanoCpus")
    pids = limits.get("pidsLimit")
    if not isinstance(memory, int) or not 256 * 1024**2 <= memory <= 2 * 1024**3:
        raise SystemExit("bounded-load memory limit must stay finite and launch-representative")
    if not isinstance(cpus, int) or not 500_000_000 <= cpus <= 2_000_000_000:
        raise SystemExit("bounded-load CPU limit must stay finite and no larger than two CPUs")
    if not isinstance(pids, int) or not 32 <= pids <= 256:
        raise SystemExit("bounded-load PID limit must stay finite")

    profiles = config.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != {"ci", "prelaunch"}:
        raise SystemExit("bounded-load profiles must be exactly ci and prelaunch")
    caregiver_floor = alerts.get("caregiverTransition5xx", {}).get("minimumRequests")
    today_floor = alerts.get("todayReadP95Ms", {}).get("minimumRequests")
    if not isinstance(caregiver_floor, int) or not isinstance(today_floor, int):
        raise SystemExit("canonical alert-policy sample floors are unavailable")

    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise SystemExit(f"bounded-load profile {name} must be an object")
        workers = profile.get("workers")
        duration = profile.get("durationMs")
        interval = profile.get("requestIntervalMs")
        setup_pace = profile.get("setupPaceMs")
        reset = profile.get("concurrencyResetMs")
        wave_size = profile.get("transitionWaveSize")
        waves = profile.get("transitionWaves")
        if not isinstance(workers, int) or not 4 <= workers <= 16:
            raise SystemExit(f"bounded-load profile {name} workers escaped the bounded range")
        if not isinstance(duration, int) or not 20_000 <= duration <= 60_000:
            raise SystemExit(f"bounded-load profile {name} duration escaped the bounded range")
        if not isinstance(interval, int) or interval < 700 or 60_000 / interval >= 100:
            raise SystemExit(f"bounded-load profile {name} can exceed the long HTTP throttle budget")
        if not isinstance(setup_pace, int) or setup_pace < 400:
            raise SystemExit(f"bounded-load profile {name} setup can manufacture short-window 429s")
        if not isinstance(reset, int) or reset < 1_050:
            raise SystemExit(f"bounded-load profile {name} concurrency waves do not reset short throttle")
        if wave_size != 2 or not isinstance(waves, int) or waves < 2:
            raise SystemExit(f"bounded-load profile {name} replay waves must remain two-wide and repeated")
        if workers * wave_size * waves < caregiver_floor:
            raise SystemExit(f"bounded-load profile {name} misses caregiver transition sample floor")
        if profile.get("minimumAuthSamples", 0) < today_floor:
            raise SystemExit(f"bounded-load profile {name} auth sample floor is too small")
        if profile.get("minimumReadinessSamples", 0) < today_floor:
            raise SystemExit(f"bounded-load profile {name} readiness sample floor is too small")


def validate_source() -> None:
    required_paths = [CONFIG, *HARNESS_FILES, WORKFLOW, DOC, ALERT_POLICY]
    for path in required_paths:
        if not path.is_file():
            raise SystemExit(f"required bounded-load source missing: {path.relative_to(ROOT)}")

    config = load_json(CONFIG)
    alerts = load_json(ALERT_POLICY)
    if config.get("schemaVersion") != "woof-bounded-load-config-v1":
        raise SystemExit("bounded-load config schemaVersion drifted")
    if config.get("harnessVersion") != "bounded-load-v1":
        raise SystemExit("bounded-load harnessVersion drifted")
    if config.get("environmentClass") != "github-actions-production-image":
        raise SystemExit("bounded-load environment class must stay explicit")
    if config.get("alertPolicyPath") != "ops/alerts/woof-api-alert-policy.v1.json":
        raise SystemExit("bounded-load config must consume the canonical alert policy")
    validate_profiles(config, alerts)

    today = alerts.get("todayReadP95Ms", {})
    if today.get("operations") != [
        "AdventureController.getMine",
        "CompanionController.getState",
        "CompanionController.getReadiness",
        "CaregiverController.getCaregiverToday",
    ]:
        raise SystemExit("bounded-load Today authority no longer matches canonical alert operations")
    if today.get("warningMs") != 750 or today.get("criticalMs") != 1500:
        raise SystemExit("canonical Today alert latency boundaries drifted; review load qualification")

    harness = "\n".join(path.read_text() for path in HARNESS_FILES)
    require_markers(
        harness,
        [
            "createRequire(new URL('../../apps/web/package.json'",
            "requireFromWeb('socket.io-client')",
            "'fly-client-ip': clientIp",
            "'/auth/register'",
            "'/auth/me'",
            "'/intelligence/daily-signals'",
            "'/companion/state'",
            "'/companion/readiness'",
            "'/caregiver/grants'",
            "'/ops/health/ready'",
            "'/ops/metrics.json'",
            "dailySignals.filter((result) => result.payload?.duplicate === false).length !== 1",
            "divergent.status !== 409",
            "finalPayload?.bondXp !== worker.baselineBondXp",
            "rateLimited429 < 1",
            "snapshot?.release !== expectedReleaseSha",
            "privacy.userIdentifiersCollected !== false",
            "todayPolicy.minimumRequests",
            "alertPolicy.caregiverTransition5xx.minimumRequests",
            "report.invariants.realtimeSessionsReady",
            "report.invariants.realtimeDisconnectClean",
            "report.invariants.rateLimit429Observed",
            "schemaVersion: 'woof-bounded-load-report-v1'",
            "syntheticDataOnly: true",
            "accumulator,",
        ],
        "bounded-load harness",
    )
    reject_markers(
        harness,
        [
            "@woof/database",
            "PrismaClient",
            "DATABASE_URL",
            "NODE_ENV = 'test'",
            "NODE_ENV: 'test'",
            "woof-api-prod.fly.dev",
            "woof-api-staging.fly.dev",
            "console.log(payload",
            "console.log(result",
            "console.error(error",
            "client.accumulator",
        ],
        "bounded-load harness",
    )

    workflow = WORKFLOW.read_text()
    require_markers(
        workflow,
        [
            "name: dogOS Bounded Operational Load CI",
            "pgvector/pgvector:pg15",
            "QUALIFIED_SHA: ${{ github.event.pull_request.head.sha || github.sha }}",
            "ref: ${{ env.QUALIFIED_SHA }}",
            'test "$(git rev-parse HEAD)" = "$QUALIFIED_SHA"',
            "pnpm install --frozen-lockfile",
            "db:migrate:deploy",
            "node --check ops/load/run-bounded-load.mjs",
            "node --check ops/load/load-support.mjs",
            "node --check ops/load/load-scenarios.mjs",
            "node --check ops/load/load-telemetry.mjs",
            "python .github/scripts/assert-bounded-load-qualification.py",
            '--build-arg WOOF_RELEASE_SHA="${QUALIFIED_SHA}"',
            "--memory=1g",
            "--cpus=2",
            "--pids-limit=256",
            "NODE_ENV=production",
            "ENABLE_ADVENTURE_SYSTEM=true",
            "FLY_APP_NAME=woof-load-ci",
            "FLY_MACHINE_ID=bounded-load-machine",
            "OPS_METRICS_TOKEN",
            'WOOF_LOAD_EXPECTED_SHA="${QUALIFIED_SHA}"',
            "actions/upload-artifact@v7",
            "operational-load-${{ github.run_id }}",
            "retention-days: 7",
        ],
        "bounded-load workflow",
    )
    reject_markers(
        workflow,
        [
            "NODE_ENV=test",
            "NODE_ENV: test",
            "skipIf",
            "--memory=0",
            "--cpus=0",
            'WOOF_RELEASE_SHA="${GITHUB_SHA}"',
            'WOOF_LOAD_EXPECTED_SHA="${GITHUB_SHA}"',
            "continue-on-error: false",
        ],
        "bounded-load workflow",
    )

    doc = DOC.read_text()
    require_markers(
        doc,
        [
            "synthetic-only",
            "production image",
            "rate limiting stays active",
            "Daily Signals",
            "caregiver",
            "Socket.IO",
            "operational metrics",
            "750 ms",
            "1500 ms",
            "productionQualified",
            "not live-production proof",
            "literal PR head SHA",
            "Phase 5",
        ],
        "bounded-load documentation",
    )


def walk_json(value: Any, path: str = "$"):
    if isinstance(value, dict):
        for key, child in value.items():
            yield path, key, child
            yield from walk_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk_json(child, f"{path}[{index}]")


def validate_report(report_path: Path) -> None:
    report = load_json(report_path)
    config = load_json(CONFIG)

    if report.get("schemaVersion") != "woof-bounded-load-report-v1":
        raise SystemExit("bounded-load report schemaVersion drifted")
    if report.get("harnessVersion") != config.get("harnessVersion"):
        raise SystemExit("bounded-load report harness version mismatch")
    if report.get("environmentClass") != config.get("environmentClass"):
        raise SystemExit("bounded-load report environment class mismatch")
    profile_name = report.get("profile")
    if profile_name not in config.get("profiles", {}):
        raise SystemExit("bounded-load report profile is unknown")
    if report.get("resourceLimits") != config.get("resourceLimits"):
        raise SystemExit("bounded-load report resource limits do not match committed profile")

    expected = report.get("expectedReleaseSha")
    observed = report.get("observedReleaseSha")
    if not isinstance(expected, str) or not SHA_RE.fullmatch(expected):
        raise SystemExit("bounded-load report expected release SHA is invalid")
    if observed is not None and (not isinstance(observed, str) or not SHA_RE.fullmatch(observed)):
        raise SystemExit("bounded-load report observed release SHA is invalid")

    forbidden_keys = {
        "userId",
        "petId",
        "householdId",
        "grantId",
        "careEventId",
        "ledgerId",
        "socketId",
        "accessToken",
        "access_token",
        "authorization",
        "token",
        "email",
        "password",
        "endpoint",
        "note",
        "requestUrl",
        "requestBody",
        "responseBody",
        "rawPayload",
        "payload",
        "ciphertext",
    }
    for json_path, key, child in walk_json(report):
        if key in forbidden_keys:
            raise SystemExit(f"bounded-load report contains forbidden key at {json_path}: {key}")
        if isinstance(child, str):
            if UUID_RE.search(child):
                raise SystemExit(f"bounded-load report contains UUID-like identifier at {json_path}.{key}")
            if EMAIL_RE.search(child):
                raise SystemExit(f"bounded-load report contains email-like identifier at {json_path}.{key}")
            if URL_RE.search(child):
                raise SystemExit(f"bounded-load report contains URL at {json_path}.{key}")
            if "Bearer " in child:
                raise SystemExit(f"bounded-load report contains bearer material at {json_path}.{key}")

    failure_codes = report.get("failureCodes")
    if not isinstance(failure_codes, list) or any(
        not isinstance(code, str) or not re.fullmatch(r"[A-Z0-9_:-]+", code)
        for code in failure_codes
    ):
        raise SystemExit("bounded-load report failure codes must remain bounded machine labels")
    warnings = report.get("warnings")
    if not isinstance(warnings, list) or any(
        not isinstance(code, str) or not code.startswith("TODAY_READ_WARNING:") for code in warnings
    ):
        raise SystemExit("bounded-load report warnings must remain bounded Today-read labels")

    passed = report.get("passed")
    if not isinstance(passed, bool):
        raise SystemExit("bounded-load report passed field must be boolean")
    if not passed:
        if not failure_codes:
            raise SystemExit("failed bounded-load report must retain a bounded failure code")
        print("Bounded-load failure artifact is privacy-safe and structurally valid.")
        return

    if failure_codes:
        raise SystemExit("passed bounded-load report cannot retain failure codes")
    if expected != observed:
        raise SystemExit("passed bounded-load report release identity is not exact")
    invariants = report.get("invariants")
    if not isinstance(invariants, dict) or not invariants or any(value is not True for value in invariants.values()):
        raise SystemExit("passed bounded-load report is missing a proven invariant")
    telemetry = report.get("telemetry")
    if not isinstance(telemetry, dict) or telemetry.get("privacyContractPassed") is not True:
        raise SystemExit("passed bounded-load report must retain privacy-safe server telemetry evidence")
    if telemetry.get("totalServer5xx") != 0 or telemetry.get("totalDurationInvalid") != 0:
        raise SystemExit("passed bounded-load report cannot contain server failures or invalid timing")
    abuse = report.get("abuseControl")
    if not isinstance(abuse, dict) or abuse.get("success2xx", 0) < 1 or abuse.get("rateLimited429", 0) < 1:
        raise SystemExit("passed bounded-load report must prove active HTTP rate limiting")

    operations = report.get("operations")
    if not isinstance(operations, dict):
        raise SystemExit("passed bounded-load report operations are missing")
    for label in [
        "authMe",
        "adventureMine",
        "companionState",
        "companionReadiness",
        "caregiverToday",
        "healthReady",
    ]:
        evidence = operations.get(label)
        if not isinstance(evidence, dict) or evidence.get("attempts", 0) < 1:
            raise SystemExit(f"passed bounded-load report missing representative operation: {label}")
        if any(evidence.get(key) != 0 for key in ["client4xx", "rateLimited429", "server5xx", "other"]):
            raise SystemExit(f"representative operation was not clean: {label}")
        if evidence.get("success2xx") != evidence.get("attempts"):
            raise SystemExit(f"representative operation success count drifted: {label}")

    today = telemetry.get("todayReads")
    if not isinstance(today, dict) or len(today) != 4:
        raise SystemExit("passed bounded-load report must retain all four Today-read telemetry lanes")
    if any(
        value.get("classification") == "CRITICAL"
        for value in today.values()
        if isinstance(value, dict)
    ):
        raise SystemExit("passed bounded-load report cannot retain critical Today latency")

    print(
        "Bounded operational load evidence is exact-SHA-bound, synthetic-only, privacy-safe, "
        "rate-limit-aware, and free of server failures or duplicate authority drift."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    validate_source()
    if args.report:
        validate_report(args.report.resolve())
    else:
        print(
            "Bounded load qualification source is explicit: production-image load stays synthetic, "
            "resource-bounded, abuse-controlled, exact-release-bound, and privacy-safe."
        )


if __name__ == "__main__":
    main()
