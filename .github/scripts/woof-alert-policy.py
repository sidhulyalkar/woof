#!/usr/bin/env python3
"""Render and verify Woof's privacy-safe operational alert policy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "ops" / "alerts" / "woof-api-alert-policy.v1.json"
RULES_PATH = ROOT / "ops" / "alerts" / "woof-api.rules.yml"
FIXTURES_PATH = ROOT / "ops" / "alerts" / "fixtures" / "woof-api-alert-fixtures.v1.json"

CONTROLLER_SOURCES = {
    "AuthController": ROOT / "apps" / "api" / "src" / "auth" / "auth.controller.ts",
    "AdventureController": ROOT / "apps" / "api" / "src" / "adventure" / "adventure.controller.ts",
    "CompanionController": ROOT / "apps" / "api" / "src" / "companion" / "companion.controller.ts",
    "CaregiverController": ROOT / "apps" / "api" / "src" / "caregiver" / "caregiver.controller.ts",
    "ObservabilityController": ROOT
    / "apps"
    / "api"
    / "src"
    / "observability"
    / "observability.controller.ts",
}

DURATION_PATTERN = re.compile(r"^[1-9][0-9]*(?:s|m|h|d)$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_duration(name: str, value: Any) -> None:
    if not isinstance(value, str) or DURATION_PATTERN.fullmatch(value) is None:
        raise SystemExit(f"{name} must be a positive Prometheus duration such as 30s, 5m, or 1h")


def split_operation(operation: str) -> tuple[str, str]:
    parts = operation.split(".")
    if len(parts) != 2 or any(IDENTIFIER_PATTERN.fullmatch(part) is None for part in parts):
        raise SystemExit(
            f"operation {operation!r} must be an exact Controller.method identifier without dynamic data"
        )
    return parts[0], parts[1]


def operation_regex(operations: Iterable[str]) -> str:
    patterns: list[str] = []
    for operation in operations:
        controller, method = split_operation(operation)
        patterns.append(f"{controller}[.]{method}")
    if not patterns:
        raise SystemExit("alert operation lists cannot be empty")
    return "|".join(patterns)


def validate_operation_source(operation: str) -> None:
    controller, method = split_operation(operation)
    source_path = CONTROLLER_SOURCES.get(controller)
    if source_path is None:
        raise SystemExit(f"no source binding is registered for alert controller {controller}")
    source = source_path.read_text()
    if re.search(rf"\bclass\s+{re.escape(controller)}\b", source) is None:
        raise SystemExit(f"alert controller {controller} no longer exists in {source_path}")
    if re.search(rf"\b(?:async\s+)?{re.escape(method)}\s*\(", source) is None:
        raise SystemExit(f"alert operation {operation} no longer exists in {source_path}")


def validate_ratio_policy(name: str, config: dict[str, Any], minimum_key: str) -> None:
    warning = config.get("warningRatio")
    critical = config.get("criticalRatio")
    minimum = config.get(minimum_key)
    if not isinstance(warning, (int, float)) or not isinstance(critical, (int, float)):
        raise SystemExit(f"{name} warningRatio and criticalRatio must be numeric")
    if not 0 < warning < critical < 1:
        raise SystemExit(f"{name} must satisfy 0 < warningRatio < criticalRatio < 1")
    if not isinstance(minimum, int) or minimum <= 0:
        raise SystemExit(f"{name}.{minimum_key} must be a positive integer")
    warning_bad_count = math.ceil(warning * minimum)
    critical_bad_count = math.ceil(critical * minimum)
    if warning_bad_count >= critical_bad_count:
        raise SystemExit(
            f"{name}.{minimum_key}={minimum} makes warning and critical indistinguishable "
            f"at the sample floor ({warning_bad_count} bad samples trigger both); raise the floor "
            "or separate the ratios"
        )
    validate_duration(f"{name}.window", config.get("window"))
    validate_duration(f"{name}.warningFor", config.get("warningFor"))
    validate_duration(f"{name}.criticalFor", config.get("criticalFor"))


def validate_count_policy(name: str, config: dict[str, Any]) -> None:
    warning = config.get("warningCount")
    critical = config.get("criticalCount")
    if not isinstance(warning, int) or not isinstance(critical, int):
        raise SystemExit(f"{name} warningCount and criticalCount must be integers")
    if not 0 < warning < critical:
        raise SystemExit(f"{name} warningCount must be positive and below criticalCount")
    validate_duration(f"{name}.window", config.get("window"))
    validate_duration(f"{name}.warningFor", config.get("warningFor"))
    validate_duration(f"{name}.criticalFor", config.get("criticalFor"))


def validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("version") != "woof-api-alert-policy-v1":
        raise SystemExit("alert policy version must remain explicitly bound to woof-api-alert-policy-v1")
    if policy.get("service") != "woof-api":
        raise SystemExit("alert policy service must remain the low-cardinality woof-api identity")
    validate_duration("evaluationInterval", policy.get("evaluationInterval"))

    missing = policy.get("telemetryMissing", {})
    validate_duration("telemetryMissing.warningFor", missing.get("warningFor"))
    validate_duration("telemetryMissing.criticalFor", missing.get("criticalFor"))
    unknown = policy.get("unknownRelease", {})
    validate_duration("unknownRelease.warningFor", unknown.get("warningFor"))

    readiness = policy.get("readinessFailures", {})
    if not isinstance(readiness.get("warningFailures"), int) or not isinstance(
        readiness.get("criticalFailures"), int
    ):
        raise SystemExit("readiness failure thresholds must be integers")
    if not 0 < readiness["warningFailures"] < readiness["criticalFailures"]:
        raise SystemExit("readiness warningFailures must be positive and below criticalFailures")
    validate_duration("readinessFailures.window", readiness.get("window"))
    validate_duration("readinessFailures.warningFor", readiness.get("warningFor"))
    validate_duration("readinessFailures.criticalFor", readiness.get("criticalFor"))
    validate_operation_source(readiness.get("operation", ""))

    http = policy.get("http5xx", {})
    auth = policy.get("auth5xx", {})
    caregiver = policy.get("caregiverTransition5xx", {})
    connector = policy.get("connectorRejected", {})
    validate_ratio_policy("http5xx", http, "minimumRequests")
    validate_ratio_policy("auth5xx", auth, "minimumRequests")
    validate_ratio_policy("caregiverTransition5xx", caregiver, "minimumRequests")
    validate_ratio_policy("connectorRejected", connector, "minimumImports")

    operation_sets = {
        "http5xx.excludedOperations": http.get("excludedOperations"),
        "auth5xx.operations": auth.get("operations"),
        "todayReadP95Ms.operations": policy.get("todayReadP95Ms", {}).get("operations"),
        "caregiverTransition5xx.operations": caregiver.get("operations"),
    }
    for name, operations in operation_sets.items():
        if not isinstance(operations, list) or not operations or not all(
            isinstance(operation, str) for operation in operations
        ):
            raise SystemExit(f"{name} must be a non-empty list of exact controller operations")
        if len(operations) != len(set(operations)):
            raise SystemExit(f"{name} cannot contain duplicate operations")
        for operation in operations:
            validate_operation_source(operation)

    today = policy.get("todayReadP95Ms", {})
    quantile = today.get("quantile")
    warning_ms = today.get("warningMs")
    critical_ms = today.get("criticalMs")
    minimum_requests = today.get("minimumRequests")
    if not isinstance(quantile, (int, float)) or not 0 < quantile < 1:
        raise SystemExit("todayReadP95Ms.quantile must be between 0 and 1")
    if not isinstance(warning_ms, (int, float)) or not isinstance(critical_ms, (int, float)):
        raise SystemExit("todayReadP95Ms latency thresholds must be numeric")
    if not 0 < warning_ms < critical_ms:
        raise SystemExit("todayReadP95Ms warningMs must be positive and below criticalMs")
    if not isinstance(minimum_requests, int) or minimum_requests <= 0:
        raise SystemExit("todayReadP95Ms.minimumRequests must be a positive integer")
    validate_duration("todayReadP95Ms.window", today.get("window"))
    validate_duration("todayReadP95Ms.warningFor", today.get("warningFor"))
    validate_duration("todayReadP95Ms.criticalFor", today.get("criticalFor"))

    validate_count_policy("requestDurationInvalid", policy.get("requestDurationInvalid", {}))
    validate_count_policy("deviceContractRejections", policy.get("deviceContractRejections", {}))

    deferred = policy.get("deferredSignals")
    if not isinstance(deferred, list) or not deferred:
        raise SystemExit("deferredSignals must explicitly preserve unavailable operational signals")
    deferred_names = [entry.get("name") for entry in deferred if isinstance(entry, dict)]
    if len(deferred_names) != len(deferred) or len(deferred_names) != len(set(deferred_names)):
        raise SystemExit("deferredSignals must be uniquely named objects")


def alert_rule(
    *,
    name: str,
    expr: str,
    duration: str,
    severity: str,
    service: str,
    policy_version: str,
    operation_class: str,
    summary: str,
    description: str,
    static_release: str | None = None,
) -> list[str]:
    labels = [
        f'        severity: "{severity}"',
        f'        service: "{service}"',
        f'        operation_class: "{operation_class}"',
        f'        policy_version: "{policy_version}"',
    ]
    if static_release is not None:
        labels.append(f'        release: "{static_release}"')
    return [
        f"    - alert: {name}",
        "      expr: >-",
        *[f"        {line}" for line in expr.splitlines()],
        f"      for: {duration}",
        "      labels:",
        *labels,
        "      annotations:",
        f'        summary: "{summary}"',
        f'        description: "{description}"',
    ]


def ratio_expr(
    *,
    metric: str,
    service: str,
    window: str,
    bad_selector: str,
    ratio: float,
    minimum: int,
    group_by: tuple[str, ...] = ("release",),
    selectors: tuple[str, ...] = (),
    denominator_selector: str | None = None,
) -> str:
    grouping = ", ".join(group_by)
    selector_prefix = ",".join((f'service="{service}"', *selectors))
    bad = f"{selector_prefix},{bad_selector}"
    denominator = selector_prefix
    if denominator_selector is not None:
        denominator = f"{denominator},{denominator_selector}"
    return "\n".join(
        [
            "(",
            f"  sum by ({grouping}) (rate({metric}{{{bad}}}[{window}]))",
            "  /",
            f"  clamp_min(sum by ({grouping}) (rate({metric}{{{denominator}}}[{window}])), 0.000001)",
            f") >= {ratio}",
            f"and on ({grouping})",
            f"sum by ({grouping}) (increase({metric}{{{denominator}}}[{window}])) >= {minimum}",
        ]
    )


def render_rules(policy: dict[str, Any]) -> str:
    service = policy["service"]
    version = policy["version"]
    lines = [
        "# GENERATED by .github/scripts/woof-alert-policy.py from woof-api-alert-policy.v1.json.",
        "# Do not edit thresholds here. Change the policy JSON and regenerate.",
        "groups:",
        "  - name: woof-api-operational-alerts-v1",
        f"    interval: {policy['evaluationInterval']}",
        "    rules:",
    ]

    def add(**kwargs: Any) -> None:
        lines.extend(alert_rule(service=service, policy_version=version, **kwargs))

    missing = policy["telemetryMissing"]
    missing_expr = f'absent(woof_process_uptime_seconds{{service="{service}"}})'
    for severity, duration in [
        ("warning", missing["warningFor"]),
        ("critical", missing["criticalFor"]),
    ]:
        add(
            name=f"WoofOperationalMetricsMissing{severity.title()}",
            expr=missing_expr,
            duration=duration,
            severity=severity,
            operation_class="telemetry",
            summary=f"Woof API operational metrics are missing ({severity})",
            description="The API metrics heartbeat is absent. Missing telemetry is an observability incident, not evidence of health.",
            static_release="unknown",
        )

    unknown = policy["unknownRelease"]
    add(
        name="WoofReleaseIdentityUnknownWarning",
        expr=f'woof_release_info{{service="{service}",release="unknown"}} == 1',
        duration=unknown["warningFor"],
        severity="warning",
        operation_class="release_identity",
        summary="Woof API is exporting an unknown release identity",
        description="Operational evidence is not tied to an exact 40-hex Git SHA; do not promote this process as a qualified release.",
    )

    readiness = policy["readinessFailures"]
    readiness_selector = f'service="{service}",operation="{readiness["operation"]}"'
    readiness_missing_expr = "\n".join(
        [
            f'max by (release) (woof_release_info{{service="{service}"}})',
            "unless on (release)",
            f"count by (release) (woof_http_requests_total{{{readiness_selector}}})",
        ]
    )
    for severity, duration in [
        ("warning", missing["warningFor"]),
        ("critical", missing["criticalFor"]),
    ]:
        add(
            name=f"WoofReadinessProbeMissing{severity.title()}",
            expr=readiness_missing_expr,
            duration=duration,
            severity=severity,
            operation_class="readiness",
            summary=f"Woof API readiness probe telemetry is missing ({severity})",
            description="A scraped Woof release has no readiness-probe series. Verify external probing before treating that release as healthy.",
        )
    for severity, threshold, duration in [
        ("warning", readiness["warningFailures"], readiness["warningFor"]),
        ("critical", readiness["criticalFailures"], readiness["criticalFor"]),
    ]:
        expr = (
            f'sum by (release) (increase(woof_http_requests_total{{{readiness_selector},status_class="5xx"}}'
            f'[{readiness["window"]}])) >= {threshold}'
        )
        add(
            name=f"WoofReadinessFailures{severity.title()}",
            expr=expr,
            duration=duration,
            severity=severity,
            operation_class="readiness",
            summary=f"Woof API readiness is failing ({severity})",
            description="Database-backed readiness has returned server errors within the bounded alert window for release {{ $labels.release }}.",
        )

    server_outcome_selector = 'status_class=~"2xx|3xx|5xx"'
    http = policy["http5xx"]
    http_selector = f'operation!~"{operation_regex(http["excludedOperations"])}"'
    for severity, ratio, duration in [
        ("warning", http["warningRatio"], http["warningFor"]),
        ("critical", http["criticalRatio"], http["criticalFor"]),
    ]:
        add(
            name=f"WoofHttp5xxRatio{severity.title()}",
            expr=ratio_expr(
                metric="woof_http_requests_total",
                service=service,
                window=http["window"],
                bad_selector='status_class="5xx"',
                ratio=ratio,
                minimum=http["minimumRequests"],
                selectors=(http_selector,),
                denominator_selector=server_outcome_selector,
            ),
            duration=duration,
            severity=severity,
            operation_class="http_service",
            summary=f"Woof API application 5xx ratio is elevated ({severity})",
            description="The bounded application-request 5xx ratio exceeded policy for release {{ $labels.release }} after the minimum eligible-request floor was met. Health/metrics endpoints and 4xx client outcomes are excluded from this ratio.",
        )

    auth = policy["auth5xx"]
    auth_selector = f'operation=~"{operation_regex(auth["operations"])}"'
    for severity, ratio, duration in [
        ("warning", auth["warningRatio"], auth["warningFor"]),
        ("critical", auth["criticalRatio"], auth["criticalFor"]),
    ]:
        add(
            name=f"WoofAuth5xxRatio{severity.title()}",
            expr=ratio_expr(
                metric="woof_http_requests_total",
                service=service,
                window=auth["window"],
                bad_selector='status_class="5xx"',
                ratio=ratio,
                minimum=auth["minimumRequests"],
                group_by=("release", "operation"),
                selectors=(auth_selector,),
                denominator_selector=server_outcome_selector,
            ),
            duration=duration,
            severity=severity,
            operation_class="auth_session",
            summary=f"Woof authentication/session operation 5xx ratio is elevated ({severity})",
            description="Auth/session operation {{ $labels.operation }} is returning server errors above policy for release {{ $labels.release }}. Expected 4xx authentication denials are excluded from both numerator and denominator.",
        )

    today = policy["todayReadP95Ms"]
    today_selector = (
        f'service="{service}",operation=~"{operation_regex(today["operations"])}",status_class="2xx"'
    )
    for severity, threshold, duration in [
        ("warning", today["warningMs"], today["warningFor"]),
        ("critical", today["criticalMs"], today["criticalFor"]),
    ]:
        expr = "\n".join(
            [
                "(",
                f"  histogram_quantile({today['quantile']},",
                "    sum by (le, release, operation) (",
                f"      rate(woof_http_request_duration_ms_bucket{{{today_selector}}}[{today['window']}])",
                "    )",
                "  )",
                f") >= {threshold}",
                "and on (release, operation)",
                f"sum by (release, operation) (increase(woof_http_request_duration_ms_count{{{today_selector}}}[{today['window']}])) >= {today['minimumRequests']}",
            ]
        )
        add(
            name=f"WoofTodayReadP95Latency{severity.title()}",
            expr=expr,
            duration=duration,
            severity=severity,
            operation_class="today_read",
            summary=f"Woof Today/read operation p95 latency is elevated ({severity})",
            description="Histogram-derived successful-response p95 handler latency for {{ $labels.operation }} exceeded the initial operator guardrail for release {{ $labels.release }} after the minimum sample floor was met.",
        )

    invalid_duration = policy["requestDurationInvalid"]
    for severity, threshold, duration in [
        ("warning", invalid_duration["warningCount"], invalid_duration["warningFor"]),
        ("critical", invalid_duration["criticalCount"], invalid_duration["criticalFor"]),
    ]:
        expr = (
            f'sum by (release, operation) (increase(woof_http_request_duration_invalid_total{{service="{service}"}}'
            f'[{invalid_duration["window"]}])) >= {threshold}'
        )
        add(
            name=f"WoofRequestDurationTelemetryInvalid{severity.title()}",
            expr=expr,
            duration=duration,
            severity=severity,
            operation_class="telemetry_quality",
            summary=f"Woof request-duration telemetry is invalid ({severity})",
            description="Request timing for {{ $labels.operation }} produced non-finite or negative samples for release {{ $labels.release }}. Invalid samples are excluded from latency histograms rather than recorded as fake zeroes.",
        )

    caregiver = policy["caregiverTransition5xx"]
    caregiver_selector = f'operation=~"{operation_regex(caregiver["operations"])}"'
    for severity, ratio, duration in [
        ("warning", caregiver["warningRatio"], caregiver["warningFor"]),
        ("critical", caregiver["criticalRatio"], caregiver["criticalFor"]),
    ]:
        add(
            name=f"WoofCaregiverTransition5xxRatio{severity.title()}",
            expr=ratio_expr(
                metric="woof_http_requests_total",
                service=service,
                window=caregiver["window"],
                bad_selector='status_class="5xx"',
                ratio=ratio,
                minimum=caregiver["minimumRequests"],
                group_by=("release", "operation"),
                selectors=(caregiver_selector,),
                denominator_selector=server_outcome_selector,
            ),
            duration=duration,
            severity=severity,
            operation_class="caregiver_transition",
            summary=f"Woof caregiver transition operation 5xx ratio is elevated ({severity})",
            description="Pet-scoped caregiver operation {{ $labels.operation }} is returning server errors above policy for release {{ $labels.release }}. Client/authorization 4xx outcomes do not dilute the server-error ratio.",
        )

    connector = policy["connectorRejected"]
    for severity, ratio, duration in [
        ("warning", connector["warningRatio"], connector["warningFor"]),
        ("critical", connector["criticalRatio"], connector["criticalFor"]),
    ]:
        add(
            name=f"WoofConnectorRejectedRatio{severity.title()}",
            expr=ratio_expr(
                metric="woof_connector_imports_total",
                service=service,
                window=connector["window"],
                bad_selector='outcome="REJECTED"',
                ratio=ratio,
                minimum=connector["minimumImports"],
                group_by=("release", "provider", "kind"),
            ),
            duration=duration,
            severity=severity,
            operation_class="connector_import",
            summary=f"Woof connector rejection ratio is elevated ({severity})",
            description="Verified {{ $labels.provider }} {{ $labels.kind }} imports are being rejected above the initial operator guardrail for release {{ $labels.release }}.",
        )

    device = policy["deviceContractRejections"]
    for severity, threshold, duration in [
        ("warning", device["warningCount"], device["warningFor"]),
        ("critical", device["criticalCount"], device["criticalFor"]),
    ]:
        expr = (
            f'sum by (release) (increase(woof_device_contract_rejections_total{{service="{service}"}}'
            f'[{device["window"]}])) >= {threshold}'
        )
        add(
            name=f"WoofDeviceContractRejections{severity.title()}",
            expr=expr,
            duration=duration,
            severity=severity,
            operation_class="device_contract",
            summary=f"Woof device-contract rejections are elevated ({severity})",
            description="Untrusted device envelopes are being rejected above policy for release {{ $labels.release }}. Inspect partner transport without logging raw payloads.",
        )

    rendered = "\n".join(lines) + "\n"
    validate_rendered_rules(rendered)
    return rendered


def validate_rendered_rules(rendered: str) -> None:
    required = [
        'operation!~"ObservabilityController[.]liveness|ObservabilityController[.]readiness|ObservabilityController[.]prometheus|ObservabilityController[.]snapshot"',
        'status_class=~"2xx|3xx|5xx"',
        'status_class="2xx"',
        "sum by (release, operation)",
        "sum by (le, release, operation)",
        "and on (release, operation)",
        "sum by (release, provider, kind)",
        "and on (release, provider, kind)",
        "unless on (release)",
        "woof_http_request_duration_invalid_total",
    ]
    for marker in required:
        if marker not in rendered:
            raise SystemExit(f"generated alert rules lost required grouping/exclusion contract: {marker}")
    if "\\." in rendered:
        raise SystemExit("generated PromQL must not rely on invalid backslash-dot Go string escapes")
    if rendered.count("    - alert: ") != 21:
        raise SystemExit("woof-api-alert-policy-v1 must render exactly 21 named alert rules")


def classify_ratio(total: float, bad: float, minimum: int, warning: float, critical: float) -> str:
    if total < minimum:
        return "INSUFFICIENT_DATA"
    ratio = bad / total if total > 0 else 0
    if ratio >= critical:
        return "CRITICAL"
    if ratio >= warning:
        return "WARNING"
    return "OK"


def classify_count(value: float, warning: float, critical: float) -> str:
    if value >= critical:
        return "CRITICAL"
    if value >= warning:
        return "WARNING"
    return "OK"


def classify_fixture(policy: dict[str, Any], fixture: dict[str, Any]) -> dict[str, str]:
    sample = fixture["sample"]
    if not sample["telemetryPresent"]:
        return {
            "telemetry": "UNKNOWN",
            "release": "UNKNOWN",
            "readiness": "UNKNOWN",
            "http5xx": "UNKNOWN",
            "auth5xx": "UNKNOWN",
            "todayReadP95Ms": "UNKNOWN",
            "requestDurationInvalid": "UNKNOWN",
            "caregiverTransition5xx": "UNKNOWN",
            "connectorRejected": "UNKNOWN",
            "deviceContractRejections": "UNKNOWN",
        }

    readiness = policy["readinessFailures"]
    http = policy["http5xx"]
    auth = policy["auth5xx"]
    today = policy["todayReadP95Ms"]
    invalid_duration = policy["requestDurationInvalid"]
    caregiver = policy["caregiverTransition5xx"]
    connector = policy["connectorRejected"]
    device = policy["deviceContractRejections"]

    today_state = "INSUFFICIENT_DATA"
    if sample["todayRequests"] >= today["minimumRequests"]:
        today_state = classify_count(sample["todayP95Ms"], today["warningMs"], today["criticalMs"])

    return {
        "telemetry": "OK",
        "release": "WARNING" if sample["release"] == "unknown" else "OK",
        "readiness": classify_count(
            sample["readinessFailures"],
            readiness["warningFailures"],
            readiness["criticalFailures"],
        ),
        "http5xx": classify_ratio(
            sample["httpRequests"],
            sample["http5xx"],
            http["minimumRequests"],
            http["warningRatio"],
            http["criticalRatio"],
        ),
        "auth5xx": classify_ratio(
            sample["authRequests"],
            sample["auth5xx"],
            auth["minimumRequests"],
            auth["warningRatio"],
            auth["criticalRatio"],
        ),
        "todayReadP95Ms": today_state,
        "requestDurationInvalid": classify_count(
            sample["requestDurationInvalid"],
            invalid_duration["warningCount"],
            invalid_duration["criticalCount"],
        ),
        "caregiverTransition5xx": classify_ratio(
            sample["caregiverTransitions"],
            sample["caregiver5xx"],
            caregiver["minimumRequests"],
            caregiver["warningRatio"],
            caregiver["criticalRatio"],
        ),
        "connectorRejected": classify_ratio(
            sample["connectorImports"],
            sample["connectorRejected"],
            connector["minimumImports"],
            connector["warningRatio"],
            connector["criticalRatio"],
        ),
        "deviceContractRejections": classify_count(
            sample["deviceContractRejections"],
            device["warningCount"],
            device["criticalCount"],
        ),
    }


def verify_fixtures(policy: dict[str, Any]) -> None:
    fixtures = load_json(FIXTURES_PATH)
    if fixtures.get("policyVersion") != policy["version"]:
        raise SystemExit("alert fixtures are not bound to the current policy version")
    for fixture in fixtures["fixtures"]:
        actual = classify_fixture(policy, fixture)
        expected = fixture["expected"]
        if actual != expected:
            raise SystemExit(
                f"alert fixture {fixture['name']!r} failed:\nexpected={expected}\nactual={actual}"
            )
    print(f"Verified {len(fixtures['fixtures'])} deterministic alert-policy fixtures.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if committed rules differ")
    parser.add_argument("--verify-fixtures", action="store_true", help="run deterministic policy fixtures")
    args = parser.parse_args()

    policy = load_json(POLICY_PATH)
    validate_policy(policy)
    rendered = render_rules(policy)

    if args.check:
        committed = RULES_PATH.read_text() if RULES_PATH.exists() else ""
        if committed != rendered:
            raise SystemExit(
                "committed alert rules are stale; regenerate with: "
                "python .github/scripts/woof-alert-policy.py > ops/alerts/woof-api.rules.yml"
            )
        print("Committed Prometheus alert rules match the versioned Woof alert policy.")
    elif not args.verify_fixtures:
        print(rendered, end="")

    if args.verify_fixtures:
        verify_fixtures(policy)


if __name__ == "__main__":
    main()
