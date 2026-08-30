#!/usr/bin/env python3
"""Enforce Woof's high/critical pnpm audit policy with narrow expiring exceptions."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

BLOCKING_SEVERITIES = {"high", "critical"}
MAX_EXCEPTION_DAYS = 45
GHSA_RE = re.compile(r"GHSA-[23456789cfghjmpqrvwx]{4}-[23456789cfghjmpqrvwx]{4}-[23456789cfghjmpqrvwx]{4}", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_json", type=Path)
    parser.add_argument("exceptions_json", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"unable to read valid JSON from {path}: {exc}") from exc


def advisory_id(advisory: dict[str, Any]) -> str:
    explicit = advisory.get("github_advisory_id")
    if isinstance(explicit, str) and explicit:
        return explicit.upper()

    url = advisory.get("url")
    if isinstance(url, str):
        match = GHSA_RE.search(url)
        if match:
            return match.group(0).upper()

    raise SystemExit(
        f"blocking advisory for {advisory.get('module_name', '<unknown>')} has no GHSA identifier"
    )


def advisory_paths(advisory: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    findings = advisory.get("findings")
    if not isinstance(findings, list):
        return paths

    for finding in findings:
        if not isinstance(finding, dict):
            continue
        finding_paths = finding.get("paths")
        if not isinstance(finding_paths, list):
            continue
        paths.extend(path for path in finding_paths if isinstance(path, str))
    return paths


def validate_exception_shape(entry: dict[str, Any], today: date) -> str:
    required_strings = [
        "advisory_id",
        "package",
        "severity",
        "classification",
        "allowed_path_prefix",
        "owner",
        "expires_on",
        "rationale",
    ]
    for field in required_strings:
        if not isinstance(entry.get(field), str) or not entry[field].strip():
            raise SystemExit(f"audit exception has invalid {field}: {entry!r}")

    advisory = entry["advisory_id"].upper()
    if not GHSA_RE.fullmatch(advisory):
        raise SystemExit(f"audit exception must use one exact GHSA id, got {advisory!r}")
    if entry["severity"] not in BLOCKING_SEVERITIES:
        raise SystemExit(f"audit exception {advisory} has non-blocking severity")
    if entry["classification"] != "mobile-build-tool-only":
        raise SystemExit(f"audit exception {advisory} uses unsupported classification")
    if entry["allowed_path_prefix"] != "apps/mobile >":
        raise SystemExit(f"audit exception {advisory} must stay scoped to apps/mobile")

    markers = entry.get("required_path_substrings")
    if (
        not isinstance(markers, list)
        or not markers
        or any(not isinstance(marker, str) or not marker for marker in markers)
    ):
        raise SystemExit(f"audit exception {advisory} requires exact path markers")

    try:
        expires = date.fromisoformat(entry["expires_on"])
    except ValueError as exc:
        raise SystemExit(f"audit exception {advisory} has invalid expiry") from exc

    if expires < today:
        raise SystemExit(f"audit exception {advisory} expired on {expires.isoformat()}")
    if expires > today + timedelta(days=MAX_EXCEPTION_DAYS):
        raise SystemExit(
            f"audit exception {advisory} exceeds {MAX_EXCEPTION_DAYS}-day maximum horizon"
        )

    return advisory


def main() -> int:
    args = parse_args()
    report = load_json(args.audit_json)
    config = load_json(args.exceptions_json)

    if not isinstance(report, dict) or not isinstance(report.get("advisories"), dict):
        raise SystemExit("pnpm audit response did not contain an advisories object")
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise SystemExit("unsupported audit exception schema")

    raw_exceptions = config.get("exceptions")
    if not isinstance(raw_exceptions, list):
        raise SystemExit("audit exception config must contain an exceptions list")

    today = date.today()
    exceptions: dict[str, dict[str, Any]] = {}
    for raw in raw_exceptions:
        if not isinstance(raw, dict):
            raise SystemExit("audit exception entries must be objects")
        key = validate_exception_shape(raw, today)
        if key in exceptions:
            raise SystemExit(f"duplicate audit exception {key}")
        exceptions[key] = raw

    blocking: dict[str, dict[str, Any]] = {}
    for raw in report["advisories"].values():
        if not isinstance(raw, dict):
            continue
        severity = raw.get("severity")
        if severity not in BLOCKING_SEVERITIES:
            continue
        key = advisory_id(raw)
        if key in blocking:
            raise SystemExit(f"duplicate blocking advisory {key} in audit response")
        blocking[key] = raw

    failures: list[str] = []
    accepted: list[str] = []

    for key, advisory in sorted(blocking.items()):
        exception = exceptions.get(key)
        if exception is None:
            failures.append(
                f"unaccepted {advisory.get('severity')} advisory {key} "
                f"for {advisory.get('module_name', '<unknown>')}"
            )
            continue

        if advisory.get("module_name") != exception["package"]:
            failures.append(f"{key} package changed from accepted {exception['package']}")
            continue
        if advisory.get("severity") != exception["severity"]:
            failures.append(f"{key} severity changed from accepted {exception['severity']}")
            continue

        paths = advisory_paths(advisory)
        if not paths:
            failures.append(f"{key} has no dependency paths to prove build-tool isolation")
            continue

        prefix = exception["allowed_path_prefix"]
        markers = exception["required_path_substrings"]
        bad_paths = [
            path
            for path in paths
            if not path.startswith(prefix)
            or any(marker not in path for marker in markers)
        ]
        if bad_paths:
            failures.append(
                f"{key} escaped accepted mobile build-tool path boundary: {bad_paths[0]}"
            )
            continue

        accepted.append(
            f"{key} {exception['package']} through {exception['expires_on']} "
            f"({len(paths)} audited paths, mobile build tooling only)"
        )

    stale = sorted(set(exceptions) - set(blocking))
    if stale:
        failures.append(
            "stale audit exceptions are not allowed; remove or re-justify: " + ", ".join(stale)
        )

    if failures:
        print("Dependency audit policy failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    if accepted:
        print("Accepted time-bounded dependency findings:")
        for item in accepted:
            print(f"- {item}")
    else:
        print("No high or critical dependency advisories reported.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
