#!/usr/bin/env python3
"""Fail closed if operational alert artifacts contain private identifier fields.

The guard intentionally matches forbidden tokens/field expressions rather than raw
substrings. A substring grep is unsafe here because ordinary operational prose such
as "handler" contains "handle" and would produce false positives.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


ALERT_ROOT = Path("ops/alerts")

FORBIDDEN_TOKENS = (
    "userId",
    "user_id",
    "petId",
    "pet_id",
    "externalObjectId",
    "external_object_id",
    "externalPetId",
    "external_pet_id",
    "accountId",
    "account_id",
    "request_url",
    "requestUrl",
    "email",
    "handle",
)

FORBIDDEN_EXPRESSIONS = (
    "request.body",
    "request.query",
    "request.params",
    "request.url",
    "request.originalUrl",
    "rawPayload",
    "raw_payload",
)

TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:" + "|".join(re.escape(token) for token in FORBIDDEN_TOKENS) + r")(?![A-Za-z0-9_])"
)
EXPRESSION_PATTERN = re.compile("|".join(re.escape(value) for value in FORBIDDEN_EXPRESSIONS))


def violations(text: str) -> list[str]:
    found = {match.group(0) for match in TOKEN_PATTERN.finditer(text)}
    found.update(match.group(0) for match in EXPRESSION_PATTERN.finditer(text))
    return sorted(found)


def self_test() -> None:
    allowed = (
        "handler latency",
        "request handler completed",
        "emailing an operator is external routing",
        "accounting for replicas",
        "handlebars are not user handles",
    )
    forbidden = (
        "label: userId",
        'email="person@example.test"',
        "field=handle",
        "request.body",
        "raw_payload",
        "pet_id",
    )

    for sample in allowed:
        if violations(sample):
            raise SystemExit(f"privacy guard false-positive regression: {sample!r}")
    for sample in forbidden:
        if not violations(sample):
            raise SystemExit(f"privacy guard false-negative regression: {sample!r}")

    print("Alert privacy token matcher self-test passed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()

    if not ALERT_ROOT.exists():
        raise SystemExit(f"missing alert root: {ALERT_ROOT}")

    failures: list[str] = []
    for path in sorted(ALERT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        matches = violations(text)
        if matches:
            failures.append(f"{path}: {', '.join(matches)}")

    if failures:
        print(
            "Operational alert artifacts contain forbidden private identifier/request fields:",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        raise SystemExit(1)

    print("Operational alert artifacts are free of forbidden private identifier/request fields.")


if __name__ == "__main__":
    main()
