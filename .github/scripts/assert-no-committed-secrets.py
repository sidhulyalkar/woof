#!/usr/bin/env python3
"""Fail closed on high-confidence committed credential material.

This deliberately favors a small high-confidence pattern set over noisy entropy
heuristics. Findings report only path, line number, and credential class. The
matching material itself is never printed.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

MAX_TEXT_FILE_BYTES = 5_000_000

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private-key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    (
        "github-token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ),
    (
        "aws-access-key",
        re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    ),
    (
        "slack-webhook",
        re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/_-]{20,}"),
    ),
    (
        "stripe-live-key",
        re.compile(r"\b(?:sk|rk)_live_[A-Za-z0-9]{16,}\b"),
    ),
    (
        "openai-api-key",
        re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    ),
    (
        "fly-api-token",
        re.compile(r"\bFlyV1\s+[A-Za-z0-9._-]{20,}\b"),
    ),
    (
        "npm-token",
        re.compile(r"\bnpm_[A-Za-z0-9]{20,}\b"),
    ),
)


def tracked_paths() -> list[Path]:
    raw = subprocess.check_output(["git", "ls-files", "-z"])
    return [Path(item.decode("utf-8")) for item in raw.split(b"\0") if item]


def read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except (FileNotFoundError, OSError):
        return None

    if len(data) > MAX_TEXT_FILE_BYTES or b"\0" in data:
        return None

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def main() -> int:
    findings: list[str] = []

    for path in tracked_paths():
        if not path.is_file():
            continue
        text = read_text(path)
        if text is None:
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS:
                if pattern.search(line):
                    findings.append(f"{path}:{line_number}: {label}")

    if findings:
        print("High-confidence committed credential material detected:")
        for finding in findings:
            print(f"- {finding}")
        print("Matching credential values are intentionally suppressed.")
        return 1

    print("No high-confidence committed credential material detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
