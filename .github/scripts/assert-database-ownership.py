#!/usr/bin/env python3
"""Fail closed on Prisma ownership changes, with explicit add-only exceptions."""

from __future__ import annotations

import subprocess
import sys


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if len(sys.argv) < 2:
        fail("usage: assert-database-ownership.py BASE_SHA [ALLOW_ADDED_PATH ...]")

    base_sha = sys.argv[1]
    allowed_added = set(sys.argv[2:])
    diff = subprocess.run(
        [
            "git",
            "diff",
            "--name-status",
            "--find-renames",
            f"{base_sha}...HEAD",
            "--",
            "packages/database/prisma/",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    violations: list[str] = []
    for raw_line in diff.splitlines():
        if not raw_line.strip():
            continue
        fields = raw_line.split("\t")
        status = fields[0]
        paths = fields[1:]

        if status == "A" and len(paths) == 1 and paths[0] in allowed_added:
            continue

        violations.append(raw_line)

    if violations:
        print("Database ownership boundary rejected these Prisma changes:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        if allowed_added:
            print("Only these newly-added paths are allowed in this lane:", file=sys.stderr)
            for path in sorted(allowed_added):
                print(f"  A\t{path}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
