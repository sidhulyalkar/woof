#!/usr/bin/env python3
"""Fail closed on Prisma ownership changes, with explicit add-only exceptions.

The guard can also be scoped to domain-owned files. That lets shared composition
changes (for example app.module.ts) run a subsystem's regression lane without
making that subsystem veto a database migration owned and qualified elsewhere.
"""

from __future__ import annotations

import argparse
from fnmatch import fnmatch
import subprocess
import sys


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reject Prisma ownership changes for a release slice."
    )
    parser.add_argument("base_sha")
    parser.add_argument(
        "allowed_added",
        nargs="*",
        help="Exact newly-added Prisma paths explicitly owned by this lane.",
    )
    parser.add_argument(
        "--enforce-if-changed",
        action="append",
        default=[],
        metavar="GLOB",
        help=(
            "Only enforce the database boundary when at least one changed file "
            "matches this domain-owned glob. Repeat for multiple globs."
        ),
    )
    args = parser.parse_intermixed_args()

    if args.enforce_if_changed:
        changed_files = git_output(
            "diff",
            "--name-only",
            f"{args.base_sha}...HEAD",
        ).splitlines()
        owned_changes = [
            path
            for path in changed_files
            if any(fnmatch(path, pattern) for pattern in args.enforce_if_changed)
        ]
        if not owned_changes:
            print(
                "Database ownership guard skipped: no domain-owned files changed; "
                "shared composition regressions may still run."
            )
            return
        print("Database ownership guard active for domain changes:")
        for path in owned_changes:
            print(f"  {path}")

    allowed_added = set(args.allowed_added)
    diff = git_output(
        "diff",
        "--name-status",
        "--find-renames",
        f"{args.base_sha}...HEAD",
        "--",
        "packages/database/prisma/",
    )

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
