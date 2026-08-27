#!/usr/bin/env python3
"""Fail closed on Prisma ownership changes, with explicit add-only exceptions.

The guard can also be scoped to domain-owned files. That lets shared composition
changes (for example app.module.ts) run a subsystem's regression lane without
making that subsystem veto a database migration owned and qualified elsewhere.
"""

from __future__ import annotations

import argparse
from fnmatch import fnmatch
import os
import subprocess
import sys


LEGACY_WORKFLOW_SCOPES: dict[str, tuple[str, ...]] = {
    "dogOS Session Migration Immutability CI": (
        "packages/database/prisma/migrations/20260824233000_add_dogos_auth_sessions/**",
    ),
    "dogOS Realtime Session Readiness CI": (
        "apps/api/src/auth/session-authority.service.ts",
        "apps/api/src/chat/chat.gateway.ts",
        "apps/api/src/chat/chat.gateway.spec.ts",
        "apps/api/src/chat/chat-session-ready.spec.ts",
        "apps/web/src/lib/socket.ts",
        "apps/web/src/lib/socket.spec.ts",
        "packages/database/prisma/migrations/20260824233000_add_dogos_auth_sessions/**",
        "docs/DOGOS_SESSION_AUTHORITY.md",
    ),
    "dogOS Session Authority CI": (
        "apps/api/src/auth/**",
        "apps/api/src/chat/chat.gateway.ts",
        "apps/api/src/chat/chat.gateway.spec.ts",
        "apps/api/src/chat/chat-session-ready.spec.ts",
        "apps/api/src/chat/chat-security.service.ts",
        "apps/api/src/chat/chat-security.service.spec.ts",
        "apps/api/src/chat/chat.module.ts",
        "apps/web/src/lib/api.ts",
        "apps/web/src/lib/socket.ts",
        "apps/web/src/lib/socket.spec.ts",
        "apps/mobile/src/api/auth.ts",
        "packages/database/prisma/migrations/20260824233000_add_dogos_auth_sessions/**",
        "docs/DOGOS_SESSION_AUTHORITY.md",
    ),
}


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

    scope_patterns = list(args.enforce_if_changed)
    if not scope_patterns:
        scope_patterns.extend(LEGACY_WORKFLOW_SCOPES.get(os.environ.get("GITHUB_WORKFLOW", ""), ()))

    if scope_patterns:
        changed_files = git_output(
            "diff",
            "--name-only",
            f"{args.base_sha}...HEAD",
        ).splitlines()
        owned_changes = [
            path
            for path in changed_files
            if not path.startswith(".github/workflows/")
            and path != ".github/scripts/assert-database-ownership.py"
            and any(fnmatch(path, pattern) for pattern in scope_patterns)
        ]
        if not owned_changes:
            print(
                "Database ownership guard skipped: no domain-owned files changed; "
                "shared composition and qualification regressions may still run."
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
