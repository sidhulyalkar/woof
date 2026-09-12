#!/usr/bin/env python3
"""Wait for the canonical exact-SHA main release checks before provider side effects."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request

SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_WORKFLOWS = (
    "ci.yml",
    "security-baseline-ci.yml",
    "codeql.yml",
    "release-promotion-authority-ci.yml",
)
DEFAULT_TIMEOUT_SECONDS = 30 * 60
DEFAULT_POLL_SECONDS = 15


@dataclass(frozen=True)
class WorkflowState:
    workflow: str
    success: bool
    states: tuple[str, ...]


def require_value(name: str, value: str | None) -> str:
    if value is None or not value.strip():
        raise SystemExit(f"{name} is required")
    return value.strip()


def matching_states(payload: dict[str, Any], release_sha: str) -> tuple[str, ...]:
    runs = payload.get("workflow_runs")
    if not isinstance(runs, list):
        return ()

    states: list[str] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        if run.get("head_sha") != release_sha or run.get("event") != "push":
            continue
        status = str(run.get("status") or "unknown")
        conclusion = run.get("conclusion")
        states.append(str(conclusion) if conclusion else status)
    return tuple(states)


def evaluate_workflow(workflow: str, payload: dict[str, Any], release_sha: str) -> WorkflowState:
    states = matching_states(payload, release_sha)
    return WorkflowState(workflow=workflow, success="success" in states, states=states)


def fetch_runs(repository: str, workflow: str, release_sha: str, token: str) -> dict[str, Any]:
    encoded_workflow = parse.quote(workflow, safe="")
    query = parse.urlencode({"head_sha": release_sha, "event": "push", "per_page": 20})
    url = (
        f"https://api.github.com/repos/{repository}/actions/workflows/"
        f"{encoded_workflow}/runs?{query}"
    )
    req = request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "woof-release-authority",
        },
    )
    try:
        with request.urlopen(req, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"GitHub Actions query failed for {workflow}: HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"GitHub Actions query failed for {workflow}: {exc.reason}") from exc


def wait_for_release_checks(
    repository: str,
    release_sha: str,
    token: str,
    timeout_seconds: int,
    poll_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds

    while True:
        states = [
            evaluate_workflow(
                workflow,
                fetch_runs(repository, workflow, release_sha, token),
                release_sha,
            )
            for workflow in REQUIRED_WORKFLOWS
        ]
        missing = [state for state in states if not state.success]
        if not missing:
            print(
                "Exact-SHA main release checks verified for "
                f"{release_sha}: {', '.join(REQUIRED_WORKFLOWS)}"
            )
            return

        summary = "; ".join(
            f"{state.workflow}={','.join(state.states) if state.states else 'not-found'}"
            for state in missing
        )
        if time.monotonic() >= deadline:
            raise SystemExit(
                "Timed out waiting for exact-SHA canonical main release checks: " + summary
            )
        print(f"Waiting for exact-SHA main release checks: {summary}")
        time.sleep(poll_seconds)


def self_test() -> None:
    release_sha = "a" * 40
    payload = {
        "workflow_runs": [
            {
                "head_sha": release_sha,
                "event": "push",
                "status": "completed",
                "conclusion": "success",
            },
            {
                "head_sha": "b" * 40,
                "event": "push",
                "status": "completed",
                "conclusion": "failure",
            },
            {
                "head_sha": release_sha,
                "event": "pull_request",
                "status": "completed",
                "conclusion": "success",
            },
        ]
    }
    state = evaluate_workflow("ci.yml", payload, release_sha)
    if not state.success or state.states != ("success",):
        raise SystemExit("release-check evaluator accepted the wrong workflow evidence")

    pending = evaluate_workflow(
        "ci.yml",
        {
            "workflow_runs": [
                {
                    "head_sha": release_sha,
                    "event": "push",
                    "status": "in_progress",
                    "conclusion": None,
                }
            ]
        },
        release_sha,
    )
    if pending.success or pending.states != ("in_progress",):
        raise SystemExit("pending release-check self-test failed")

    if any(workflow not in REQUIRED_WORKFLOWS for workflow in REQUIRED_WORKFLOWS):
        raise SystemExit("required workflow self-test failed")

    print("exact-SHA main release-check verifier self-test passed")


def main() -> None:
    if "--self-test" in sys.argv:
        self_test()
        return

    repository = require_value("GITHUB_REPOSITORY", os.environ.get("GITHUB_REPOSITORY"))
    release_sha = require_value("RELEASE_SHA", os.environ.get("RELEASE_SHA"))
    token = require_value("GH_TOKEN", os.environ.get("GH_TOKEN"))
    if not SHA_PATTERN.fullmatch(release_sha):
        raise SystemExit("RELEASE_SHA must be an exact lowercase 40-character Git SHA")

    timeout_seconds = int(os.environ.get("RELEASE_CHECK_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    poll_seconds = int(os.environ.get("RELEASE_CHECK_POLL_SECONDS", DEFAULT_POLL_SECONDS))
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise SystemExit("release-check timeout and poll interval must be positive")

    wait_for_release_checks(repository, release_sha, token, timeout_seconds, poll_seconds)


if __name__ == "__main__":
    main()
