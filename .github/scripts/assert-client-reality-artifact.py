#!/usr/bin/env python3
"""Fail closed if Client Reality stops sharing one verified production Web build."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/client-reality-ci.yml"
ARTIFACT = ROOT / ".github/scripts/client-reality-web-artifact.py"

for path in [WORKFLOW, ARTIFACT]:
    if not path.is_file():
        raise SystemExit(f"required shared-artifact source missing: {path.relative_to(ROOT)}")

workflow = WORKFLOW.read_text()
artifact = ARTIFACT.read_text()

required_workflow = [
    "NEXT_PUBLIC_API_URL: http://127.0.0.1:59999/api/v1",
    "web-build:",
    "name: Build exact production Web artifact once",
    "needs: contract",
    "Build Web client for production-server qualification",
    "python .github/scripts/client-reality-web-artifact.py create",
    '--event-head-sha "${{ github.event.pull_request.head.sha }}"',
    '--event-base-sha "${{ github.event.pull_request.base.sha }}"',
    '--api-url "${NEXT_PUBLIC_API_URL}"',
    "uses: actions/upload-artifact@v7",
    "name: client-reality-web-${{ github.run_id }}",
    "browser-matrix:",
    "needs: [contract, web-build]",
    "uses: actions/download-artifact@v8",
    "Verify and restore exact Web build artifact",
    "python .github/scripts/client-reality-web-artifact.py verify",
    "python .github/scripts/client-reality-web-artifact.py self-test",
    "playwright install --with-deps ${{ matrix.browser }}",
    "slug: desktop-chromium",
    "slug: mobile-chromium",
    "slug: desktop-firefox",
    "slug: desktop-webkit",
]
missing = [marker for marker in required_workflow if marker not in workflow]
if missing:
    raise SystemExit(f"Client Reality shared-artifact workflow is incomplete: {missing}")

if "NEXT_PUBLIC_API_URL: ${{ env." in workflow:
    raise SystemExit(
        "Client Reality must not define a job env variable from another env value; "
        "the workflow-level public API URL is the single build/runtime authority"
    )

build_command = "pnpm --filter @woof/web build"
if workflow.count(build_command) != 1:
    raise SystemExit(
        "Client Reality must compile the production Web exactly once per workflow run"
    )
browser_start = workflow.index("  browser-matrix:")
if build_command in workflow[browser_start:]:
    raise SystemExit("browser consumers must never rebuild the shared production Web artifact")
if "pnpm --filter @woof/web dev" in workflow:
    raise SystemExit("Client Reality release evidence must never fall back to next dev")

producer_start = workflow.index("  web-build:")
if not producer_start < workflow.index(build_command) < browser_start:
    raise SystemExit("the only production Web build must belong to the artifact producer")

required_artifact = [
    'SCHEMA_VERSION = 1',
    'ARCHIVE_NAME = "client-reality-web.tar.gz"',
    '"checkoutSha"',
    '"eventHeadSha"',
    '"eventBaseSha"',
    '"apiUrl"',
    '"buildTreeSha256"',
    '"nodeVersion"',
    '"pnpmVersion"',
    'excludedBuildPaths',
    'relative.parts[0] == "cache"',
    "sha256_file(archive)",
    "git rev-parse",
    "Unsafe Client Reality artifact member",
    "Unexpected Client Reality artifact member",
    "Client Reality artifact checksum mismatch",
    "Client Reality artifact manifest mismatch",
    "tree digest does not match its manifest",
    "Self-test failed to reject a mismatched PR head SHA",
    "Self-test failed to reject a tampered artifact",
]
missing = [marker for marker in required_artifact if marker not in artifact]
if missing:
    raise SystemExit(f"Client Reality artifact verifier is incomplete: {missing}")

for forbidden in [
    "extractall(root)\n",
    "eventHeadSha = checkoutSha",
    "sha256_file(archive) == sha256_file(archive)",
]:
    if forbidden in artifact:
        raise SystemExit(f"forbidden shared-artifact shortcut detected: {forbidden!r}")

print(
    "Client Reality artifact contract preserves one build, checked transfer bytes, "
    "actual checkout identity, PR head/base provenance, build config, and four independent engines."
)
