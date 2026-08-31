#!/usr/bin/env python3
"""Fail closed if Behavior Vision provider failures regain private diagnostic logging."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "apps/api/src/behavior-vision/behavior-vision.model.ts"
TEST = ROOT / "apps/api/src/behavior-vision/behavior-vision.model.spec.ts"
WORKFLOW = ROOT / ".github/workflows/dogos-behavior-moments-ci.yml"

for path in [MODEL, TEST, WORKFLOW]:
    if not path.is_file():
        raise SystemExit(f"required Behavior Vision transport source missing: {path.relative_to(ROOT)}")

model = MODEL.read_text()
test = TEST.read_text()
workflow = WORKFLOW.read_text()

required_model = [
    "type BehaviorVisionFailureReason =",
    "'provider_http_error'",
    "'invalid_json'",
    "'timeout'",
    "'transport_error'",
    "headers.Authorization = `Bearer ${this.serviceToken}`",
    "this.warnFailure('provider_http_error', response.status)",
    "this.warnFailure('invalid_json')",
    "this.warnFailure('timeout')",
    "this.errorFailure('transport_error')",
    "Behavior vision provider failure reason=${reason}",
]
missing = [marker for marker in required_model if marker not in model]
if missing:
    raise SystemExit(f"Behavior Vision transport privacy contract is incomplete: {missing}")

for forbidden in [
    "response.text()",
    "await response.text",
    "error.message",
    "JSON.stringify(error)",
    "logger.warn(error",
    "logger.error(error",
]:
    if forbidden in model:
        raise SystemExit(
            f"apps/api/src/behavior-vision/behavior-vision.model.ts: forbidden private diagnostic logging marker {forbidden!r}"
        )

required_tests = [
    "never reads private provider error bodies into logs or the API error boundary",
    "fails closed on invalid JSON without logging provider response content",
    "classifies AbortError as a timeout without logging the underlying exception message",
    "classifies transport failures without logging arbitrary fetch exception details",
    "removes audio-derived evidence when audio analysis is disabled",
    "Authorization: `Bearer ${releaseConfig.BEHAVIOR_VISION_SERVICE_TOKEN}`",
]
missing = [marker for marker in required_tests if marker not in test]
if missing:
    raise SystemExit(f"Behavior Vision transport privacy tests are incomplete: {missing}")

required_workflow = [
    ".github/scripts/assert-behavior-vision-transport.py",
    "python -m py_compile .github/scripts/assert-behavior-vision-transport.py",
    "python .github/scripts/assert-behavior-vision-transport.py",
]
missing = [marker for marker in required_workflow if marker not in workflow]
if missing:
    raise SystemExit(f"Behavior Moments CI does not own transport privacy regression: {missing}")

print(
    "Behavior Vision transport contract preserves bearer authentication, bounded failure classes, "
    "private provider-body suppression, and audio-disabled evidence filtering."
)
