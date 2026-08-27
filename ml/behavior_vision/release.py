"""Deployment-owned release identity for the Behavior Vision worker.

The API sends the release it expects. The worker independently loads the release it actually serves
from its own environment and rejects disagreement. Request metadata is never treated as release
authority.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from .contracts import ContractError, ReleaseIdentity

RELEASE_ENV = {
    "release_id": "WOOF_BEHAVIOR_RELEASE_ID",
    "model_version": "WOOF_BEHAVIOR_MODEL_VERSION",
    "feature_version": "WOOF_BEHAVIOR_FEATURE_VERSION",
    "artifact_sha256": "WOOF_BEHAVIOR_ARTIFACT_SHA256",
}


def load_release_identity(
    environ: Mapping[str, str] | None = None,
) -> ReleaseIdentity | None:
    env = os.environ if environ is None else environ
    values = {field: (env.get(name) or "").strip() for field, name in RELEASE_ENV.items()}
    configured = [bool(value) for value in values.values()]
    if not any(configured):
        return None
    if not all(configured):
        missing = [RELEASE_ENV[field] for field, value in values.items() if not value]
        raise ContractError(
            "Behavior Vision worker release identity is incomplete; missing " + ", ".join(missing)
        )
    return ReleaseIdentity(**values)


def require_matching_release(
    expected: ReleaseIdentity | None,
    actual: ReleaseIdentity | None,
) -> ReleaseIdentity:
    if actual is None:
        raise ContractError("Behavior Vision worker has no configured release identity")
    if expected is None:
        raise ContractError("Behavior Vision request is missing expectedRelease")
    if expected != actual:
        raise ContractError("Behavior Vision expectedRelease does not match the deployed worker")
    return actual
