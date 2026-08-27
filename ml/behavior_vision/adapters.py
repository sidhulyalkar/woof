"""Pluggable perception adapters for Woof Behavior Vision.

Heavy vision models are optional runtime dependencies. This module keeps model-specific code behind a
small evidence interface so no tracker, pose network, or video encoder becomes the product's policy
layer. Production adapters should live in their own modules and be explicitly enabled by import path.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Protocol

from .contracts import AdapterObservation, RequestMetadata
from .registry import BehaviorModelRegistry


@dataclass(frozen=True)
class MediaInput:
    bytes: bytes
    mime_type: str
    filename: str


class BehaviorAdapter(Protocol):
    adapter_id: str

    def analyze(self, media: MediaInput, metadata: RequestMetadata) -> AdapterObservation:
        """Return objective evidence/dimensions. Must not return training advice."""


def load_adapter(import_path: str) -> BehaviorAdapter:
    """Load `package.module:factory_or_instance` lazily.

    Factories are called with no arguments. This keeps optional heavyweight imports out of the base
    worker and makes deployment manifests explicit about exactly which adapters are active.
    """

    if ":" not in import_path:
        raise ValueError(f"adapter path must look like package.module:object, got {import_path!r}")
    module_name, object_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    candidate = getattr(module, object_name)
    instance = candidate() if callable(candidate) and not hasattr(candidate, "analyze") else candidate
    if not hasattr(instance, "analyze") or not getattr(instance, "adapter_id", ""):
        raise TypeError(f"{import_path} does not implement the BehaviorAdapter contract")
    return instance


def parse_configured_adapter_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(
            "Behavior Vision adapters must be configured as registry-id=package.module:object"
        )
    adapter_id, import_path = (part.strip() for part in spec.split("=", 1))
    if not adapter_id or not import_path:
        raise ValueError(
            "Behavior Vision adapters must include both registry id and import path"
        )
    return adapter_id, import_path


def load_configured_adapters(
    registry: BehaviorModelRegistry | None = None,
) -> list[BehaviorAdapter]:
    raw = os.getenv("WOOF_BEHAVIOR_ADAPTERS", "").strip()
    if not raw:
        return []
    authority = registry or BehaviorModelRegistry.load()
    adapters: list[BehaviorAdapter] = []
    configured_ids: list[str] = []
    for item in raw.split(","):
        spec = item.strip()
        if not spec:
            continue
        adapter_id, import_path = parse_configured_adapter_spec(spec)
        authority.assert_primary_runtime_adapter(adapter_id)
        adapter = load_adapter(import_path)
        if adapter.adapter_id != adapter_id:
            raise TypeError(
                f"configured Behavior Vision adapter {adapter_id} identified itself as {adapter.adapter_id}"
            )
        adapters.append(adapter)
        configured_ids.append(adapter_id)
    authority.validate_configured_adapter_ids(configured_ids)
    return adapters
