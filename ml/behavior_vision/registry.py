"""Runtime authority for the checked-in Behavior Vision model registry.

This module intentionally does not claim checkpoint attestation. It proves which Woof registry and
runtime policy allowed a perception adapter to participate, while keeping exact model-byte
attestation as a separate future release boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REGISTRY_SCHEMA_VERSION = "woof-behavior-model-registry-v1"
RUNTIME_POLICY_VERSION = "behavior-runtime-provenance-v1"
REGISTRY_HASH_ALGORITHM = "sha256-canonical-json-v1"
ARTIFACT_ATTESTATION_STATUS = "not-available"
REGISTRY_PATH = Path(__file__).with_name("model_registry.json")


class RegistryError(RuntimeError):
    """Raised when runtime configuration violates the checked-in registry contract."""


@dataclass(frozen=True)
class RegistryComponent:
    id: str
    status: str
    behavior_authority: bool
    primary_runtime_eligible: bool

    def configured_api(self) -> dict[str, object]:
        return {
            "id": self.id,
            "status": self.status,
            "behaviorAuthority": self.behavior_authority,
        }


class BehaviorModelRegistry:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self._validate_root()
        self.components = self._load_components()
        self.sha256 = self._canonical_sha256(payload)

    @classmethod
    def load(cls, path: Path = REGISTRY_PATH) -> "BehaviorModelRegistry":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError("Behavior Vision model registry is unreadable") from exc
        if not isinstance(payload, dict):
            raise RegistryError("Behavior Vision model registry must be a JSON object")
        return cls(payload)

    def _validate_root(self) -> None:
        if self.payload.get("schemaVersion") != REGISTRY_SCHEMA_VERSION:
            raise RegistryError("Behavior Vision registry schemaVersion is not supported")
        if self.payload.get("runtimePolicyVersion") != RUNTIME_POLICY_VERSION:
            raise RegistryError("Behavior Vision runtime policy version is not supported")

        policy = self.payload.get("policy")
        if not isinstance(policy, dict):
            raise RegistryError("Behavior Vision registry policy must be an object")
        required_policy = {
            "authoritativeEmotionInference": False,
            "automaticDogGreetingRecommendation": False,
            "rawUserMediaTrainingDefault": False,
            "requiresOptInForUserMediaTraining": True,
            "productionPromotionRequiresCalibratedDogSpecificHoldout": True,
        }
        for key, expected in required_policy.items():
            if policy.get(key) is not expected:
                raise RegistryError(f"Behavior Vision registry must enforce policy {key}={expected}")

    def _load_components(self) -> dict[str, RegistryComponent]:
        raw_components = self.payload.get("components")
        if not isinstance(raw_components, list):
            raise RegistryError("Behavior Vision registry components must be an array")

        components: dict[str, RegistryComponent] = {}
        for raw in raw_components:
            if not isinstance(raw, dict):
                raise RegistryError("Behavior Vision registry component must be an object")
            component_id = raw.get("id")
            status = raw.get("status")
            behavior_authority = raw.get("behaviorAuthority")
            primary_runtime_eligible = raw.get("primaryRuntimeEligible")
            if not isinstance(component_id, str) or not component_id.strip():
                raise RegistryError("Behavior Vision registry component id must be non-empty")
            if component_id in components:
                raise RegistryError(f"Duplicate Behavior Vision registry component: {component_id}")
            if not isinstance(status, str) or not status.strip():
                raise RegistryError(f"Behavior Vision component {component_id} has no status")
            if behavior_authority is not False:
                raise RegistryError(
                    f"Behavior Vision component {component_id} must not hold behavior authority"
                )
            if not isinstance(primary_runtime_eligible, bool):
                raise RegistryError(
                    f"Behavior Vision component {component_id} must declare primaryRuntimeEligible"
                )
            components[component_id] = RegistryComponent(
                id=component_id,
                status=status,
                behavior_authority=False,
                primary_runtime_eligible=primary_runtime_eligible,
            )
        return components

    def assert_primary_runtime_adapter(self, adapter_id: str) -> RegistryComponent:
        component = self.components.get(adapter_id)
        if component is None:
            raise RegistryError(f"Unregistered Behavior Vision adapter: {adapter_id}")
        if not component.primary_runtime_eligible:
            raise RegistryError(
                f"Behavior Vision adapter {adapter_id} is not eligible for the primary runtime"
            )
        return component

    def validate_configured_adapter_ids(self, adapter_ids: Iterable[str]) -> tuple[str, ...]:
        ids = tuple(adapter_ids)
        if len(ids) != len(set(ids)):
            raise RegistryError("Behavior Vision runtime cannot configure duplicate adapter ids")
        for adapter_id in ids:
            self.assert_primary_runtime_adapter(adapter_id)
        return ids

    def runtime_provenance(
        self,
        configured_adapter_ids: Iterable[str],
        contributors: Iterable[tuple[str, str]],
        *,
        enforce_registry: bool = True,
    ) -> dict[str, object]:
        configured_ids = tuple(configured_adapter_ids)
        contributor_pairs = tuple(contributors)
        if enforce_registry:
            self.validate_configured_adapter_ids(configured_ids)
            contributor_ids = tuple(adapter_id for adapter_id, _ in contributor_pairs)
            if len(contributor_ids) != len(set(contributor_ids)):
                raise RegistryError("Behavior Vision analysis contains duplicate contributing adapters")
            if not set(contributor_ids).issubset(set(configured_ids)):
                raise RegistryError("Behavior Vision contributor was not configured for this runtime")

        configured: list[dict[str, object]] = []
        for adapter_id in configured_ids:
            component = self.components.get(adapter_id)
            configured.append(
                component.configured_api()
                if component is not None
                else {
                    "id": adapter_id,
                    "status": "test-only-unregistered",
                    "behaviorAuthority": False,
                }
            )

        contributing: list[dict[str, object]] = []
        for adapter_id, model_version in contributor_pairs:
            component = self.components.get(adapter_id)
            contributing.append(
                {
                    "id": adapter_id,
                    "status": component.status if component is not None else "test-only-unregistered",
                    "behaviorAuthority": False,
                    "modelVersion": model_version,
                }
            )

        return {
            "policyVersion": RUNTIME_POLICY_VERSION,
            "registrySchemaVersion": REGISTRY_SCHEMA_VERSION,
            "registrySha256": self.sha256,
            "registryHashAlgorithm": REGISTRY_HASH_ALGORITHM,
            "artifactAttestation": ARTIFACT_ATTESTATION_STATUS,
            "configuredAdapters": configured,
            "contributingAdapters": contributing,
        }

    @staticmethod
    def _canonical_sha256(payload: dict[str, Any]) -> str:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
