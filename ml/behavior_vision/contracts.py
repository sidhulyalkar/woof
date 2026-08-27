"""Canonical data contracts for the Behavior Vision worker.

This module intentionally uses only the Python standard library so contract tests can run in
lightweight CI without downloading any vision checkpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SCHEMA_VERSION = "woof-behavior-observation-v1"
FEATURE_VERSION = "behavior-evidence-fusion-v1"

DIMENSIONS = {
    "arousal",
    "body-tension",
    "social-orientation",
    "approach-tendency",
    "avoidance-tendency",
    "handler-engagement",
    "environment-engagement",
    "recovery",
}

SOURCES = {"pose", "motion", "face", "audio", "interaction", "context", "owner"}

HYPOTHESES = {
    "social-approach-with-arousal",
    "barrier-frustration-compatible-pattern",
    "avoidance-or-conflict-compatible-pattern",
    "play-compatible-pattern",
    "overarousal-compatible-pattern",
    "settled-observation",
    "insufficient-evidence",
}


class ContractError(ValueError):
    """Raised when an adapter or request violates the canonical behavior contract."""


def clamp01(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"expected numeric probability/value, got {value!r}") from exc
    return max(0.0, min(1.0, numeric))


@dataclass(frozen=True)
class Evidence:
    label: str
    source: Literal["pose", "motion", "face", "audio", "interaction", "context", "owner"]
    confidence: float
    start_ms: int | None = None
    end_ms: int | None = None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ContractError("evidence label must be non-empty")
        if self.source not in SOURCES:
            raise ContractError(f"unsupported evidence source: {self.source}")
        object.__setattr__(self, "confidence", clamp01(self.confidence))
        if self.start_ms is not None and self.start_ms < 0:
            raise ContractError("start_ms must be non-negative")
        if self.end_ms is not None and self.end_ms < 0:
            raise ContractError("end_ms must be non-negative")
        if (
            self.start_ms is not None
            and self.end_ms is not None
            and self.end_ms < self.start_ms
        ):
            raise ContractError("end_ms must not precede start_ms")

    def to_api(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label.strip(),
            "source": self.source,
            "confidence": self.confidence,
        }
        if self.start_ms is not None:
            payload["startMs"] = self.start_ms
        if self.end_ms is not None:
            payload["endMs"] = self.end_ms
        return payload


@dataclass(frozen=True)
class DimensionEstimate:
    dimension: str
    value: float
    confidence: float
    basis: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dimension not in DIMENSIONS:
            raise ContractError(f"unsupported behavior dimension: {self.dimension}")
        object.__setattr__(self, "value", clamp01(self.value))
        object.__setattr__(self, "confidence", clamp01(self.confidence))
        object.__setattr__(self, "basis", tuple(item for item in self.basis if item)[:8])

    def to_api(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "value": self.value,
            "confidence": self.confidence,
            "basis": list(self.basis),
        }


@dataclass(frozen=True)
class Hypothesis:
    id: str
    confidence: float
    statement: str
    supporting_evidence: tuple[str, ...] = ()
    contradictory_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.id not in HYPOTHESES:
            raise ContractError(f"unsupported behavior hypothesis: {self.id}")
        object.__setattr__(self, "confidence", clamp01(self.confidence))
        if not self.statement.strip():
            raise ContractError("hypothesis statement must be non-empty")
        object.__setattr__(
            self,
            "supporting_evidence",
            tuple(item for item in self.supporting_evidence if item)[:8],
        )
        object.__setattr__(
            self,
            "contradictory_evidence",
            tuple(item for item in self.contradictory_evidence if item)[:8],
        )

    def to_api(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "confidence": self.confidence,
            "statement": self.statement.strip(),
            "supportingEvidence": list(self.supporting_evidence),
            "contradictoryEvidence": list(self.contradictory_evidence),
        }


@dataclass(frozen=True)
class AdapterObservation:
    """One perception adapter's evidence contribution.

    Adapters may emit evidence and calibrated dimension estimates. They may suggest a broad
    behavior-compatible hypothesis for auditability, but they never emit coaching actions.
    """

    adapter_id: str
    model_version: str
    evidence: tuple[Evidence, ...] = ()
    dimensions: tuple[DimensionEstimate, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    quality: float = 1.0
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.adapter_id.strip() or not self.model_version.strip():
            raise ContractError("adapter_id and model_version must be non-empty")
        object.__setattr__(self, "quality", clamp01(self.quality))
        object.__setattr__(self, "warnings", tuple(item for item in self.warnings if item)[:8])


@dataclass(frozen=True)
class MediaQuality:
    usable: bool
    confidence: float
    issues: tuple[str, ...] = ()
    recapture_instructions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", clamp01(self.confidence))
        object.__setattr__(self, "issues", tuple(item for item in self.issues if item)[:8])
        object.__setattr__(
            self,
            "recapture_instructions",
            tuple(item for item in self.recapture_instructions if item)[:6],
        )

    def to_api(self) -> dict[str, Any]:
        return {
            "usable": self.usable,
            "confidence": self.confidence,
            "issues": list(self.issues),
            "recaptureInstructions": list(self.recapture_instructions),
        }


@dataclass(frozen=True)
class CanonicalAnalysis:
    model_version: str
    media_quality: MediaQuality
    evidence: tuple[Evidence, ...]
    dimensions: tuple[DimensionEstimate, ...]
    hypotheses: tuple[Hypothesis, ...]
    observable_summary: str
    uncertainty: str
    runtime_provenance: dict[str, Any] = field(default_factory=dict)
    feature_version: str = FEATURE_VERSION
    schema_version: str = SCHEMA_VERSION

    def to_api(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "modelVersion": self.model_version,
            "featureVersion": self.feature_version,
            "runtimeProvenance": self.runtime_provenance,
            "mediaQuality": self.media_quality.to_api(),
            "evidence": [entry.to_api() for entry in self.evidence[:40]],
            "dimensions": [entry.to_api() for entry in self.dimensions],
            "hypotheses": [entry.to_api() for entry in self.hypotheses[:6]],
            "observableSummary": self.observable_summary,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True)
class RequestMetadata:
    pet: dict[str, Any]
    context: dict[str, Any]
    question: str | None = None
    prior_profile_summary: dict[str, Any] | None = None
    policy: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "RequestMetadata":
        if payload.get("schemaVersion") != SCHEMA_VERSION:
            raise ContractError("request schemaVersion does not match Behavior Vision contract")
        pet = payload.get("pet")
        context = payload.get("context")
        policy = payload.get("policy") or {}
        if not isinstance(pet, dict) or not isinstance(context, dict) or not isinstance(policy, dict):
            raise ContractError("pet, context, and policy must be objects")

        required_policy = {
            "objectiveObservationOnly": True,
            "noDefinitiveEmotionInference": True,
            "noAutomaticGreetingRecommendation": True,
            "noHumanFaceRecognition": True,
            "noBiometricIdentityInference": True,
        }
        for key, expected in required_policy.items():
            if policy.get(key) is not expected:
                raise ContractError(f"request must enforce policy {key}={expected}")

        audio_policy = policy.get("audioAnalysisAllowed")
        context_audio = context.get("audioAnalysisAllowed")
        if not isinstance(audio_policy, bool) or audio_policy is not context_audio:
            raise ContractError(
                "request audioAnalysisAllowed policy must match the observation context"
            )

        question = payload.get("question")
        if question is not None and not isinstance(question, str):
            raise ContractError("question must be a string or null")
        prior = payload.get("priorProfileSummary")
        if prior is not None and not isinstance(prior, dict):
            raise ContractError("priorProfileSummary must be an object or null")

        return cls(
            pet=pet,
            context=context,
            question=question,
            prior_profile_summary=prior,
            policy=policy,
        )
