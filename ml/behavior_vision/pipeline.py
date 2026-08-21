"""Evidence fusion for Woof Behavior Vision.

The pipeline does not infer hidden emotion. It aggregates calibrated measurements from independent
perception adapters and emits behavior-compatible hypotheses only when multiple observable signals
support them. The NestJS layer separately owns longitudinal N-of-1 personalization and coaching.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Iterable

from .adapters import BehaviorAdapter, MediaInput, load_configured_adapters
from .contracts import (
    AdapterObservation,
    CanonicalAnalysis,
    DimensionEstimate,
    Evidence,
    Hypothesis,
    MediaQuality,
    RequestMetadata,
    clamp01,
)


class BehaviorVisionPipeline:
    def __init__(self, adapters: Iterable[BehaviorAdapter] | None = None) -> None:
        self.adapters = list(adapters) if adapters is not None else load_configured_adapters()

    def analyze(self, media: MediaInput, metadata: RequestMetadata) -> CanonicalAnalysis:
        if not media.bytes:
            return self._insufficient("empty media payload", "Capture or upload a non-empty clip.")
        if not self.adapters:
            return self._insufficient(
                "no vetted perception adapters are enabled",
                "This deployment has not enabled the specialized pose/action model stack.",
            )

        observations: list[AdapterObservation] = []
        adapter_failures: list[str] = []
        for adapter in self.adapters:
            try:
                observation = adapter.analyze(media, metadata)
            except Exception as exc:  # adapter isolation is intentional
                adapter_failures.append(f"{getattr(adapter, 'adapter_id', 'adapter')}: {type(exc).__name__}")
                continue
            if observation.quality <= 0:
                continue
            observations.append(observation)

        if not observations:
            reason = "all enabled perception adapters abstained or failed"
            if adapter_failures:
                reason += f" ({', '.join(adapter_failures[:4])})"
            return self._insufficient(reason, "Try a shorter, brighter clip with the full dog visible.")

        evidence = self._fuse_evidence(observations)
        dimensions = self._fuse_dimensions(observations)
        hypotheses = self._derive_hypotheses(evidence, dimensions, metadata)
        model_version = "+".join(
            sorted({f"{entry.adapter_id}@{entry.model_version}" for entry in observations})
        )[:240]
        quality_values = [entry.quality for entry in observations]
        quality = clamp01(fmean(quality_values) if quality_values else 0)
        warnings = [warning for entry in observations for warning in entry.warnings]
        issues = tuple(dict.fromkeys([*warnings, *adapter_failures]))[:8]
        usable = quality >= 0.35 and bool(evidence or dimensions)

        if not usable:
            return CanonicalAnalysis(
                model_version=model_version or "behavior-evidence-fusion-v1",
                media_quality=MediaQuality(
                    usable=False,
                    confidence=quality,
                    issues=issues or ("insufficient calibrated observable evidence",),
                    recapture_instructions=(
                        "Keep the full dog visible for 10–20 seconds.",
                        "Include the nearby environment and leash when relevant.",
                        "Avoid filming directly into bright backlight.",
                    ),
                ),
                evidence=tuple(evidence),
                dimensions=tuple(dimensions),
                hypotheses=(
                    Hypothesis(
                        id="insufficient-evidence",
                        confidence=1.0,
                        statement="The available model evidence is not reliable enough for behavior coaching.",
                    ),
                ),
                observable_summary="The clip did not produce enough reliable objective evidence.",
                uncertainty="Woof abstained rather than inferring behavior from weak model signals.",
            )

        summary = self._observable_summary(evidence, dimensions)
        return CanonicalAnalysis(
            model_version=model_version or "behavior-evidence-fusion-v1",
            media_quality=MediaQuality(
                usable=True,
                confidence=quality,
                issues=issues,
                recapture_instructions=(),
            ),
            evidence=tuple(evidence),
            dimensions=tuple(dimensions),
            hypotheses=tuple(hypotheses),
            observable_summary=summary,
            uncertainty=(
                "Dimensions summarize observable movement/posture evidence. They do not reveal the dog’s "
                "internal emotion or prove social intent."
            ),
        )

    def _fuse_evidence(self, observations: list[AdapterObservation]) -> list[Evidence]:
        best: dict[tuple[str, str, int | None, int | None], Evidence] = {}
        for observation in observations:
            for item in observation.evidence:
                confidence = clamp01(item.confidence * observation.quality)
                fused = Evidence(
                    label=item.label,
                    source=item.source,
                    confidence=confidence,
                    start_ms=item.start_ms,
                    end_ms=item.end_ms,
                )
                key = (fused.label, fused.source, fused.start_ms, fused.end_ms)
                if key not in best or fused.confidence > best[key].confidence:
                    best[key] = fused
        return sorted(best.values(), key=lambda item: item.confidence, reverse=True)[:40]

    def _fuse_dimensions(self, observations: list[AdapterObservation]) -> list[DimensionEstimate]:
        values: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
        for observation in observations:
            for estimate in observation.dimensions:
                weight = clamp01(estimate.confidence * observation.quality)
                if weight <= 0:
                    continue
                values[estimate.dimension].append(
                    (estimate.value, weight, f"{observation.adapter_id}:{','.join(estimate.basis)}")
                )

        fused: list[DimensionEstimate] = []
        for dimension, entries in values.items():
            total_weight = sum(weight for _, weight, _ in entries)
            if total_weight <= 0:
                continue
            mean = sum(value * weight for value, weight, _ in entries) / total_weight
            agreement = 1.0
            if len(entries) > 1:
                spread = sum(abs(value - mean) * weight for value, weight, _ in entries) / total_weight
                agreement = max(0.25, 1.0 - 2.0 * spread)
            confidence = clamp01((total_weight / max(1, len(entries))) * agreement)
            basis = tuple(dict.fromkeys(source for _, _, source in entries))[:8]
            fused.append(
                DimensionEstimate(
                    dimension=dimension,
                    value=mean,
                    confidence=confidence,
                    basis=basis,
                )
            )

        return sorted(fused, key=lambda item: item.confidence, reverse=True)

    def _derive_hypotheses(
        self,
        evidence: list[Evidence],
        dimensions: list[DimensionEstimate],
        metadata: RequestMetadata,
    ) -> list[Hypothesis]:
        values = {item.dimension: item for item in dimensions if item.confidence >= 0.35}
        arousal = values.get("arousal")
        tension = values.get("body-tension")
        social = values.get("social-orientation")
        approach = values.get("approach-tendency")
        avoidance = values.get("avoidance-tendency")
        recovery = values.get("recovery")
        other_dogs = bool(metadata.context.get("otherDogsPresent"))
        hypotheses: list[Hypothesis] = []

        evidence_labels = [item.label for item in evidence if item.confidence >= 0.4]

        if (
            other_dogs
            and arousal
            and social
            and approach
            and arousal.value >= 0.6
            and social.value >= 0.6
            and approach.value >= 0.55
        ):
            confidence = min(arousal.confidence, social.confidence, approach.confidence)
            hypotheses.append(
                Hypothesis(
                    id="social-approach-with-arousal",
                    confidence=confidence,
                    statement=(
                        "The visible pattern is compatible with strong orientation/approach toward another dog "
                        "while activation is elevated."
                    ),
                    supporting_evidence=tuple(evidence_labels[:5]),
                    contradictory_evidence=(
                        "The clip alone cannot distinguish social excitement, frustration, uncertainty, fear, or mixed motivation.",
                    ),
                )
            )
            hypotheses.append(
                Hypothesis(
                    id="barrier-frustration-compatible-pattern",
                    confidence=confidence * 0.7,
                    statement=(
                        "If access or movement is being restricted by leash/barrier, this pattern can be compatible "
                        "with barrier frustration, but the video does not prove that explanation."
                    ),
                    supporting_evidence=tuple(evidence_labels[:5]),
                    contradictory_evidence=("Other motivations can produce similar visible behavior.",),
                )
            )

        if avoidance and tension and avoidance.value >= 0.55 and tension.value >= 0.55:
            hypotheses.append(
                Hypothesis(
                    id="avoidance-or-conflict-compatible-pattern",
                    confidence=min(avoidance.confidence, tension.confidence),
                    statement=(
                        "Retreat/avoidance-compatible movement and body tension are both elevated in the observable evidence."
                    ),
                    supporting_evidence=tuple(evidence_labels[:5]),
                    contradictory_evidence=(
                        "The cause of those movements cannot be determined from video alone.",
                    ),
                )
            )

        if arousal and tension and arousal.value >= 0.75 and tension.value >= 0.65:
            hypotheses.append(
                Hypothesis(
                    id="overarousal-compatible-pattern",
                    confidence=min(arousal.confidence, tension.confidence),
                    statement="The observable activation and body-tension dimensions are both high.",
                    supporting_evidence=tuple(evidence_labels[:5]),
                )
            )

        if arousal and tension and recovery and arousal.value <= 0.35 and tension.value <= 0.35:
            hypotheses.append(
                Hypothesis(
                    id="settled-observation",
                    confidence=min(arousal.confidence, tension.confidence, recovery.confidence),
                    statement="The clip contains relatively low activation/tension with evidence of recovery or settling.",
                    supporting_evidence=tuple(evidence_labels[:5]),
                )
            )

        if not hypotheses:
            hypotheses.append(
                Hypothesis(
                    id="insufficient-evidence",
                    confidence=0.65,
                    statement=(
                        "The objective signals do not support one sufficiently specific behavior-compatible pattern."
                    ),
                    supporting_evidence=tuple(evidence_labels[:4]),
                )
            )

        return sorted(hypotheses, key=lambda item: item.confidence, reverse=True)[:6]

    def _observable_summary(
        self, evidence: list[Evidence], dimensions: list[DimensionEstimate]
    ) -> str:
        reliable_dimensions = [item for item in dimensions if item.confidence >= 0.45][:4]
        if reliable_dimensions:
            phrases = [
                f"{item.dimension.replace('-', ' ')} {item.value:.0%}"
                for item in reliable_dimensions
            ]
            return "Calibrated observable dimensions: " + ", ".join(phrases) + "."
        labels = [item.label for item in evidence if item.confidence >= 0.5][:4]
        if labels:
            return "Reliable visible/motion evidence includes: " + ", ".join(labels) + "."
        return "The clip contains limited reliable objective behavior evidence."

    def _insufficient(self, issue: str, instruction: str) -> CanonicalAnalysis:
        return CanonicalAnalysis(
            model_version="behavior-evidence-fusion-v1",
            media_quality=MediaQuality(
                usable=False,
                confidence=0.0,
                issues=(issue,),
                recapture_instructions=(instruction,),
            ),
            evidence=(),
            dimensions=(),
            hypotheses=(
                Hypothesis(
                    id="insufficient-evidence",
                    confidence=1.0,
                    statement="No reliable automated behavior observation was produced.",
                ),
            ),
            observable_summary="No reliable automated behavior observation was produced.",
            uncertainty="Woof abstained rather than inferring behavior without adequate evidence.",
        )
