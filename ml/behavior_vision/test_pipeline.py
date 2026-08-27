from __future__ import annotations

import unittest

from ml.behavior_vision.adapters import MediaInput
from ml.behavior_vision.contracts import (
    AdapterObservation,
    DimensionEstimate,
    Evidence,
    RequestMetadata,
    SCHEMA_VERSION,
)
from ml.behavior_vision.pipeline import BehaviorVisionPipeline


class FakePoseAdapter:
    adapter_id = "fake-pose"

    def analyze(self, media: MediaInput, metadata: RequestMetadata) -> AdapterObservation:
        return AdapterObservation(
            adapter_id=self.adapter_id,
            model_version="1",
            quality=0.9,
            evidence=(
                Evidence(label="body oriented toward other dog", source="pose", confidence=0.9),
            ),
            dimensions=(
                DimensionEstimate(
                    dimension="social-orientation",
                    value=0.85,
                    confidence=0.9,
                    basis=("relative body orientation",),
                ),
                DimensionEstimate(
                    dimension="approach-tendency",
                    value=0.75,
                    confidence=0.85,
                    basis=("forward displacement",),
                ),
            ),
        )


class FakeMotionAdapter:
    adapter_id = "fake-motion"

    def analyze(self, media: MediaInput, metadata: RequestMetadata) -> AdapterObservation:
        return AdapterObservation(
            adapter_id=self.adapter_id,
            model_version="2",
            quality=0.9,
            evidence=(
                Evidence(label="repeated forward movement", source="motion", confidence=0.85),
            ),
            dimensions=(
                DimensionEstimate(
                    dimension="arousal",
                    value=0.82,
                    confidence=0.85,
                    basis=("movement rate",),
                ),
                DimensionEstimate(
                    dimension="body-tension",
                    value=0.55,
                    confidence=0.7,
                    basis=("pose dynamics",),
                ),
            ),
        )


def metadata(other_dogs: bool = True) -> RequestMetadata:
    return RequestMetadata.from_json(
        {
            "schemaVersion": SCHEMA_VERSION,
            "pet": {"name": "Nova", "species": "DOG"},
            "context": {
                "context": "street",
                "otherDogsPresent": other_dogs,
                "audioAnalysisAllowed": False,
            },
            "question": None,
            "priorProfileSummary": None,
            "policy": {
                "objectiveObservationOnly": True,
                "noDefinitiveEmotionInference": True,
                "noAutomaticGreetingRecommendation": True,
                "noHumanFaceRecognition": True,
                "noBiometricIdentityInference": True,
                "audioAnalysisAllowed": False,
            },
        }
    )


class BehaviorVisionPipelineTest(unittest.TestCase):
    def test_no_adapters_abstains(self) -> None:
        result = BehaviorVisionPipeline(adapters=[]).analyze(
            MediaInput(bytes=b"video", mime_type="video/webm", filename="dog.webm"),
            metadata(),
        )
        self.assertFalse(result.media_quality.usable)
        self.assertEqual(result.hypotheses[0].id, "insufficient-evidence")
        self.assertEqual(result.dimensions, ())

    def test_social_orientation_is_hypothesis_not_greeting_advice(self) -> None:
        result = BehaviorVisionPipeline(adapters=[FakePoseAdapter(), FakeMotionAdapter()]).analyze(
            MediaInput(bytes=b"video", mime_type="video/webm", filename="dog.webm"),
            metadata(),
        )
        ids = {hypothesis.id for hypothesis in result.hypotheses}
        self.assertIn("social-approach-with-arousal", ids)
        self.assertIn("barrier-frustration-compatible-pattern", ids)
        serialized = result.to_api()
        serialized_text = str(serialized).lower()
        self.assertNotIn("let them greet", serialized_text)
        self.assertNotIn("needs to greet", serialized_text)
        self.assertIn("cannot distinguish", serialized_text)

    def test_policy_flags_are_required(self) -> None:
        with self.assertRaises(ValueError):
            RequestMetadata.from_json(
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "pet": {"name": "Nova"},
                    "context": {},
                    "policy": {"objectiveObservationOnly": True},
                }
            )


    def test_runtime_provenance_is_emitted_without_claiming_checkpoint_attestation(self) -> None:
        result = BehaviorVisionPipeline(adapters=[FakePoseAdapter()]).analyze(
            MediaInput(bytes=b"video", mime_type="video/webm", filename="dog.webm"),
            metadata(),
        )
        provenance = result.to_api()["runtimeProvenance"]
        self.assertEqual(provenance["artifactAttestation"], "not-available")
        self.assertEqual(provenance["policyVersion"], "behavior-runtime-provenance-v1")
        self.assertEqual(provenance["contributingAdapters"][0]["id"], "fake-pose")

    def test_audio_policy_must_match_context(self) -> None:
        with self.assertRaises(ValueError):
            RequestMetadata.from_json(
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "pet": {"name": "Nova"},
                    "context": {"audioAnalysisAllowed": False},
                    "policy": {
                        "objectiveObservationOnly": True,
                        "noDefinitiveEmotionInference": True,
                        "noAutomaticGreetingRecommendation": True,
                        "noHumanFaceRecognition": True,
                        "noBiometricIdentityInference": True,
                        "audioAnalysisAllowed": True,
                    },
                }
            )

    def test_adapter_failure_isolated(self) -> None:
        class BrokenAdapter:
            adapter_id = "broken"

            def analyze(self, media: MediaInput, metadata: RequestMetadata) -> AdapterObservation:
                raise RuntimeError("boom")

        result = BehaviorVisionPipeline(adapters=[BrokenAdapter(), FakePoseAdapter()]).analyze(
            MediaInput(bytes=b"video", mime_type="video/webm", filename="dog.webm"),
            metadata(),
        )
        self.assertTrue(result.media_quality.usable)
        self.assertTrue(any("broken" in issue for issue in result.media_quality.issues))


if __name__ == "__main__":
    unittest.main()
