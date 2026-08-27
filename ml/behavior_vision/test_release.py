from __future__ import annotations

import unittest

from ml.behavior_vision.contracts import (
    CanonicalAnalysis,
    ContractError,
    Hypothesis,
    MediaQuality,
    ReleaseIdentity,
    RequestMetadata,
    SCHEMA_VERSION,
)
from ml.behavior_vision.release import load_release_identity, require_matching_release

SHA = "a" * 64


def release(**overrides: str) -> ReleaseIdentity:
    values = {
        "release_id": "behavior-shadow-2026-08-27",
        "model_version": "shadow-model-1",
        "feature_version": "features-1",
        "artifact_sha256": SHA,
    }
    values.update(overrides)
    return ReleaseIdentity(**values)


class BehaviorVisionReleaseTest(unittest.TestCase):
    def test_worker_loads_complete_release_from_its_own_environment(self) -> None:
        identity = load_release_identity(
            {
                "WOOF_BEHAVIOR_RELEASE_ID": "behavior-shadow-2026-08-27",
                "WOOF_BEHAVIOR_MODEL_VERSION": "shadow-model-1",
                "WOOF_BEHAVIOR_FEATURE_VERSION": "features-1",
                "WOOF_BEHAVIOR_ARTIFACT_SHA256": SHA.upper(),
            }
        )
        self.assertEqual(identity, release())

    def test_worker_rejects_partial_release_configuration(self) -> None:
        with self.assertRaises(ContractError):
            load_release_identity(
                {
                    "WOOF_BEHAVIOR_RELEASE_ID": "behavior-shadow-2026-08-27",
                    "WOOF_BEHAVIOR_MODEL_VERSION": "shadow-model-1",
                }
            )

    def test_worker_requires_request_and_deployment_release_to_match(self) -> None:
        actual = release()
        self.assertEqual(require_matching_release(release(), actual), actual)

        with self.assertRaises(ContractError):
            require_matching_release(
                release(artifact_sha256="b" * 64),
                actual,
            )
        with self.assertRaises(ContractError):
            require_matching_release(None, actual)
        with self.assertRaises(ContractError):
            require_matching_release(actual, None)

    def test_release_identity_rejects_malformed_artifact_hash(self) -> None:
        with self.assertRaises(ContractError):
            release(artifact_sha256="not-a-sha")

    def test_request_parser_rejects_non_string_release_metadata_cleanly(self) -> None:
        with self.assertRaises(ContractError):
            RequestMetadata.from_json(
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "pet": {"name": "Nova", "species": "DOG"},
                    "context": {"context": "street"},
                    "policy": {
                        "objectiveObservationOnly": True,
                        "noDefinitiveEmotionInference": True,
                        "noAutomaticGreetingRecommendation": True,
                    },
                    "expectedRelease": {
                        "releaseId": "behavior-shadow-2026-08-27",
                        "modelVersion": "shadow-model-1",
                        "featureVersion": "features-1",
                        "artifactSha256": 12345,
                        "responseContract": SCHEMA_VERSION,
                    },
                }
            )

    def test_http_serialization_uses_worker_release_not_adapter_diagnostic_version(self) -> None:
        analysis = CanonicalAnalysis(
            model_version="fake-pose@1+fake-motion@2",
            feature_version="behavior-evidence-fusion-v1",
            media_quality=MediaQuality(usable=False, confidence=0),
            evidence=(),
            dimensions=(),
            hypotheses=(
                Hypothesis(
                    id="insufficient-evidence",
                    confidence=1,
                    statement="Not enough reliable evidence.",
                ),
            ),
            observable_summary="No reliable observation.",
            uncertainty="Abstained.",
        )

        payload = analysis.to_api(release())

        self.assertEqual(payload["schemaVersion"], SCHEMA_VERSION)
        self.assertEqual(payload["releaseId"], "behavior-shadow-2026-08-27")
        self.assertEqual(payload["modelVersion"], "shadow-model-1")
        self.assertEqual(payload["featureVersion"], "features-1")
        self.assertEqual(payload["artifactSha256"], SHA)
        self.assertNotEqual(payload["modelVersion"], analysis.model_version)


if __name__ == "__main__":
    unittest.main()
