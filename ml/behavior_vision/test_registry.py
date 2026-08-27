from __future__ import annotations

import unittest

from ml.behavior_vision.registry import (
    ARTIFACT_ATTESTATION_STATUS,
    REGISTRY_HASH_ALGORITHM,
    RUNTIME_POLICY_VERSION,
    BehaviorModelRegistry,
    RegistryError,
)

EXPECTED_REGISTRY_SHA256 = "ebaaac0c99b67dcfd81787a30fc88d03d762e580e5bf253699fcf47479fef796"


class BehaviorModelRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = BehaviorModelRegistry.load()

    def test_checked_in_registry_has_release_pinned_canonical_identity(self) -> None:
        self.assertEqual(self.registry.sha256, EXPECTED_REGISTRY_SHA256)
        self.assertEqual(self.registry.payload["runtimePolicyVersion"], RUNTIME_POLICY_VERSION)

    def test_primary_runtime_eligibility_is_explicit(self) -> None:
        self.assertTrue(
            self.registry.assert_primary_runtime_adapter("sam2-video-tracking").primary_runtime_eligible
        )
        self.assertTrue(
            self.registry.assert_primary_runtime_adapter("vitposepp-ap10k").primary_runtime_eligible
        )
        for component_id in (
            "sleap-dog-adapter",
            "animal-clip",
            "ethoclip-animalband",
            "dogfacs-ontology",
            "woof-individual-adapter",
        ):
            with self.assertRaises(RegistryError):
                self.registry.assert_primary_runtime_adapter(component_id)

    def test_unknown_and_duplicate_runtime_adapters_fail_closed(self) -> None:
        with self.assertRaises(RegistryError):
            self.registry.validate_configured_adapter_ids(["mystery-adapter"])
        with self.assertRaises(RegistryError):
            self.registry.validate_configured_adapter_ids(
                ["sam2-video-tracking", "sam2-video-tracking"]
            )

    def test_runtime_provenance_reports_registry_not_checkpoint_attestation(self) -> None:
        provenance = self.registry.runtime_provenance(
            ["sam2-video-tracking"],
            [("sam2-video-tracking", "sam2-test-version")],
        )

        self.assertEqual(provenance["registrySha256"], EXPECTED_REGISTRY_SHA256)
        self.assertEqual(provenance["registryHashAlgorithm"], REGISTRY_HASH_ALGORITHM)
        self.assertEqual(provenance["policyVersion"], RUNTIME_POLICY_VERSION)
        self.assertEqual(provenance["artifactAttestation"], ARTIFACT_ATTESTATION_STATUS)
        self.assertEqual(
            provenance["contributingAdapters"],
            [
                {
                    "id": "sam2-video-tracking",
                    "status": "integration-candidate",
                    "behaviorAuthority": False,
                    "modelVersion": "sam2-test-version",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
