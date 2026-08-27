from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ml.common.attestation import sha256_file, verify_release_receipt
from ml.evaluation.attest_promotion import attest_promotion


def uncertainty_slice(cluster_column: str) -> dict:
    return {
        "available": True,
        "source": {
            "predictionSha256": "a" * 64,
            "clusterSourceSha256": "b" * 64,
            "clusterJoinKey": "outcome_id",
            "clusterLabelVerified": True,
            "clusterColumn": cluster_column,
        },
    }


class ReleaseAttestationTests(unittest.TestCase):
    def _candidate(
        self,
        root: Path,
        *,
        statistical_passed: bool = True,
        statistical_schema: str = "woof-model-promotion-receipt-v2",
        authoritative: bool = True,
        uncertainty_passed: bool = True,
        mutate_statistical=None,
    ):
        model = root / "compatibility_model.joblib"
        calibration = root / "calibration.joblib"
        contract = root / "feature_contract.json"
        manifest = root / "training_manifest.json"
        statistical = root / "statistical_receipt.json"

        model.write_bytes(b"exact-model-artifact-v1")
        calibration.write_bytes(b"exact-calibration-artifact-v1")
        contract.write_text(
            json.dumps(
                {
                    "schemaVersion": "woof-compatibility-feature-contract-v1",
                    "modelVersion": "canonical-test-v1",
                    "featureVersion": "compatibility-features-v1",
                    "orderedFeatureNames": ["energy_gap", "social_risk_max"],
                }
            ),
            encoding="utf-8",
        )
        manifest.write_text(
            json.dumps(
                {
                    "schemaVersion": "woof-canonical-model-training-v2",
                    "modelVersion": "canonical-test-v1",
                    "featureVersion": "compatibility-features-v1",
                    "modelSha256": sha256_file(model),
                    "calibration": {
                        "version": "isotonic-v1-canonical-test-v1",
                        "sha256": sha256_file(calibration),
                    },
                    "featureContract": {"sha256": sha256_file(contract)},
                    "releaseStatus": "shadow_candidate_only",
                }
            ),
            encoding="utf-8",
        )
        payload = {
            "schemaVersion": statistical_schema,
            "generatedAt": "2026-08-27T00:00:00+00:00",
            "decision": (
                "promote"
                if statistical_passed and authoritative
                else "hold_shadow" if not statistical_passed else "research_only"
            ),
            "passed": statistical_passed,
            "authoritativeEligible": authoritative,
            "safety": {"name": "safety", "available": False},
            "uncertaintyEvidence": {
                "required": True,
                "passed": uncertainty_passed,
                "schemaVersion": "woof-compatibility-uncertainty-v1",
                "policy": {
                    "pairedRows": True,
                    "relationshipClusterBootstrap": True,
                    "clusterIdentityFromEvaluationSplits": True,
                    "splitLabelAgreementRequired": True,
                    "confidenceLevel": 0.95,
                    "resamples": 4000,
                },
                "futureTest": uncertainty_slice("owner_pair_key"),
                "coldPair": uncertainty_slice("pair_key"),
                "coldOwner": uncertainty_slice("owner_pair_key"),
                "safety": {"name": "safety", "available": False},
            },
        }
        if mutate_statistical is not None:
            mutate_statistical(payload)
        statistical.write_text(json.dumps(payload), encoding="utf-8")
        return model, calibration, manifest, contract, statistical

    def _attest(self, root: Path, **candidate_kwargs):
        model, calibration, manifest, contract, statistical = self._candidate(
            root, **candidate_kwargs
        )
        receipt = attest_promotion(
            statistical_receipt_path=statistical,
            model_path=model,
            calibration_path=calibration,
            training_manifest_path=manifest,
            feature_contract_path=contract,
            signing_key="test-release-secret",
            key_id="test-key-v1",
        )
        return receipt, model, calibration, manifest, contract

    def test_signed_receipt_is_bound_to_exact_candidate_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt, model, calibration, manifest, contract = self._attest(root)

            self.assertTrue(receipt["passed"])
            self.assertEqual(receipt["decision"], "promote")
            self.assertEqual(receipt["releaseStatus"], "promoted")
            self.assertEqual(
                receipt["evidence"]["statisticalReceiptSchema"],
                "woof-model-promotion-receipt-v2",
            )
            self.assertEqual(
                receipt["evidence"]["uncertaintyReportSchema"],
                "woof-compatibility-uncertainty-v1",
            )
            valid, failures = verify_release_receipt(
                receipt,
                key="test-release-secret",
                model_path=model,
                calibration_path=calibration,
                training_manifest_path=manifest,
                feature_contract_path=contract,
            )
            self.assertTrue(valid)
            self.assertEqual(failures, [])

    def test_legacy_point_estimate_receipt_cannot_authorize_release(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, *_ = self._attest(
                Path(directory),
                statistical_schema="woof-model-promotion-receipt-v1",
            )
            self.assertFalse(receipt["passed"])
            self.assertEqual(receipt["releaseStatus"], "shadow")
            self.assertIn(
                "statistical_receipt_schema_not_authoritative",
                receipt["failures"],
            )

    def test_research_only_receipt_cannot_authorize_release(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, *_ = self._attest(Path(directory), authoritative=False)
            self.assertFalse(receipt["passed"])
            self.assertIn("statistical_gate_not_passed", receipt["failures"])
            self.assertIn("statistical_receipt_not_authoritative", receipt["failures"])

    def test_failed_uncertainty_gate_cannot_authorize_release(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, *_ = self._attest(Path(directory), uncertainty_passed=False)
            self.assertFalse(receipt["passed"])
            self.assertIn("uncertainty_gate_not_passed", receipt["failures"])

    def test_missing_split_provenance_cannot_authorize_release(self):
        with tempfile.TemporaryDirectory() as directory:
            def mutate(payload):
                del payload["uncertaintyEvidence"]["futureTest"]["source"][
                    "clusterSourceSha256"
                ]

            receipt, *_ = self._attest(Path(directory), mutate_statistical=mutate)
            self.assertFalse(receipt["passed"])
            self.assertIn("futureTest_cluster_source_hash_missing", receipt["failures"])

    def test_self_declared_cluster_policy_cannot_authorize_release(self):
        with tempfile.TemporaryDirectory() as directory:
            def mutate(payload):
                payload["uncertaintyEvidence"]["policy"][
                    "clusterIdentityFromEvaluationSplits"
                ] = False

            receipt, *_ = self._attest(Path(directory), mutate_statistical=mutate)
            self.assertFalse(receipt["passed"])
            self.assertIn("uncertainty_split_cluster_authority_missing", receipt["failures"])

    def test_wrong_cluster_dimension_cannot_authorize_release(self):
        with tempfile.TemporaryDirectory() as directory:
            def mutate(payload):
                payload["uncertaintyEvidence"]["coldPair"]["source"][
                    "clusterColumn"
                ] = "owner_pair_key"

            receipt, *_ = self._attest(Path(directory), mutate_statistical=mutate)
            self.assertFalse(receipt["passed"])
            self.assertIn("coldPair_cluster_column_invalid", receipt["failures"])

    def test_artifact_swap_invalidates_an_otherwise_valid_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt, model, calibration, manifest, contract = self._attest(root)
            model.write_bytes(b"swapped-model")

            valid, failures = verify_release_receipt(
                receipt,
                key="test-release-secret",
                model_path=model,
                calibration_path=calibration,
                training_manifest_path=manifest,
                feature_contract_path=contract,
            )
            self.assertFalse(valid)
            self.assertIn("modelSha256_mismatch", failures)

    def test_wrong_signing_key_invalidates_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt, model, calibration, manifest, contract = self._attest(root)
            valid, failures = verify_release_receipt(
                receipt,
                key="different-secret",
                model_path=model,
                calibration_path=calibration,
                training_manifest_path=manifest,
                feature_contract_path=contract,
            )
            self.assertFalse(valid)
            self.assertIn("signature_mismatch", failures)

    def test_failed_statistical_gate_can_only_produce_shadow_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, *_ = self._attest(Path(directory), statistical_passed=False)
            self.assertFalse(receipt["passed"])
            self.assertEqual(receipt["releaseStatus"], "shadow")
            self.assertIn("statistical_gate_not_passed", receipt["failures"])


if __name__ == "__main__":
    unittest.main()
