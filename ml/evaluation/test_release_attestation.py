from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ml.common.attestation import sha256_file, verify_release_receipt
from ml.evaluation.attest_promotion import attest_promotion


class ReleaseAttestationTests(unittest.TestCase):
    def _candidate(self, root: Path, statistical_passed: bool = True):
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
        statistical.write_text(
            json.dumps(
                {
                    "schemaVersion": "woof-model-promotion-receipt-v1",
                    "generatedAt": "2026-08-20T00:00:00+00:00",
                    "decision": "promote" if statistical_passed else "hold_shadow",
                    "passed": statistical_passed,
                }
            ),
            encoding="utf-8",
        )
        return model, calibration, manifest, contract, statistical

    def test_signed_receipt_is_bound_to_exact_candidate_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, calibration, manifest, contract, statistical = self._candidate(root)
            receipt = attest_promotion(
                statistical_receipt_path=statistical,
                model_path=model,
                calibration_path=calibration,
                training_manifest_path=manifest,
                feature_contract_path=contract,
                signing_key="test-release-secret",
                key_id="test-key-v1",
            )

            self.assertTrue(receipt["passed"])
            self.assertEqual(receipt["decision"], "promote")
            self.assertEqual(receipt["releaseStatus"], "promoted")
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

    def test_artifact_swap_invalidates_an_otherwise_valid_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, calibration, manifest, contract, statistical = self._candidate(root)
            receipt = attest_promotion(
                statistical_receipt_path=statistical,
                model_path=model,
                calibration_path=calibration,
                training_manifest_path=manifest,
                feature_contract_path=contract,
                signing_key="test-release-secret",
                key_id="test-key-v1",
            )
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
            model, calibration, manifest, contract, statistical = self._candidate(root)
            receipt = attest_promotion(
                statistical_receipt_path=statistical,
                model_path=model,
                calibration_path=calibration,
                training_manifest_path=manifest,
                feature_contract_path=contract,
                signing_key="test-release-secret",
                key_id="test-key-v1",
            )
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
            root = Path(directory)
            model, calibration, manifest, contract, statistical = self._candidate(
                root, statistical_passed=False
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
            self.assertFalse(receipt["passed"])
            self.assertEqual(receipt["releaseStatus"], "shadow")
            self.assertIn("statistical_gate_not_passed", receipt["failures"])


if __name__ == "__main__":
    unittest.main()
