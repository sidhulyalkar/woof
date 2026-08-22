"""Bind a statistically eligible Woof model to exact artifacts and sign its release receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml.common.attestation import (
    RELEASE_RECEIPT_SCHEMA,
    artifact_hashes,
    canonical_json_bytes,
    load_json,
    sha256_file,
    sign_receipt,
)


def _artifact_consistency_failures(
    manifest: dict[str, Any],
    contract: dict[str, Any],
    hashes: dict[str, str],
) -> list[str]:
    failures: list[str] = []
    if manifest.get("modelSha256") != hashes["modelSha256"]:
        failures.append("manifest_model_hash_mismatch")

    calibration = manifest.get("calibration")
    if not isinstance(calibration, dict):
        failures.append("manifest_calibration_missing")
    elif calibration.get("sha256") != hashes["calibrationSha256"]:
        failures.append("manifest_calibration_hash_mismatch")

    feature_contract = manifest.get("featureContract")
    if not isinstance(feature_contract, dict):
        failures.append("manifest_feature_contract_missing")
    elif feature_contract.get("sha256") != hashes["featureContractSha256"]:
        failures.append("manifest_feature_contract_hash_mismatch")

    if manifest.get("featureVersion") != contract.get("featureVersion"):
        failures.append("feature_version_mismatch")
    if manifest.get("modelVersion") != contract.get("modelVersion"):
        failures.append("feature_contract_model_version_mismatch")
    if manifest.get("releaseStatus") != "shadow_candidate_only":
        failures.append("candidate_release_state_invalid")
    return failures


def attest_promotion(
    *,
    statistical_receipt_path: Path,
    model_path: Path,
    calibration_path: Path,
    training_manifest_path: Path,
    feature_contract_path: Path,
    signing_key: str,
    key_id: str,
) -> dict[str, Any]:
    statistical = load_json(statistical_receipt_path)
    manifest = load_json(training_manifest_path)
    contract = load_json(feature_contract_path)
    hashes = artifact_hashes(
        model_path,
        calibration_path,
        training_manifest_path,
        feature_contract_path,
    )

    failures: list[str] = []
    if statistical.get("passed") is not True or statistical.get("decision") != "promote":
        failures.append("statistical_gate_not_passed")
    failures.extend(_artifact_consistency_failures(manifest, contract, hashes))

    calibration = manifest.get("calibration")
    calibration_version = calibration.get("version") if isinstance(calibration, dict) else None
    if not isinstance(calibration_version, str) or not calibration_version:
        failures.append("calibration_version_missing")

    identity = {
        "modelVersion": manifest.get("modelVersion"),
        "featureVersion": manifest.get("featureVersion"),
        "calibrationVersion": calibration_version,
    }
    if not all(isinstance(value, str) and value for value in identity.values()):
        failures.append("model_identity_incomplete")

    deduplicated = list(dict.fromkeys(failures))
    release_status = "promoted" if not deduplicated else "shadow"
    decision = "promote" if not deduplicated else "hold_shadow"
    statistical_hash = sha256_file(statistical_receipt_path)

    attestation_seed = {
        "identity": identity,
        "artifacts": hashes,
        "statisticalReceiptSha256": statistical_hash,
    }
    attestation_id = "woof-" + hashlib.sha256(canonical_json_bytes(attestation_seed)).hexdigest()[:24]

    receipt: dict[str, Any] = {
        "schemaVersion": RELEASE_RECEIPT_SCHEMA,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "attestationId": attestation_id,
        "decision": decision,
        "passed": not deduplicated,
        "releaseStatus": release_status,
        "identity": identity,
        "artifacts": hashes,
        "evidence": {
            "statisticalReceiptSha256": statistical_hash,
            "statisticalReceiptSchema": statistical.get("schemaVersion"),
            "statisticalDecision": statistical.get("decision"),
            "statisticalGeneratedAt": statistical.get("generatedAt"),
        },
        "failures": deduplicated,
    }
    return sign_receipt(receipt, signing_key, key_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistical-receipt", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--key-id",
        default=os.getenv("ML_PROMOTION_ATTESTATION_KEY_ID", "woof-release-v1"),
    )
    parser.add_argument("--non-blocking", action="store_true")
    args = parser.parse_args()

    key = os.getenv("ML_PROMOTION_ATTESTATION_KEY", "")
    if not key:
        raise SystemExit("ML_PROMOTION_ATTESTATION_KEY is required for release attestation")

    receipt = attest_promotion(
        statistical_receipt_path=args.statistical_receipt,
        model_path=args.model,
        calibration_path=args.calibration,
        training_manifest_path=args.training_manifest,
        feature_contract_path=args.feature_contract,
        signing_key=key,
        key_id=args.key_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if not receipt["passed"] and not args.non_blocking:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
