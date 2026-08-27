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

STATISTICAL_RECEIPT_SCHEMA = "woof-model-promotion-receipt-v2"
UNCERTAINTY_REPORT_SCHEMA = "woof-compatibility-uncertainty-v1"
EXPECTED_CLUSTER_COLUMNS = {
    "futureTest": "owner_pair_key",
    "coldPair": "pair_key",
    "coldOwner": "owner_pair_key",
    "safety": "owner_pair_key",
}


def _sha256_hex(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


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


def _uncertainty_slice_authority_failures(
    uncertainty: dict[str, Any], slice_name: str
) -> list[str]:
    failures: list[str] = []
    block = uncertainty.get(slice_name)
    if not isinstance(block, dict) or block.get("available") is not True:
        return [f"{slice_name}_uncertainty_authority_missing"]

    source = block.get("source")
    if not isinstance(source, dict):
        return [f"{slice_name}_uncertainty_source_missing"]

    if not _sha256_hex(source.get("predictionSha256")):
        failures.append(f"{slice_name}_prediction_hash_missing")
    if not _sha256_hex(source.get("clusterSourceSha256")):
        failures.append(f"{slice_name}_cluster_source_hash_missing")
    if source.get("clusterJoinKey") != "outcome_id":
        failures.append(f"{slice_name}_cluster_join_key_invalid")
    if source.get("clusterLabelVerified") is not True:
        failures.append(f"{slice_name}_cluster_label_verification_missing")
    if source.get("clusterColumn") != EXPECTED_CLUSTER_COLUMNS[slice_name]:
        failures.append(f"{slice_name}_cluster_column_invalid")
    return failures


def _statistical_authority_failures(statistical: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if statistical.get("schemaVersion") != STATISTICAL_RECEIPT_SCHEMA:
        failures.append("statistical_receipt_schema_not_authoritative")
    if statistical.get("passed") is not True or statistical.get("decision") != "promote":
        failures.append("statistical_gate_not_passed")
    if statistical.get("authoritativeEligible") is not True:
        failures.append("statistical_receipt_not_authoritative")

    uncertainty = statistical.get("uncertaintyEvidence")
    if not isinstance(uncertainty, dict):
        failures.append("uncertainty_authority_missing")
        return failures

    if uncertainty.get("required") is not True:
        failures.append("uncertainty_not_required_by_statistical_gate")
    if uncertainty.get("passed") is not True:
        failures.append("uncertainty_gate_not_passed")
    if uncertainty.get("schemaVersion") != UNCERTAINTY_REPORT_SCHEMA:
        failures.append("uncertainty_schema_not_authoritative")

    policy = uncertainty.get("policy")
    if not isinstance(policy, dict):
        failures.append("uncertainty_policy_missing")
    else:
        if policy.get("pairedRows") is not True:
            failures.append("uncertainty_paired_rows_policy_missing")
        if policy.get("relationshipClusterBootstrap") is not True:
            failures.append("uncertainty_cluster_bootstrap_policy_missing")
        if policy.get("clusterIdentityFromEvaluationSplits") is not True:
            failures.append("uncertainty_split_cluster_authority_missing")
        if policy.get("splitLabelAgreementRequired") is not True:
            failures.append("uncertainty_split_label_authority_missing")

    for slice_name in ("futureTest", "coldPair", "coldOwner"):
        failures.extend(_uncertainty_slice_authority_failures(uncertainty, slice_name))

    safety = statistical.get("safety")
    if isinstance(safety, dict) and safety.get("available") is True:
        failures.extend(_uncertainty_slice_authority_failures(uncertainty, "safety"))
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

    failures = _statistical_authority_failures(statistical)
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

    uncertainty = statistical.get("uncertaintyEvidence")
    uncertainty_schema = uncertainty.get("schemaVersion") if isinstance(uncertainty, dict) else None
    uncertainty_policy = uncertainty.get("policy") if isinstance(uncertainty, dict) else None

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
            "statisticalAuthoritativeEligible": statistical.get("authoritativeEligible"),
            "uncertaintyReportSchema": uncertainty_schema,
            "uncertaintyPolicy": uncertainty_policy,
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
