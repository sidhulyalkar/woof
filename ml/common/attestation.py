"""Cryptographic release-attestation helpers for Woof ML artifacts."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Mapping

RELEASE_RECEIPT_SCHEMA = "woof-model-promotion-receipt-v2"
SIGNATURE_ALGORITHM = "HMAC-SHA256"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def unsigned_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key != "signature"}


def sign_receipt(receipt: Mapping[str, Any], key: str, key_id: str) -> dict[str, Any]:
    if not key:
        raise ValueError("promotion attestation key must not be empty")
    payload = unsigned_receipt(receipt)
    digest = hmac.new(key.encode("utf-8"), canonical_json_bytes(payload), hashlib.sha256).hexdigest()
    return {
        **payload,
        "signature": {
            "algorithm": SIGNATURE_ALGORITHM,
            "keyId": key_id,
            "digest": digest,
        },
    }


def verify_signature(receipt: Mapping[str, Any], key: str) -> tuple[bool, str | None]:
    signature = receipt.get("signature")
    if not isinstance(signature, dict):
        return False, "signature_missing"
    if signature.get("algorithm") != SIGNATURE_ALGORITHM:
        return False, "signature_algorithm_invalid"
    digest = signature.get("digest")
    if not isinstance(digest, str) or len(digest) != 64:
        return False, "signature_digest_invalid"
    expected = hmac.new(
        key.encode("utf-8"),
        canonical_json_bytes(unsigned_receipt(receipt)),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(digest, expected):
        return False, "signature_mismatch"
    return True, None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def artifact_hashes(
    model_path: Path,
    calibration_path: Path,
    training_manifest_path: Path,
    feature_contract_path: Path,
) -> dict[str, str]:
    return {
        "modelSha256": sha256_file(model_path),
        "calibrationSha256": sha256_file(calibration_path),
        "trainingManifestSha256": sha256_file(training_manifest_path),
        "featureContractSha256": sha256_file(feature_contract_path),
    }


def verify_release_receipt(
    receipt: Mapping[str, Any],
    *,
    key: str,
    model_path: Path,
    calibration_path: Path,
    training_manifest_path: Path,
    feature_contract_path: Path,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if receipt.get("schemaVersion") != RELEASE_RECEIPT_SCHEMA:
        failures.append("receipt_schema_invalid")
    if receipt.get("decision") != "promote" or receipt.get("passed") is not True:
        failures.append("receipt_not_promoted")
    if receipt.get("releaseStatus") != "promoted":
        failures.append("release_status_not_promoted")

    signature_ok, signature_failure = verify_signature(receipt, key)
    if not signature_ok and signature_failure:
        failures.append(signature_failure)

    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        failures.append("artifact_binding_missing")
        return False, list(dict.fromkeys(failures))

    expected = artifact_hashes(
        model_path,
        calibration_path,
        training_manifest_path,
        feature_contract_path,
    )
    for name, digest in expected.items():
        if artifacts.get(name) != digest:
            failures.append(f"{name}_mismatch")

    manifest = load_json(training_manifest_path)
    contract = load_json(feature_contract_path)
    identity = receipt.get("identity")
    if not isinstance(identity, dict):
        failures.append("identity_missing")
    else:
        if identity.get("modelVersion") != manifest.get("modelVersion"):
            failures.append("model_version_mismatch")
        if identity.get("featureVersion") != manifest.get("featureVersion"):
            failures.append("manifest_feature_version_mismatch")
        if identity.get("featureVersion") != contract.get("featureVersion"):
            failures.append("feature_contract_version_mismatch")
        calibration = manifest.get("calibration")
        calibration_version = calibration.get("version") if isinstance(calibration, dict) else None
        if identity.get("calibrationVersion") != calibration_version:
            failures.append("calibration_version_mismatch")

    return not failures, list(dict.fromkeys(failures))
