"""Package a canonical Woof compatibility candidate into independently hashable artifacts.

The trainer initially stores the estimator and calibrator together for research
convenience. Release review needs stronger boundaries, so this command rewrites
that candidate into:

- compatibility_model.joblib: estimator/scaler/blend only
- calibration.joblib: calibration object and version
- feature_contract.json: exact ordered feature contract
- training_manifest.json: final artifact identities and hashes

The resulting manifest remains a *shadow candidate*. Promotion still requires a
signed release receipt from ``attest_promotion.py``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from ml.common.attestation import sha256_file

PACKAGE_SCHEMA = "woof-canonical-model-package-v1"
FEATURE_CONTRACT_SCHEMA = "woof-compatibility-feature-contract-v1"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def package_candidate(output_dir: Path) -> dict[str, Any]:
    model_path = output_dir / "compatibility_model.joblib"
    manifest_path = output_dir / "training_manifest.json"
    if not model_path.exists() or not manifest_path.exists():
        raise ValueError("candidate directory must contain compatibility_model.joblib and training_manifest.json")

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise ValueError("compatibility_model.joblib must contain a dictionary bundle")

    model_version = bundle.get("modelVersion")
    feature_version = bundle.get("featureVersion")
    feature_names = bundle.get("featureNames")
    calibrator = bundle.get("calibrator")
    if not isinstance(model_version, str) or not model_version:
        raise ValueError("model bundle is missing modelVersion")
    if not isinstance(feature_version, str) or not feature_version:
        raise ValueError("model bundle is missing featureVersion")
    if not isinstance(feature_names, list) or not all(isinstance(name, str) for name in feature_names):
        raise ValueError("model bundle is missing ordered featureNames")
    if calibrator is None:
        raise ValueError("model bundle is missing calibrator")

    calibration_version = f"isotonic-v1-{model_version}"
    calibration_path = output_dir / "calibration.joblib"
    contract_path = output_dir / "feature_contract.json"

    core_bundle = {
        key: value
        for key, value in bundle.items()
        if key not in {"calibrator", "servingPolicy"}
    }
    core_bundle.update(
        {
            "packageSchemaVersion": PACKAGE_SCHEMA,
            "servingPolicy": "signed_promotion_receipt_required",
            "calibrationVersion": calibration_version,
        }
    )
    joblib.dump(core_bundle, model_path)
    joblib.dump(
        {
            "schemaVersion": "woof-calibration-artifact-v1",
            "modelVersion": model_version,
            "featureVersion": feature_version,
            "calibrationVersion": calibration_version,
            "calibrator": calibrator,
        },
        calibration_path,
    )

    feature_contract = {
        "schemaVersion": FEATURE_CONTRACT_SCHEMA,
        "modelVersion": model_version,
        "featureVersion": feature_version,
        "orderedFeatureNames": feature_names,
        "orderInvariantPetPair": True,
    }
    contract_path.write_text(json.dumps(feature_contract, indent=2), encoding="utf-8")

    manifest = load_json(manifest_path)
    if manifest.get("modelVersion") != model_version:
        raise ValueError("training manifest modelVersion does not match model artifact")
    if manifest.get("featureVersion") != feature_version:
        raise ValueError("training manifest featureVersion does not match model artifact")

    manifest.update(
        {
            "schemaVersion": "woof-canonical-model-training-v2",
            "packagedAt": datetime.now(timezone.utc).isoformat(),
            "packageSchemaVersion": PACKAGE_SCHEMA,
            "modelPath": str(model_path),
            "modelSha256": sha256_file(model_path),
            "calibration": {
                "version": calibration_version,
                "path": str(calibration_path),
                "sha256": sha256_file(calibration_path),
            },
            "featureContract": {
                "path": str(contract_path),
                "sha256": sha256_file(contract_path),
            },
            "releaseStatus": "shadow_candidate_only",
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "schemaVersion": PACKAGE_SCHEMA,
        "modelVersion": model_version,
        "featureVersion": feature_version,
        "calibrationVersion": calibration_version,
        "artifacts": {
            "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
            "calibration": {
                "path": str(calibration_path),
                "sha256": sha256_file(calibration_path),
            },
            "featureContract": {
                "path": str(contract_path),
                "sha256": sha256_file(contract_path),
            },
            "trainingManifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
        },
        "releaseStatus": "shadow_candidate_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package_candidate(args.candidate_dir), indent=2))


if __name__ == "__main__":
    main()
