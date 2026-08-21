"""Woof internal learned compatibility service.

The service supports two learned paths:

1. A canonical tabular candidate trained on the same behavior/outcome contract as
   the product API. It may emit ``releaseStatus=promoted`` only when its exact
   model, calibration, feature-contract and training-manifest bytes verify against
   a signed promotion receipt.
2. The historical neural checkpoint, retained as a low-confidence shadow adapter.

The NestJS API remains the authorization, safety-filtering and deterministic
fallback authority.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import redis
import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml.common.attestation import load_json, sha256_file, verify_release_receipt
from models.compatibility_model import load_model as load_compat_model

BASE_DIR = Path(__file__).resolve().parent
FEATURE_VERSION = "compatibility-features-v1"
LEGACY_MODEL_VERSION = "legacy-neural-adapter-v1"
LEGACY_CALIBRATION_VERSION = "uncalibrated-shadow-v1"
LEGACY_MODEL_PATH = BASE_DIR / "models" / "compatibility_model.pth"
BREED_ENCODING_PATH = BASE_DIR / "data" / "breed_encoding.json"
CANONICAL_DIR = Path(os.getenv("ML_CANONICAL_CANDIDATE_DIR", str(BASE_DIR / "artifacts" / "canonical")))
CANONICAL_MODEL_PATH = CANONICAL_DIR / "compatibility_model.joblib"
CANONICAL_CALIBRATION_PATH = CANONICAL_DIR / "calibration.joblib"
CANONICAL_MANIFEST_PATH = CANONICAL_DIR / "training_manifest.json"
CANONICAL_FEATURE_CONTRACT_PATH = CANONICAL_DIR / "feature_contract.json"
CANONICAL_RECEIPT_PATH = CANONICAL_DIR / "promotion_receipt.json"

app = FastAPI(
    title="Woof Compatibility Model Service",
    description="Internal learned compatibility scoring behind the Woof API router.",
    version="4.0.0-beta.1",
)

MODELS: Dict[str, Any] = {}
RELEASE_STATE: Dict[str, Any] = {
    "canonicalLoaded": False,
    "attestationVerified": False,
    "attestationFailures": [],
    "releaseStatus": "shadow",
}

try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        decode_responses=True,
        socket_connect_timeout=0.25,
        socket_timeout=0.25,
    )
    redis_client.ping()
    REDIS_AVAILABLE = True
except Exception:
    redis_client = None
    REDIS_AVAILABLE = False


class BehaviorFeatures(BaseModel):
    energy: Optional[float] = Field(default=None, ge=0, le=1)
    sociability: Optional[float] = Field(default=None, ge=0, le=1)
    caution: Optional[float] = Field(default=None, ge=0, le=1)
    excitability: Optional[float] = Field(default=None, ge=0, le=1)
    trainability: Optional[float] = Field(default=None, ge=0, le=1)
    socialRisk: Optional[float] = Field(default=None, ge=0, le=1)
    coverage: float = Field(ge=0, le=1)


class CanonicalPetFeatures(BaseModel):
    species: str
    breed: Optional[str] = None
    ageYears: Optional[float] = Field(default=None, ge=0, le=40)
    behavior: BehaviorFeatures


class OutcomeFeatures(BaseModel):
    sampleCount: int = Field(ge=0, le=1000)
    meanRating: Optional[float] = Field(default=None, ge=1, le=5)
    positiveRate: Optional[float] = Field(default=None, ge=0, le=1)
    repeatMeetupCount: int = Field(ge=0, le=1000)
    lastOutcomeDaysAgo: Optional[float] = Field(default=None, ge=0)


class CompatibilityRequest(BaseModel):
    featureVersion: str
    petA: CanonicalPetFeatures
    petB: CanonicalPetFeatures
    outcomes: OutcomeFeatures


class ArtifactHashes(BaseModel):
    modelSha256: str
    calibrationSha256: str
    trainingManifestSha256: str
    featureContractSha256: str


class Provenance(BaseModel):
    scorer: str
    modelVersion: str
    featureVersion: str
    calibrationVersion: str
    generatedAt: str
    fallback: bool = False
    fallbackReason: Optional[str] = None
    releaseStatus: str = "shadow"
    attestationId: Optional[str] = None
    promotionReceiptSha256: Optional[str] = None
    artifactHashes: Optional[ArtifactHashes] = None


class CompatibilityResponse(BaseModel):
    compatibilityScore: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    source: str
    factors: Dict[str, float]
    explanation: list[str]
    provenance: Provenance


def _active_identity() -> str:
    if "canonical" in MODELS:
        canonical = MODELS["canonical"]
        return ":".join(
            [
                str(canonical["modelVersion"]),
                str(RELEASE_STATE.get("releaseStatus", "shadow")),
                str(RELEASE_STATE.get("attestationId") or "unattested"),
            ]
        )
    return LEGACY_MODEL_VERSION


def _cache_key(request: CompatibilityRequest) -> str:
    payload = request.model_dump_json(exclude_none=True)
    digest = hashlib.sha256(f"{_active_identity()}:{payload}".encode("utf-8")).hexdigest()
    return f"woof:compatibility:v2:{digest}"


def _get_cached(key: str) -> Optional[Dict[str, Any]]:
    if not REDIS_AVAILABLE or redis_client is None:
        return None
    try:
        value = redis_client.get(key)
        return json.loads(value) if value else None
    except Exception:
        return None


def _set_cached(key: str, value: Dict[str, Any]) -> None:
    if not REDIS_AVAILABLE or redis_client is None:
        return
    try:
        redis_client.setex(key, 900, json.dumps(value))
    except Exception:
        return


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _canonical_feature_row(request: CompatibilityRequest) -> Dict[str, float]:
    pet_a = request.petA
    pet_b = request.petB
    output: Dict[str, float] = {}
    trait_names = ("energy", "sociability", "caution", "excitability", "trainability")
    observed_a = 0
    observed_b = 0
    for trait in trait_names:
        raw_a = getattr(pet_a.behavior, trait)
        raw_b = getattr(pet_b.behavior, trait)
        observed_a += int(raw_a is not None)
        observed_b += int(raw_b is not None)
        a = _clip(float(raw_a if raw_a is not None else 0.5), 0.0, 1.0)
        b = _clip(float(raw_b if raw_b is not None else 0.5), 0.0, 1.0)
        output[f"{trait}_mean"] = (a + b) / 2.0
        output[f"{trait}_gap"] = abs(a - b)

    risk_a = _clip(float(pet_a.behavior.socialRisk if pet_a.behavior.socialRisk is not None else 0.25), 0.0, 1.0)
    risk_b = _clip(float(pet_b.behavior.socialRisk if pet_b.behavior.socialRisk is not None else 0.25), 0.0, 1.0)
    output["social_risk_mean"] = (risk_a + risk_b) / 2.0
    output["social_risk_max"] = max(risk_a, risk_b)

    age_a = _clip(float(pet_a.ageYears if pet_a.ageYears is not None else 5.0), 0.0, 25.0)
    age_b = _clip(float(pet_b.ageYears if pet_b.ageYears is not None else 5.0), 0.0, 25.0)
    output["age_mean_scaled"] = _clip(((age_a + age_b) / 2.0) / 15.0, 0.0, 1.5)
    output["age_gap_scaled"] = _clip(abs(age_a - age_b) / 10.0, 0.0, 1.5)
    coverage_a = observed_a / len(trait_names)
    coverage_b = observed_b / len(trait_names)
    output["coverage_mean"] = (coverage_a + coverage_b) / 2.0
    output["coverage_min"] = min(coverage_a, coverage_b)

    prior_count = max(0.0, float(request.outcomes.sampleCount))
    output["prior_outcome_log_count"] = float(np.log1p(prior_count))
    output["prior_mean_rating_scaled"] = _clip(
        float(request.outcomes.meanRating if request.outcomes.meanRating is not None else 0.0),
        0.0,
        5.0,
    ) / 5.0
    output["prior_positive_rate"] = _clip(
        float(request.outcomes.positiveRate if request.outcomes.positiveRate is not None else 0.0),
        0.0,
        1.0,
    )
    output["prior_repeat_log_count"] = float(np.log1p(max(0.0, request.outcomes.repeatMeetupCount)))
    days = float(request.outcomes.lastOutcomeDaysAgo) if request.outcomes.lastOutcomeDaysAgo is not None else -1.0
    output["has_prior_outcome"] = 1.0 if days >= 0 else 0.0
    output["days_since_prior_scaled"] = _clip(max(days, 0.0), 0.0, 365.0) / 365.0
    return output


def _canonical_predict(model_data: Dict[str, Any], request: CompatibilityRequest) -> float:
    row = _canonical_feature_row(request)
    feature_names = model_data["featureNames"]
    if set(row) != set(feature_names):
        missing = sorted(set(feature_names) - set(row))
        extra = sorted(set(row) - set(feature_names))
        raise ValueError(f"serving feature contract mismatch missing={missing} extra={extra}")
    matrix = np.asarray([[row[name] for name in feature_names]], dtype=float)
    scaler = model_data["scaler"]
    logistic = model_data["logistic"]
    booster = model_data["booster"]
    blend = float(model_data["blendWeightBooster"])
    logistic_score = float(logistic.predict_proba(scaler.transform(matrix))[0, 1])
    booster_score = float(booster.predict_proba(matrix)[0, 1])
    raw = blend * booster_score + (1.0 - blend) * logistic_score
    calibrator = model_data["calibrator"]
    return _clip(float(calibrator.predict([raw])[0]), 0.0, 1.0)


def _energy_bucket(value: Optional[float]) -> str:
    if value is None:
        return "medium"
    if value < 0.34:
        return "low"
    if value > 0.66:
        return "high"
    return "medium"


def _temperament_bucket(behavior: BehaviorFeatures) -> str:
    candidates = [
        (behavior.sociability, "friendly"),
        (behavior.excitability, "playful"),
        (behavior.trainability, "intelligent"),
        (behavior.caution, "nervous"),
    ]
    present = [(value, label) for value, label in candidates if value is not None]
    if not present:
        return "calm"
    value, label = max(present, key=lambda item: item[0])
    return label if value >= 0.55 else "calm"


def _legacy_pet(pet: CanonicalPetFeatures) -> Dict[str, Any]:
    return {
        "breed": pet.breed or "unknown",
        "size": "medium",
        "energy": _energy_bucket(pet.behavior.energy),
        "temperament": _temperament_bucket(pet.behavior),
        "age": pet.ageYears if pet.ageYears is not None else 5.0,
        "social": pet.behavior.sociability if pet.behavior.sociability is not None else 0.5,
        "weight": 30.0,
    }


def _legacy_predict(model_data: Dict[str, Any], pet_a: Dict[str, Any], pet_b: Dict[str, Any]) -> float:
    model = model_data["model"]
    breed_to_idx = model_data["breed_to_idx"]
    temp_to_idx = model_data["temp_to_idx"]
    model.eval()
    with torch.no_grad():
        score = model.forward(
            torch.tensor([breed_to_idx.get(pet_a["breed"], 0)]),
            [pet_a["size"]],
            [pet_a["energy"]],
            torch.tensor([temp_to_idx.get(pet_a["temperament"], 0)]),
            torch.tensor([pet_a["age"]], dtype=torch.float32),
            torch.tensor([pet_a["social"]], dtype=torch.float32),
            torch.tensor([pet_a["weight"]], dtype=torch.float32),
            torch.tensor([breed_to_idx.get(pet_b["breed"], 0)]),
            [pet_b["size"]],
            [pet_b["energy"]],
            torch.tensor([temp_to_idx.get(pet_b["temperament"], 0)]),
            torch.tensor([pet_b["age"]], dtype=torch.float32),
            torch.tensor([pet_b["social"]], dtype=torch.float32),
            torch.tensor([pet_b["weight"]], dtype=torch.float32),
        )
    return float(score.item())


def _load_canonical() -> None:
    required = [
        CANONICAL_MODEL_PATH,
        CANONICAL_CALIBRATION_PATH,
        CANONICAL_MANIFEST_PATH,
        CANONICAL_FEATURE_CONTRACT_PATH,
    ]
    if not all(path.exists() for path in required):
        RELEASE_STATE.update(
            {
                "canonicalLoaded": False,
                "attestationVerified": False,
                "attestationFailures": ["canonical_artifacts_missing"],
                "releaseStatus": "shadow",
            }
        )
        return

    model_bundle = joblib.load(CANONICAL_MODEL_PATH)
    calibration_bundle = joblib.load(CANONICAL_CALIBRATION_PATH)
    manifest = load_json(CANONICAL_MANIFEST_PATH)
    contract = load_json(CANONICAL_FEATURE_CONTRACT_PATH)
    if not isinstance(model_bundle, dict) or not isinstance(calibration_bundle, dict):
        raise ValueError("canonical model artifacts must contain dictionary bundles")
    if model_bundle.get("modelVersion") != manifest.get("modelVersion"):
        raise ValueError("canonical modelVersion does not match training manifest")
    if model_bundle.get("featureVersion") != FEATURE_VERSION:
        raise ValueError("canonical model featureVersion is unsupported")
    if contract.get("orderedFeatureNames") != model_bundle.get("featureNames"):
        raise ValueError("canonical feature contract does not match model feature order")
    calibration_version = calibration_bundle.get("calibrationVersion")
    if not isinstance(calibration_version, str) or not calibration_version:
        raise ValueError("canonical calibration artifact is missing calibrationVersion")
    if calibration_bundle.get("modelVersion") != model_bundle.get("modelVersion"):
        raise ValueError("calibration modelVersion does not match model")

    canonical = {
        **model_bundle,
        "calibrator": calibration_bundle["calibrator"],
        "calibrationVersion": calibration_version,
        "manifest": manifest,
        "contract": contract,
    }
    MODELS["canonical"] = canonical
    RELEASE_STATE.update(
        {
            "canonicalLoaded": True,
            "attestationVerified": False,
            "attestationFailures": ["promotion_receipt_missing"],
            "releaseStatus": "shadow",
            "attestationId": None,
            "promotionReceiptSha256": None,
        }
    )

    key = os.getenv("ML_PROMOTION_ATTESTATION_KEY", "")
    if not CANONICAL_RECEIPT_PATH.exists():
        return
    if not key:
        RELEASE_STATE["attestationFailures"] = ["attestation_key_missing"]
        return

    receipt = load_json(CANONICAL_RECEIPT_PATH)
    verified, failures = verify_release_receipt(
        receipt,
        key=key,
        model_path=CANONICAL_MODEL_PATH,
        calibration_path=CANONICAL_CALIBRATION_PATH,
        training_manifest_path=CANONICAL_MANIFEST_PATH,
        feature_contract_path=CANONICAL_FEATURE_CONTRACT_PATH,
    )
    RELEASE_STATE.update(
        {
            "attestationVerified": verified,
            "attestationFailures": failures,
            "releaseStatus": "promoted" if verified else "shadow",
            "attestationId": receipt.get("attestationId") if verified else None,
            "promotionReceiptSha256": sha256_file(CANONICAL_RECEIPT_PATH) if verified else None,
            "artifactHashes": receipt.get("artifacts") if verified else None,
        }
    )


def _load_legacy() -> None:
    try:
        model, breed_to_idx, temp_to_idx = load_compat_model(
            str(LEGACY_MODEL_PATH),
            str(BREED_ENCODING_PATH),
        )
        MODELS["legacy"] = {
            "model": model,
            "breed_to_idx": breed_to_idx,
            "temp_to_idx": temp_to_idx,
        }
    except Exception as exc:
        print(f"Compatibility legacy shadow model not loaded: {exc}")


def load_models() -> None:
    MODELS.clear()
    try:
        _load_canonical()
    except Exception as exc:
        RELEASE_STATE.update(
            {
                "canonicalLoaded": False,
                "attestationVerified": False,
                "attestationFailures": [f"canonical_load_failed:{type(exc).__name__}"],
                "releaseStatus": "shadow",
            }
        )
        print(f"Canonical compatibility candidate not loaded: {exc}")
    _load_legacy()


@app.on_event("startup")
async def startup_event() -> None:
    load_models()


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    canonical = MODELS.get("canonical")
    return {
        "status": "healthy",
        "serviceVersion": app.version,
        "featureVersion": FEATURE_VERSION,
        "activeModelVersion": canonical.get("modelVersion") if canonical else LEGACY_MODEL_VERSION,
        "canonicalModelLoaded": "canonical" in MODELS,
        "legacyModelLoaded": "legacy" in MODELS,
        "releaseStatus": RELEASE_STATE.get("releaseStatus", "shadow"),
        "attestationVerified": RELEASE_STATE.get("attestationVerified", False),
        "attestationId": RELEASE_STATE.get("attestationId"),
        "attestationFailures": RELEASE_STATE.get("attestationFailures", []),
        "redis": REDIS_AVAILABLE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/compatibility/score", response_model=CompatibilityResponse)
async def score_compatibility(request: CompatibilityRequest) -> CompatibilityResponse:
    if request.featureVersion != FEATURE_VERSION:
        raise HTTPException(status_code=422, detail=f"Unsupported featureVersion: {request.featureVersion}")
    if request.petA.species != request.petB.species:
        raise HTTPException(status_code=422, detail="Cross-species learned scoring is out of domain")
    if "canonical" not in MODELS and "legacy" not in MODELS:
        raise HTTPException(status_code=503, detail="Compatibility learned models are unavailable")

    key = _cache_key(request)
    cached = _get_cached(key)
    if cached:
        cached["provenance"]["generatedAt"] = datetime.now(timezone.utc).isoformat()
        return CompatibilityResponse(**cached)

    coverage = min(request.petA.behavior.coverage, request.petB.behavior.coverage)
    outcome_signal = request.outcomes.positiveRate
    if "canonical" in MODELS:
        canonical = MODELS["canonical"]
        try:
            score = _canonical_predict(canonical, request)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Canonical compatibility inference failed") from exc

        promoted = RELEASE_STATE.get("attestationVerified") is True
        confidence = min(0.9 if promoted else 0.72, 0.42 + 0.38 * coverage + 0.03 * min(request.outcomes.sampleCount, 3))
        artifact_hashes = RELEASE_STATE.get("artifactHashes")
        response = CompatibilityResponse(
            compatibilityScore=score,
            confidence=confidence,
            source=str(canonical["modelVersion"]),
            factors={
                "canonicalCalibrated": score,
                "behaviorCoverage": coverage,
                "outcomeSignal": outcome_signal if outcome_signal is not None else 0.5,
            },
            explanation=[
                "Score from Woof's canonical order-invariant behavior/outcome model.",
                "The score is calibrated on held-out temporal data and remains subject to product safety filters.",
                (
                    "This exact artifact set has a verified signed promotion receipt."
                    if promoted
                    else "This candidate is shadow-only because a verified signed promotion receipt is not active."
                ),
            ],
            provenance=Provenance(
                scorer="learned",
                modelVersion=str(canonical["modelVersion"]),
                featureVersion=FEATURE_VERSION,
                calibrationVersion=str(canonical["calibrationVersion"]),
                generatedAt=datetime.now(timezone.utc).isoformat(),
                fallback=False,
                releaseStatus="promoted" if promoted else "shadow",
                attestationId=RELEASE_STATE.get("attestationId"),
                promotionReceiptSha256=RELEASE_STATE.get("promotionReceiptSha256"),
                artifactHashes=ArtifactHashes(**artifact_hashes) if isinstance(artifact_hashes, dict) else None,
            ),
        )
    else:
        try:
            pet_a = _legacy_pet(request.petA)
            pet_b = _legacy_pet(request.petB)
            raw_score = _legacy_predict(MODELS["legacy"], pet_a, pet_b)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Legacy compatibility inference failed") from exc
        confidence = min(0.55, 0.25 + 0.25 * coverage + 0.05 * min(request.outcomes.sampleCount, 1))
        response = CompatibilityResponse(
            compatibilityScore=_clip(raw_score, 0.0, 1.0),
            confidence=confidence,
            source=LEGACY_MODEL_VERSION,
            factors={
                "legacyNeuralRaw": _clip(raw_score, 0.0, 1.0),
                "behaviorCoverage": coverage,
                "outcomeSignal": outcome_signal if outcome_signal is not None else 0.5,
            },
            explanation=[
                "Learned score from Woof's historical neural compatibility checkpoint.",
                "This checkpoint is permanently shadow-only because its original synthetic feature set predates the beta contract.",
                "The product router continues to use the deterministic baseline unless a canonical model earns signed promotion.",
            ],
            provenance=Provenance(
                scorer="learned",
                modelVersion=LEGACY_MODEL_VERSION,
                featureVersion=FEATURE_VERSION,
                calibrationVersion=LEGACY_CALIBRATION_VERSION,
                generatedAt=datetime.now(timezone.utc).isoformat(),
                fallback=False,
                releaseStatus="shadow",
            ),
        )

    _set_cached(key, response.model_dump())
    return response


@app.delete("/cache/clear")
async def clear_cache() -> Dict[str, str]:
    if not REDIS_AVAILABLE or redis_client is None:
        return {"message": "Cache is disabled"}
    try:
        redis_client.flushdb()
        return {"message": "Cache cleared"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Cache clear failed") from exc


@app.post("/models/reload")
async def reload_models(background_tasks: BackgroundTasks) -> Dict[str, str]:
    background_tasks.add_task(load_models)
    return {"message": "Model reload initiated"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info", reload=False)
