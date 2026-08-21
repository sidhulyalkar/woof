"""Woof internal ML compatibility service.

The product API owns authorization, safety filters, model routing and fallback behavior.
This process has one job: accept the canonical compatibility feature contract and
return the same score envelope used by the deterministic scorer.

The historical neural checkpoint is intentionally exposed as a *shadow adapter*.
It was trained on older synthetic features (including size/weight) that the current
product does not require. We therefore use neutral legacy placeholders, cap its
confidence, and mark it uncalibrated so it cannot be promoted accidentally.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import redis
import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from models.compatibility_model import load_model as load_compat_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_VERSION = "legacy-neural-adapter-v1"
FEATURE_VERSION = "compatibility-features-v1"
CALIBRATION_VERSION = "uncalibrated-shadow-v1"
MODEL_PATH = BASE_DIR / "models" / "compatibility_model.pth"
BREED_ENCODING_PATH = BASE_DIR / "data" / "breed_encoding.json"

app = FastAPI(
    title="Woof Compatibility Model Service",
    description="Internal learned compatibility scoring behind the Woof API router.",
    version="3.0.0-beta.1",
)

MODELS: Dict[str, Any] = {}

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


class Provenance(BaseModel):
    scorer: str
    modelVersion: str
    featureVersion: str
    calibrationVersion: str
    generatedAt: str
    fallback: bool = False
    fallbackReason: Optional[str] = None


class CompatibilityResponse(BaseModel):
    compatibilityScore: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    source: str
    factors: Dict[str, float]
    explanation: list[str]
    provenance: Provenance


def _cache_key(request: CompatibilityRequest) -> str:
    payload = request.model_dump_json(exclude_none=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"woof:compatibility:v1:{digest}"


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
    """Map canonical evidence into the historical checkpoint's input surface.

    `size` and `weight` are neutral constants because the beta no longer requires
    those fields. This is why the adapter remains shadow-only and low confidence.
    """
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
    """Run the repaired historical model without constructing tensors from strings."""
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


def load_models() -> None:
    MODELS.clear()
    try:
        model, breed_to_idx, temp_to_idx = load_compat_model(
            str(MODEL_PATH),
            str(BREED_ENCODING_PATH),
        )
        MODELS["compatibility"] = {
            "model": model,
            "breed_to_idx": breed_to_idx,
            "temp_to_idx": temp_to_idx,
        }
    except Exception as exc:
        print(f"Compatibility shadow model not loaded: {exc}")


@app.on_event("startup")
async def startup_event() -> None:
    load_models()


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "serviceVersion": app.version,
        "featureVersion": FEATURE_VERSION,
        "modelVersion": MODEL_VERSION,
        "calibrationVersion": CALIBRATION_VERSION,
        "compatibilityModelLoaded": "compatibility" in MODELS,
        "redis": REDIS_AVAILABLE,
        "servingMode": "shadow-only",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/compatibility/score", response_model=CompatibilityResponse)
async def score_compatibility(request: CompatibilityRequest) -> CompatibilityResponse:
    if request.featureVersion != FEATURE_VERSION:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported featureVersion: {request.featureVersion}",
        )
    if request.petA.species != request.petB.species:
        raise HTTPException(status_code=422, detail="Cross-species learned scoring is out of domain")
    if "compatibility" not in MODELS:
        raise HTTPException(status_code=503, detail="Compatibility shadow model is unavailable")

    key = _cache_key(request)
    cached = _get_cached(key)
    if cached:
        cached["provenance"]["generatedAt"] = datetime.now(timezone.utc).isoformat()
        return CompatibilityResponse(**cached)

    try:
        pet_a = _legacy_pet(request.petA)
        pet_b = _legacy_pet(request.petB)
        raw_score = _legacy_predict(MODELS["compatibility"], pet_a, pet_b)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Compatibility inference failed") from exc

    coverage = min(request.petA.behavior.coverage, request.petB.behavior.coverage)
    outcome_signal = request.outcomes.positiveRate
    # Confidence is deliberately capped below the API promotion threshold. The old
    # checkpoint has not been calibrated on the canonical post-meetup outcome task.
    confidence = min(0.55, 0.25 + 0.25 * coverage + 0.05 * min(request.outcomes.sampleCount, 1))

    response = CompatibilityResponse(
        compatibilityScore=max(0.0, min(1.0, raw_score)),
        confidence=confidence,
        source=MODEL_VERSION,
        factors={
            "legacyNeuralRaw": max(0.0, min(1.0, raw_score)),
            "behaviorCoverage": coverage,
            "outcomeSignal": outcome_signal if outcome_signal is not None else 0.5,
        },
        explanation=[
            "Learned score from Woof's historical neural compatibility checkpoint.",
            "This checkpoint is shadow-only because its original synthetic feature set predates the beta contract.",
            "The production router continues to use the calibrated deterministic baseline until a canonical learned model earns promotion.",
        ],
        provenance=Provenance(
            scorer="learned",
            modelVersion=MODEL_VERSION,
            featureVersion=FEATURE_VERSION,
            calibrationVersion=CALIBRATION_VERSION,
            generatedAt=datetime.now(timezone.utc).isoformat(),
            fallback=False,
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
