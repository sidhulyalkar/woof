"""
FastAPI ML Service for Woof/PetPath

Serves machine learning models for:
- Pet compatibility prediction (GNN + basic model)
- Energy state classification
- Activity recommendations (Temporal Transformer)
- Graph-based social suggestions

Architecture:
- FastAPI for REST API
- Redis for caching predictions
- Model hot-loading without downtime
- Batch prediction support
- Health monitoring and metrics
"""

import os
import json
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import redis
from datetime import datetime, timedelta
import hashlib

# Import models
from models.compatibility_model import CompatibilityModel, load_model as load_compat_model
from models.energy_model import EnergyStateModel, load_model as load_energy_model

# Initialize FastAPI app
app = FastAPI(
    title="Woof ML API",
    description="Machine Learning service for pet compatibility and activity prediction",
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis cache (optional)
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=0,
        decode_responses=True,
    )
    redis_client.ping()
    REDIS_AVAILABLE = True
except:
    REDIS_AVAILABLE = False
    print("⚠️  Redis not available, caching disabled")

# Model storage
MODELS = {}


# ===== Request/Response Models =====

class PetFeatures(BaseModel):
    """Pet feature schema"""
    breed: str
    size: str  # small, medium, large
    energy: str  # low, medium, high
    temperament: str
    age: float = Field(gt=0, le=20)
    social: float = Field(ge=0, le=1)
    weight: float = Field(gt=0)


class CompatibilityRequest(BaseModel):
    """Compatibility prediction request"""
    pet1: PetFeatures
    pet2: PetFeatures


class CompatibilityResponse(BaseModel):
    """Compatibility prediction response"""
    compatibility_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    factors: Dict[str, Any]
    cached: bool = False


class EnergyRequest(BaseModel):
    """Energy state prediction request"""
    age: float
    breed: str
    base_energy_level: str
    hours_since_last_activity: float
    total_distance_24h: float  # meters
    total_duration_24h: float  # minutes
    num_activities_24h: int
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)


class EnergyResponse(BaseModel):
    """Energy state prediction response"""
    energy_state: str  # low, medium, high
    probabilities: Dict[str, float]
    confidence: float
    recommendation: str
    cached: bool = False


class BatchCompatibilityRequest(BaseModel):
    """Batch compatibility predictions"""
    pairs: List[CompatibilityRequest]


class ActivitySequence(BaseModel):
    """Activity sequence for temporal prediction"""
    activity_types: List[int]  # Activity type IDs
    features: List[List[float]]  # Features per activity
    hours: List[int]
    days: List[int]


class ActivityRecommendationRequest(BaseModel):
    """Request for activity recommendations"""
    pet_id: str
    recent_activities: ActivitySequence
    current_energy: str
    preferences: Optional[Dict[str, Any]] = None


class ActivityRecommendation(BaseModel):
    """Activity recommendation"""
    activity_type: str
    probability: float
    optimal_time: int  # Hour of day
    expected_duration: float  # Minutes
    energy_requirement: str


class ActivityRecommendationResponse(BaseModel):
    """Activity recommendation response"""
    recommendations: List[ActivityRecommendation]
    predicted_energy: str
    confidence: float


# ===== Helper Functions =====

def generate_cache_key(prefix: str, data: Dict) -> str:
    """Generate cache key from data"""
    data_str = json.dumps(data, sort_keys=True)
    hash_value = hashlib.md5(data_str.encode()).hexdigest()
    return f"{prefix}:{hash_value}"


def get_cached_prediction(key: str) -> Optional[Dict]:
    """Get cached prediction from Redis"""
    if not REDIS_AVAILABLE:
        return None

    try:
        cached = redis_client.get(key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        print(f"Redis error: {e}")

    return None


def cache_prediction(key: str, value: Dict, ttl: int = 3600):
    """Cache prediction in Redis"""
    if not REDIS_AVAILABLE:
        return

    try:
        redis_client.setex(key, ttl, json.dumps(value))
    except Exception as e:
        print(f"Redis error: {e}")


def load_models():
    """Load all ML models"""
    global MODELS

    print("Loading ML models...")

    # Load compatibility model
    try:
        compat_model, breed_to_idx, temp_to_idx = load_compat_model(
            'models/compatibility_model.pth',
            'data/breed_encoding.json'
        )
        MODELS['compatibility'] = {
            'model': compat_model,
            'breed_to_idx': breed_to_idx,
            'temp_to_idx': temp_to_idx,
        }
        print("✅ Compatibility model loaded")
    except Exception as e:
        print(f"⚠️  Compatibility model not loaded: {e}")

    # Load energy model
    try:
        energy_model, breed_to_idx = load_energy_model(
            'models/energy_model.pth',
            'data/breed_encoding.json'
        )
        MODELS['energy'] = {
            'model': energy_model,
            'breed_to_idx': breed_to_idx,
        }
        print("✅ Energy model loaded")
    except Exception as e:
        print(f"⚠️  Energy model not loaded: {e}")

    # Note: GNN and Transformer models would be loaded here when trained
    # MODELS['gnn'] = load_gnn_model(...)
    # MODELS['transformer'] = load_transformer_model(...)


def analyze_compatibility_factors(pet1: PetFeatures, pet2: PetFeatures) -> Dict[str, Any]:
    """Analyze factors contributing to compatibility"""
    factors = {}

    # Energy match
    energy_map = {'low': 1, 'medium': 2, 'high': 3}
    energy_diff = abs(energy_map[pet1.energy] - energy_map[pet2.energy])
    factors['energy_match'] = 1.0 - (energy_diff / 2.0)

    # Size compatibility
    size_map = {'small': 1, 'medium': 2, 'large': 3}
    size_diff = abs(size_map[pet1.size] - size_map[pet2.size])
    factors['size_compatibility'] = 1.0 - (size_diff / 2.0)

    # Age proximity
    age_diff = abs(pet1.age - pet2.age)
    factors['age_proximity'] = max(0, 1.0 - (age_diff / 10.0))

    # Social scores
    factors['social_affinity'] = (pet1.social + pet2.social) / 2.0

    return factors


# ===== API Endpoints =====

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()


@app.get("/")
async def root():
    """API root"""
    return {
        "name": "Woof ML API",
        "version": "2.0.0",
        "status": "operational",
        "models": list(MODELS.keys()),
        "redis": "connected" if REDIS_AVAILABLE else "unavailable",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(MODELS),
        "redis": REDIS_AVAILABLE,
    }


@app.post("/predict/compatibility", response_model=CompatibilityResponse)
async def predict_compatibility(request: CompatibilityRequest):
    """
    Predict compatibility score between two pets

    Uses trained neural network model considering:
    - Breed characteristics
    - Size and energy level matching
    - Temperament compatibility
    - Age and social factors
    """
    # Check cache
    cache_key = generate_cache_key("compat", request.dict())
    cached = get_cached_prediction(cache_key)
    if cached:
        cached['cached'] = True
        return CompatibilityResponse(**cached)

    # Get model
    if 'compatibility' not in MODELS:
        raise HTTPException(status_code=503, detail="Compatibility model not loaded")

    model_data = MODELS['compatibility']
    model = model_data['model']
    breed_to_idx = model_data['breed_to_idx']
    temp_to_idx = model_data['temp_to_idx']

    # Prepare data
    pet1_dict = request.pet1.dict()
    pet2_dict = request.pet2.dict()

    # Predict
    try:
        score = model.predict(pet1_dict, pet2_dict, breed_to_idx, temp_to_idx)

        # Analyze factors
        factors = analyze_compatibility_factors(request.pet1, request.pet2)

        # Calculate confidence (based on model certainty)
        confidence = min(0.95, max(0.6, 1.0 - abs(score - 0.5) * 2))

        response = {
            "compatibility_score": float(score),
            "confidence": float(confidence),
            "factors": factors,
            "cached": False,
        }

        # Cache result
        cache_prediction(cache_key, response, ttl=3600)

        return CompatibilityResponse(**response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/energy", response_model=EnergyResponse)
async def predict_energy(request: EnergyRequest):
    """
    Predict current energy state of a pet

    Considers:
    - Recent activity history
    - Time of day and day of week
    - Breed characteristics
    - Age and base energy level
    """
    # Check cache
    cache_key = generate_cache_key("energy", request.dict())
    cached = get_cached_prediction(cache_key)
    if cached:
        cached['cached'] = True
        return EnergyResponse(**cached)

    # Get model
    if 'energy' not in MODELS:
        raise HTTPException(status_code=503, detail="Energy model not loaded")

    model_data = MODELS['energy']
    model = model_data['model']
    breed_to_idx = model_data['breed_to_idx']

    # Prepare data
    features = request.dict()

    # Predict
    try:
        predicted_class, probs, class_name = model.predict(features, breed_to_idx)

        # Map probabilities
        prob_dict = {
            'low': probs[0],
            'medium': probs[1],
            'high': probs[2],
        }

        # Generate recommendation
        if class_name == 'high':
            recommendation = "Great time for an active walk or play session!"
        elif class_name == 'medium':
            recommendation = "Moderate activity recommended - a casual walk or gentle play."
        else:
            recommendation = "Pet needs rest. Light activity or quiet time recommended."

        confidence = max(probs)

        response = {
            "energy_state": class_name,
            "probabilities": prob_dict,
            "confidence": float(confidence),
            "recommendation": recommendation,
            "cached": False,
        }

        # Cache result (shorter TTL since energy changes)
        cache_prediction(cache_key, response, ttl=300)

        return EnergyResponse(**response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/compatibility/batch")
async def batch_predict_compatibility(request: BatchCompatibilityRequest):
    """Batch compatibility predictions for efficiency"""
    results = []

    for pair in request.pairs:
        try:
            result = await predict_compatibility(pair)
            results.append(result.dict())
        except Exception as e:
            results.append({"error": str(e)})

    return {"results": results, "count": len(results)}


@app.post("/recommend/activities", response_model=ActivityRecommendationResponse)
async def recommend_activities(request: ActivityRecommendationRequest):
    """
    Recommend activities based on temporal patterns

    NOTE: Requires trained Temporal Transformer model
    Currently returns rule-based recommendations
    """
    # TODO: Use Temporal Transformer when trained

    # Simple rule-based recommendations for now
    energy_map = {'low': 1, 'medium': 2, 'high': 3}
    current_energy_level = energy_map.get(request.current_energy, 2)

    recommendations = []

    if current_energy_level >= 2:
        recommendations.append(ActivityRecommendation(
            activity_type="walk",
            probability=0.85,
            optimal_time=9,
            expected_duration=30.0,
            energy_requirement="medium"
        ))
        recommendations.append(ActivityRecommendation(
            activity_type="play",
            probability=0.75,
            optimal_time=16,
            expected_duration=20.0,
            energy_requirement="high"
        ))

    recommendations.append(ActivityRecommendation(
        activity_type="training",
        probability=0.60,
        optimal_time=10,
        expected_duration=15.0,
        energy_requirement="low"
    ))

    return ActivityRecommendationResponse(
        recommendations=recommendations,
        predicted_energy="medium",
        confidence=0.70
    )


@app.delete("/cache/clear")
async def clear_cache():
    """Clear all cached predictions"""
    if not REDIS_AVAILABLE:
        return {"message": "Redis not available"}

    try:
        redis_client.flushdb()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@app.post("/models/reload")
async def reload_models(background_tasks: BackgroundTasks):
    """
    Reload models without downtime
    Loads new models in background and hot-swaps them
    """
    background_tasks.add_task(load_models)
    return {"message": "Model reload initiated"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=False,  # Disable in production
    )
