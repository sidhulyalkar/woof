"""
PetPath ML Inference Script
Inference models for pet compatibility and energy state prediction
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Import model definitions from train.py
from train import CompatibilityModel, EnergyState

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompatibilityRequest(BaseModel):
    """Request model for compatibility prediction"""
    pet_a_breed: str
    pet_b_breed: str
    pet_a_age: int
    pet_b_age: int
    pet_a_size: str
    pet_b_size: str
    past_interactions: int
    play_success_rate: float
    energy_level_diff: float


class EnergyRequest(BaseModel):
    """Request model for energy state prediction"""
    activity_duration: float  # minutes
    distance_walked: float  # km
    heart_rate_avg: int
    rest_periods: int
    play_time: float  # minutes
    social_interactions: int
    time_outdoors: float  # minutes


class CompatibilityResponse(BaseModel):
    """Response model for compatibility prediction"""
    compatibility_score: float
    confidence: float
    recommendation: str


class EnergyResponse(BaseModel):
    """Response model for energy state prediction"""
    energy_state: str  # low, medium, high
    readiness_score: float
    recommendations: List[str]


class PetPathMLInference:
    """Main inference class for PetPath ML models"""
    
    def __init__(self, model_path: str = "models/"):
        self.model_path = model_path
        self.compatibility_model = None
        self.energy_model = None
        self.breed_encoder = None
        self.size_encoder = None
        self.compatibility_scaler = None
        self.energy_scaler = None
        
        # Initialize encoders and scalers
        self._initialize_preprocessors()
        
        # Load models
        self._load_models()
    
    def _initialize_preprocessors(self):
        """Initialize preprocessors for data transformation"""
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        # Breed encoder (fit on common breeds)
        self.breed_encoder = LabelEncoder()
        common_breeds = ['Golden Retriever', 'Labrador', 'German Shepherd', 'Bulldog', 'Poodle',
                         'Siamese', 'Persian', 'Maine Coon', 'Bengal', 'Ragdoll']
        self.breed_encoder.fit(common_breeds)
        
        # Size encoder
        self.size_encoder = LabelEncoder()
        self.size_encoder.fit(['small', 'medium', 'large'])
        
        # Scalers
        self.compatibility_scaler = StandardScaler()
        self.energy_scaler = StandardScaler()
        
        # Fit scalers on sample data (in production, this would be saved from training)
        compatibility_sample = np.array([
            [0, 1, 1, 2, 3, 5, 0.5, 0.2, 0.8],  # Sample compatibility features
            [1, 0, 2, 1, 5, 3, 0.7, -0.1, 0.6]
        ])
        self.compatibility_scaler.fit(compatibility_sample)
        
        energy_sample = np.array([
            [120, 5.2, 120, 3, 60, 5, 180],  # Sample energy features
            [60, 2.1, 90, 5, 30, 2, 90]
        ])
        self.energy_scaler.fit(energy_sample)
    
    def _load_models(self):
        """Load trained models"""
        try:
            # Initialize models with correct input sizes
            self.compatibility_model = CompatibilityModel(input_size=9)
            self.energy_model = EnergyStateModel(input_size=7)
            
            # Load model weights (if files exist)
            try:
                self.compatibility_model.load_state_dict(
                    torch.load(f"{self.model_path}compatibility_model.pth", map_location='cpu')
                )
                logger.info("Compatibility model loaded successfully")
            except FileNotFoundError:
                logger.warning("Compatibility model not found, using untrained model")
            
            try:
                self.energy_model.load_state_dict(
                    torch.load(f"{self.model_path}energy_model.pth", map_location='cpu')
                )
                logger.info("Energy model loaded successfully")
            except FileNotFoundError:
                logger.warning("Energy model not found, using untrained model")
            
            # Set models to evaluation mode
            self.compatibility_model.eval()
            self.energy_model.eval()
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def predict_compatibility(self, request: CompatibilityRequest) -> CompatibilityResponse:
        """Predict pet compatibility"""
        
        try:
            # Encode categorical variables
            pet_a_breed_encoded = self.breed_encoder.transform([request.pet_a_breed])[0]
            pet_b_breed_encoded = self.breed_encoder.transform([request.pet_b_breed])[0]
            pet_a_size_encoded = self.size_encoder.transform([request.pet_a_size])[0]
            pet_b_size_encoded = self.size_encoder.transform([request.pet_b_size])[0]
            
            # Create feature array
            features = np.array([[
                pet_a_breed_encoded,
                pet_b_breed_encoded,
                pet_a_size_encoded,
                pet_b_size_encoded,
                request.pet_a_age,
                request.pet_b_age,
                request.past_interactions,
                request.play_success_rate,
                request.energy_level_diff
            ]])
            
            # Scale features
            features_scaled = self.compatibility_scaler.transform(features)
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features_scaled)
            
            # Make prediction
            with torch.no_grad():
                compatibility_score = self.compatibility_model(features_tensor).item()
            
            # Generate recommendation
            if compatibility_score >= 0.8:
                recommendation = "Excellent match! These pets are highly compatible."
                confidence = 0.9
            elif compatibility_score >= 0.6:
                recommendation = "Good match. These pets should get along well."
                confidence = 0.7
            elif compatibility_score >= 0.4:
                recommendation = "Moderate compatibility. Supervised interaction recommended."
                confidence = 0.5
            else:
                recommendation = "Low compatibility. Careful introduction needed."
                confidence = 0.3
            
            return CompatibilityResponse(
                compatibility_score=round(compatibility_score, 3),
                confidence=round(confidence, 3),
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error in compatibility prediction: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def predict_energy_state(self, request: EnergyRequest) -> EnergyResponse:
        """Predict pet energy state"""
        
        try:
            # Create feature array
            features = np.array([[
                request.activity_duration,
                request.distance_walked,
                request.heart_rate_avg,
                request.rest_periods,
                request.play_time,
                request.social_interactions,
                request.time_outdoors
            ]])
            
            # Scale features
            features_scaled = self.energy_scaler.transform(features)
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features_scaled)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.energy_model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()
            
            # Map prediction to energy state
            energy_states = ["low", "medium", "high"]
            energy_state = energy_states[predicted_class]
            
            # Generate recommendations
            recommendations = []
            
            if energy_state == "low":
                recommendations.extend([
                    "Pet appears tired. Consider rest and relaxation.",
                    "Light activities recommended for the next few hours.",
                    "Ensure proper hydration and nutrition."
                ])
            elif energy_state == "medium":
                recommendations.extend([
                    "Pet has moderate energy levels.",
                    "Suitable for normal activities and light play.",
                    "Good time for training sessions."
                ])
            else:  # high
                recommendations.extend([
                    "Pet has high energy levels!",
                    "Great time for active play and exercise.",
                    "Consider social activities with other pets."
                ])
            
            # Add specific recommendations based on metrics
            if request.activity_duration > 240:  # 4 hours
                recommendations.append("High activity duration detected. Ensure adequate rest.")
            
            if request.heart_rate_avg > 150:
                recommendations.append("Elevated heart rate. Monitor for signs of fatigue.")
            
            if request.social_interactions < 2:
                recommendations.append("Consider increasing social interaction time.")
            
            return EnergyResponse(
                energy_state=energy_state,
                readiness_score=round(confidence, 3),
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in energy state prediction: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# Initialize FastAPI app
app = FastAPI(
    title="PetPath ML API",
    description="Machine Learning inference API for PetPath",
    version="1.0.0"
)

# Initialize inference engine
inference_engine = PetPathMLInference()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "PetPath ML API is running"}


@app.post("/predict/compatibility", response_model=CompatibilityResponse)
async def predict_compatibility(request: CompatibilityRequest):
    """Predict pet compatibility"""
    return inference_engine.predict_compatibility(request)


@app.post("/predict/energy", response_model=EnergyResponse)
async def predict_energy(request: EnergyRequest):
    """Predict pet energy state"""
    return inference_engine.predict_energy(request)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to PetPath ML API",
        "docs": "/docs",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "compatibility": "/predict/compatibility",
            "energy": "/predict/energy"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "infer:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )