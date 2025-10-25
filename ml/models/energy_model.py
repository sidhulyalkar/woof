"""
Pet Energy State Prediction Model

A neural network that classifies a pet's current energy state (low/medium/high)
based on recent activity, time of day, breed characteristics, and age.
"""

import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path


class EnergyStateModel(nn.Module):
    """
    Neural network for predicting pet energy state.

    Input features:
    - age (numeric)
    - breed (categorical, encoded)
    - base_energy_level (categorical: low=1, medium=2, high=3)
    - hours_since_last_activity (numeric)
    - total_distance_24h (numeric, meters)
    - total_duration_24h (numeric, minutes)
    - num_activities_24h (numeric)
    - hour_of_day (numeric, 0-23)
    - day_of_week (numeric, 0-6)

    Output: energy state class (0=low, 1=medium, 2=high)
    """

    def __init__(
        self,
        n_breeds: int = 15,
        embedding_dim: int = 8,
        hidden_dims: list = [128, 64, 32],
        dropout: float = 0.3,
        n_classes: int = 3
    ):
        super().__init__()

        # Embedding for breed
        self.breed_embedding = nn.Embedding(n_breeds, embedding_dim)

        # Input size calculation:
        # breed_emb(8) + base_energy(3) + age(1) + hours_since(1) + distance(1) +
        # duration(1) + num_activities(1) + hour_of_day(1) + day_of_week(1)
        # = 18 features
        input_dim = embedding_dim + 3 + 7

        # Neural network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer for classification
        layers.append(nn.Linear(prev_dim, n_classes))

        self.network = nn.Sequential(*layers)

        # Energy level encoding
        self.energy_map = {'low': 0, 'medium': 1, 'high': 2}

    def encode_categorical(self, value, mapping):
        """One-hot encode a categorical value"""
        encoding = torch.zeros(len(mapping))
        if value in mapping:
            encoding[mapping[value]] = 1
        return encoding

    def forward(
        self,
        age,
        breed,
        base_energy_level,
        hours_since_last_activity,
        total_distance_24h,
        total_duration_24h,
        num_activities_24h,
        hour_of_day,
        day_of_week
    ):
        """
        Forward pass through the network.

        All inputs should be batched tensors.
        """
        batch_size = breed.shape[0]

        # Embed breed
        breed_emb = self.breed_embedding(breed)

        # One-hot encode base energy level
        base_energy_enc_list = []
        for i in range(batch_size):
            enc = self.encode_categorical(base_energy_level[i], self.energy_map)
            base_energy_enc_list.append(enc)
        base_energy_enc = torch.stack(base_energy_enc_list)

        # Normalize numeric features
        age_norm = age / 15.0  # Max age ~15
        hours_norm = torch.clamp(hours_since_last_activity / 24.0, 0, 1)  # Max 24 hours
        distance_norm = torch.clamp(total_distance_24h / 10000.0, 0, 1)  # Max 10km
        duration_norm = torch.clamp(total_duration_24h / 240.0, 0, 1)  # Max 4 hours
        activities_norm = torch.clamp(num_activities_24h / 10.0, 0, 1)  # Max 10 activities
        hour_norm = hour_of_day / 24.0  # 0-1
        day_norm = day_of_week / 7.0  # 0-1

        # Concatenate all features
        features = torch.cat([
            breed_emb,
            base_energy_enc,
            age_norm.unsqueeze(1),
            hours_norm.unsqueeze(1),
            distance_norm.unsqueeze(1),
            duration_norm.unsqueeze(1),
            activities_norm.unsqueeze(1),
            hour_norm.unsqueeze(1),
            day_norm.unsqueeze(1)
        ], dim=1)

        # Forward through network
        logits = self.network(features)

        return logits

    def predict(self, features_dict, breed_to_idx):
        """
        Make a prediction for a single pet's energy state.

        Args:
            features_dict: dict with keys:
                - age, breed, base_energy_level, hours_since_last_activity,
                  total_distance_24h, total_duration_24h, num_activities_24h,
                  hour_of_day, day_of_week
            breed_to_idx: mapping from breed name to index

        Returns:
            predicted_class (0, 1, or 2), probabilities, class_name
        """
        self.eval()

        with torch.no_grad():
            # Convert to tensors (batch size 1)
            age = torch.tensor([features_dict['age']], dtype=torch.float32)
            breed = torch.tensor([breed_to_idx.get(features_dict['breed'], 0)])
            base_energy_level = torch.tensor([features_dict['base_energy_level']])
            hours_since = torch.tensor([features_dict['hours_since_last_activity']], dtype=torch.float32)
            distance = torch.tensor([features_dict['total_distance_24h']], dtype=torch.float32)
            duration = torch.tensor([features_dict['total_duration_24h']], dtype=torch.float32)
            num_activities = torch.tensor([features_dict['num_activities_24h']], dtype=torch.float32)
            hour = torch.tensor([features_dict['hour_of_day']], dtype=torch.float32)
            day = torch.tensor([features_dict['day_of_week']], dtype=torch.float32)

            logits = self.forward(
                age, breed, base_energy_level, hours_since,
                distance, duration, num_activities, hour, day
            )

            # Get probabilities
            probs = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()

            # Map to class names
            class_names = ['low', 'medium', 'high']
            class_name = class_names[predicted_class]

            return predicted_class, probs[0].tolist(), class_name

    def predict_proba(self, features_dict, breed_to_idx):
        """Get probability distribution over energy states"""
        _, probs, _ = self.predict(features_dict, breed_to_idx)
        return probs


def load_model(model_path: str, encodings_path: str = None):
    """
    Load a trained energy state model.

    Args:
        model_path: path to saved model weights
        encodings_path: path to encodings JSON (breed_encoding.json)

    Returns:
        model, breed_to_idx
    """
    # Load encodings
    if encodings_path:
        with open(encodings_path, 'r') as f:
            breed_to_idx = json.load(f)
    else:
        # Default encodings
        breed_to_idx = {}

    # Create model
    model = EnergyStateModel(n_breeds=len(breed_to_idx))

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    return model, breed_to_idx


if __name__ == '__main__':
    # Test model creation
    print("Creating energy state model...")
    model = EnergyStateModel()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 4
    age = torch.rand(batch_size) * 15
    breed = torch.randint(0, 15, (batch_size,))
    base_energy_level = torch.tensor(['high', 'low', 'medium', 'high'])
    hours_since = torch.rand(batch_size) * 12
    distance = torch.rand(batch_size) * 5000
    duration = torch.rand(batch_size) * 120
    num_activities = torch.randint(0, 5, (batch_size,)).float()
    hour = torch.randint(0, 24, (batch_size,)).float()
    day = torch.randint(0, 7, (batch_size,)).float()

    output = model(
        age, breed, base_energy_level, hours_since,
        distance, duration, num_activities, hour, day
    )

    print(f"\nTest forward pass:")
    print(f"Input batch size: {batch_size}")
    print(f"Output shape: {output.shape}")
    print(f"Logits: {output}")

    # Test with softmax
    probs = torch.softmax(output, dim=1)
    predictions = torch.argmax(probs, dim=1)
    print(f"Probabilities: {probs}")
    print(f"Predictions (0=low, 1=medium, 2=high): {predictions}")
