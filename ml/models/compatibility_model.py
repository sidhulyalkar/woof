"""
Pet Compatibility Prediction Model

A neural network that predicts compatibility scores between two pets
based on their characteristics (breed, size, energy, temperament, age, etc.)
"""

import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path


class CompatibilityModel(nn.Module):
    """
    Neural network for predicting pet compatibility scores.

    Input features (per pet):
    - breed (categorical, encoded)
    - size (categorical: small=1, medium=2, large=3)
    - energy (categorical: low=1, medium=2, high=3)
    - temperament (categorical, encoded)
    - age (numeric)
    - social score (numeric 0-1)
    - weight (numeric)

    Output: compatibility score (0-1)
    """

    def __init__(
        self,
        n_breeds: int = 15,
        n_temperaments: int = 9,
        embedding_dim: int = 8,
        hidden_dims: list = [128, 64, 32],
        dropout: float = 0.3
    ):
        super().__init__()

        # Embeddings for categorical features
        self.breed_embedding = nn.Embedding(n_breeds, embedding_dim)
        self.temperament_embedding = nn.Embedding(n_temperaments, embedding_dim)

        # Input size calculation:
        # Per pet: breed_emb(8) + size(3) + energy(3) + temp_emb(8) + age(1) + social(1) + weight(1)
        # = 25 features per pet
        # For two pets: 25 * 2 = 50 features
        input_dim = (embedding_dim * 2 + 9) * 2  # breed_emb + temp_emb + size(3) + energy(3) + numeric(3)

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

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Output in range 0-1

        self.network = nn.Sequential(*layers)

        # Store encodings
        self.size_map = {'small': 0, 'medium': 1, 'large': 2}
        self.energy_map = {'low': 0, 'medium': 1, 'high': 2}

    def encode_categorical(self, value, mapping):
        """One-hot encode a categorical value"""
        encoding = torch.zeros(len(mapping))
        if value in mapping:
            encoding[mapping[value]] = 1
        return encoding

    def encode_pet(self, breed_idx, size, energy, temp_idx, age, social, weight):
        """Encode a single pet's features"""
        # Embed breed and temperament
        breed_emb = self.breed_embedding(breed_idx)
        temp_emb = self.temperament_embedding(temp_idx)

        # One-hot encode size and energy
        size_enc = self.encode_categorical(size, self.size_map)
        energy_enc = self.encode_categorical(energy, self.energy_map)

        # Normalize numeric features
        age_norm = age / 15.0  # Max age ~15
        weight_norm = weight / 100.0  # Max weight ~100

        # Concatenate all features
        features = torch.cat([
            breed_emb,
            size_enc,
            energy_enc,
            temp_emb,
            torch.tensor([age_norm, social, weight_norm])
        ])

        return features

    def forward(
        self,
        pet1_breed, pet1_size, pet1_energy, pet1_temp,
        pet1_age, pet1_social, pet1_weight,
        pet2_breed, pet2_size, pet2_energy, pet2_temp,
        pet2_age, pet2_social, pet2_weight
    ):
        """
        Forward pass through the network.

        All inputs should be batched tensors.
        """
        batch_size = pet1_breed.shape[0]

        # Process each pet in the batch
        pet1_features_list = []
        pet2_features_list = []

        for i in range(batch_size):
            pet1_feat = self.encode_pet(
                pet1_breed[i], pet1_size[i], pet1_energy[i], pet1_temp[i],
                pet1_age[i], pet1_social[i], pet1_weight[i]
            )
            pet2_feat = self.encode_pet(
                pet2_breed[i], pet2_size[i], pet2_energy[i], pet2_temp[i],
                pet2_age[i], pet2_social[i], pet2_weight[i]
            )

            pet1_features_list.append(pet1_feat)
            pet2_features_list.append(pet2_feat)

        # Stack into batch
        pet1_features = torch.stack(pet1_features_list)
        pet2_features = torch.stack(pet2_features_list)

        # Concatenate both pets
        combined = torch.cat([pet1_features, pet2_features], dim=1)

        # Forward through network
        output = self.network(combined)

        return output.squeeze()

    def predict(self, pet1_dict, pet2_dict, breed_to_idx, temp_to_idx):
        """
        Make a prediction for a single pet pair.

        Args:
            pet1_dict: dict with keys: breed, size, energy, temperament, age, social, weight
            pet2_dict: dict with keys: breed, size, energy, temperament, age, social, weight
            breed_to_idx: mapping from breed name to index
            temp_to_idx: mapping from temperament to index

        Returns:
            compatibility score (0-1)
        """
        self.eval()

        with torch.no_grad():
            # Convert to tensors (batch size 1)
            pet1_breed = torch.tensor([breed_to_idx.get(pet1_dict['breed'], 0)])
            pet1_size = torch.tensor([pet1_dict['size']])
            pet1_energy = torch.tensor([pet1_dict['energy']])
            pet1_temp = torch.tensor([temp_to_idx.get(pet1_dict['temperament'], 0)])
            pet1_age = torch.tensor([pet1_dict['age']], dtype=torch.float32)
            pet1_social = torch.tensor([pet1_dict['social']], dtype=torch.float32)
            pet1_weight = torch.tensor([pet1_dict['weight']], dtype=torch.float32)

            pet2_breed = torch.tensor([breed_to_idx.get(pet2_dict['breed'], 0)])
            pet2_size = torch.tensor([pet2_dict['size']])
            pet2_energy = torch.tensor([pet2_dict['energy']])
            pet2_temp = torch.tensor([temp_to_idx.get(pet2_dict['temperament'], 0)])
            pet2_age = torch.tensor([pet2_dict['age']], dtype=torch.float32)
            pet2_social = torch.tensor([pet2_dict['social']], dtype=torch.float32)
            pet2_weight = torch.tensor([pet2_dict['weight']], dtype=torch.float32)

            score = self.forward(
                pet1_breed, pet1_size, pet1_energy, pet1_temp,
                pet1_age, pet1_social, pet1_weight,
                pet2_breed, pet2_size, pet2_energy, pet2_temp,
                pet2_age, pet2_social, pet2_weight
            )

            return score.item()


def load_model(model_path: str, encodings_path: str = None):
    """
    Load a trained compatibility model.

    Args:
        model_path: path to saved model weights
        encodings_path: path to encodings JSON (breed_encoding.json)

    Returns:
        model, breed_to_idx, temp_to_idx
    """
    # Load encodings
    if encodings_path:
        with open(encodings_path, 'r') as f:
            breed_to_idx = json.load(f)
    else:
        # Default encodings
        breed_to_idx = {}

    # Create temperament encoding
    temperaments = ['friendly', 'calm', 'energetic', 'protective', 'nervous',
                   'playful', 'intelligent', 'stubborn', 'bold']
    temp_to_idx = {temp: idx for idx, temp in enumerate(temperaments)}

    # Create model
    model = CompatibilityModel(
        n_breeds=len(breed_to_idx),
        n_temperaments=len(temp_to_idx)
    )

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    return model, breed_to_idx, temp_to_idx


if __name__ == '__main__':
    # Test model creation
    print("Creating compatibility model...")
    model = CompatibilityModel()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 4
    pet1_breed = torch.randint(0, 15, (batch_size,))
    pet1_size = torch.tensor(['small', 'medium', 'large', 'medium'])
    pet1_energy = torch.tensor(['high', 'low', 'medium', 'high'])
    pet1_temp = torch.randint(0, 9, (batch_size,))
    pet1_age = torch.rand(batch_size) * 15
    pet1_social = torch.rand(batch_size)
    pet1_weight = torch.rand(batch_size) * 100

    pet2_breed = torch.randint(0, 15, (batch_size,))
    pet2_size = torch.tensor(['medium', 'large', 'small', 'large'])
    pet2_energy = torch.tensor(['medium', 'high', 'low', 'medium'])
    pet2_temp = torch.randint(0, 9, (batch_size,))
    pet2_age = torch.rand(batch_size) * 15
    pet2_social = torch.rand(batch_size)
    pet2_weight = torch.rand(batch_size) * 100

    output = model(
        pet1_breed, pet1_size, pet1_energy, pet1_temp,
        pet1_age, pet1_social, pet1_weight,
        pet2_breed, pet2_size, pet2_energy, pet2_temp,
        pet2_age, pet2_social, pet2_weight
    )

    print(f"\nTest forward pass:")
    print(f"Input batch size: {batch_size}")
    print(f"Output shape: {output.shape}")
    print(f"Sample predictions: {output}")
