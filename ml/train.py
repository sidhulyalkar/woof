"""
PetPath ML Training Script
Training models for pet compatibility and energy state prediction
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import mlflow
import mlflow.pytorch
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime


class PetCompatibilityDataset(Dataset):
    """Dataset for pet compatibility prediction"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class CompatibilityModel(nn.Module):
    """Neural network for pet compatibility prediction"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [64, 32, 16]):
        super(CompatibilityModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class EnergyStateModel(nn.Module):
    """Neural network for pet energy state prediction"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32]):
        super(EnergyStateModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 3))  # 3 energy states: low, medium, high
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def generate_dummy_data(n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate dummy data for training"""
    
    # Pet compatibility data
    np.random.seed(42)
    
    compatibility_data = {
        'pet_a_breed': np.random.choice(['Golden Retriever', 'Labrador', 'German Shepherd', 'Bulldog', 'Poodle'], n_samples),
        'pet_b_breed': np.random.choice(['Golden Retriever', 'Labrador', 'German Shepherd', 'Bulldog', 'Poodle'], n_samples),
        'pet_a_age': np.random.randint(1, 15, n_samples),
        'pet_b_age': np.random.randint(1, 15, n_samples),
        'pet_a_size': np.random.choice(['small', 'medium', 'large'], n_samples),
        'pet_b_size': np.random.choice(['small', 'medium', 'large'], n_samples),
        'past_interactions': np.random.randint(0, 20, n_samples),
        'play_success_rate': np.random.uniform(0, 1, n_samples),
        'energy_level_diff': np.random.uniform(-1, 1, n_samples),
        'compatibility_score': np.random.uniform(0, 1, n_samples)
    }
    
    compatibility_df = pd.DataFrame(compatibility_data)
    
    # Energy state data
    energy_data = {
        'activity_duration': np.random.uniform(0, 480, n_samples),  # minutes
        'distance_walked': np.random.uniform(0, 20, n_samples),  # km
        'heart_rate_avg': np.random.randint(60, 180, n_samples),
        'rest_periods': np.random.randint(0, 10, n_samples),
        'play_time': np.random.uniform(0, 180, n_samples),  # minutes
        'social_interactions': np.random.randint(0, 15, n_samples),
        'time_outdoors': np.random.uniform(0, 480, n_samples),  # minutes
        'energy_state': np.random.choice([0, 1, 2], n_samples)  # 0: low, 1: medium, 2: high
    }
    
    energy_df = pd.DataFrame(energy_data)
    
    return compatibility_df, energy_df


def preprocess_compatibility_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess compatibility data for training"""
    
    # Encode categorical variables
    breed_encoder = LabelEncoder()
    size_encoder = LabelEncoder()
    
    all_breeds = list(df['pet_a_breed']) + list(df['pet_b_breed'])
    breed_encoder.fit(all_breeds)
    size_encoder.fit(['small', 'medium', 'large'])
    
    # Transform features
    features = []
    
    # Breed encoding
    pet_a_breed_encoded = breed_encoder.transform(df['pet_a_breed'])
    pet_b_breed_encoded = breed_encoder.transform(df['pet_b_breed'])
    
    # Size encoding
    pet_a_size_encoded = size_encoder.transform(df['pet_a_size'])
    pet_b_size_encoded = size_encoder.transform(df['pet_b_size'])
    
    # Numerical features
    numerical_features = df[['pet_a_age', 'pet_b_age', 'past_interactions', 
                           'play_success_rate', 'energy_level_diff']].values
    
    # Combine all features
    features = np.column_stack([
        pet_a_breed_encoded,
        pet_b_breed_encoded,
        pet_a_size_encoded,
        pet_b_size_encoded,
        numerical_features
    ])
    
    # Labels
    labels = df['compatibility_score'].values
    
    return features, labels


def preprocess_energy_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess energy data for training"""
    
    # Features
    features = df[['activity_duration', 'distance_walked', 'heart_rate_avg',
                  'rest_periods', 'play_time', 'social_interactions', 'time_outdoors']].values
    
    # Labels
    labels = df['energy_state'].values
    
    return features, labels


def train_compatibility_model(df: pd.DataFrame, epochs: int = 50) -> CompatibilityModel:
    """Train the compatibility model"""
    
    print("Training compatibility model...")
    
    # Preprocess data
    features, labels = preprocess_compatibility_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets
    train_dataset = PetCompatibilityDataset(X_train_scaled, y_train)
    test_dataset = PetCompatibilityDataset(X_test_scaled, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = CompatibilityModel(input_size=X_train_scaled.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    with mlflow.start_run(run_name="compatibility_model"):
        mlflow.log_param("model_type", "compatibility")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("input_size", X_train_scaled.shape[1])
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_features).squeeze()
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_features, batch_labels in test_loader:
                    outputs = model(batch_features).squeeze()
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        
        # Log model
        mlflow.pytorch.log_model(model, "compatibility_model")
    
    return model


def train_energy_model(df: pd.DataFrame, epochs: int = 50) -> EnergyStateModel:
    """Train the energy state model"""
    
    print("Training energy state model...")
    
    # Preprocess data
    features, labels = preprocess_energy_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets
    train_dataset = PetCompatibilityDataset(X_train_scaled, y_train)
    test_dataset = PetCompatibilityDataset(X_test_scaled, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = EnergyStateModel(input_size=X_train_scaled.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    with mlflow.start_run(run_name="energy_model"):
        mlflow.log_param("model_type", "energy_state")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("input_size", X_train_scaled.shape[1])
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels.long())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels.long()).sum().item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in test_loader:
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels.long())
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels.long()).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            train_accuracy = 100 * correct / total
            val_accuracy = 100 * val_correct / val_total
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                print(f"Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")
            
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
        
        # Log model
        mlflow.pytorch.log_model(model, "energy_model")
    
    return model


def main():
    """Main training function"""
    
    print("Starting PetPath ML training...")
    
    # Set up MLflow
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("petpath_ml")
    
    # Generate dummy data
    print("Generating dummy data...")
    compatibility_df, energy_df = generate_dummy_data(n_samples=1000)
    
    # Train compatibility model
    compatibility_model = train_compatibility_model(compatibility_df, epochs=50)
    
    # Train energy model
    energy_model = train_energy_model(energy_df, epochs=50)
    
    # Save models
    print("Saving models...")
    torch.save(compatibility_model.state_dict(), "models/compatibility_model.pth")
    torch.save(energy_model.state_dict(), "models/energy_model.pth")
    
    print("Training completed!")


if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    main()