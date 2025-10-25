"""
Training script for the Pet Compatibility Model

This script loads synthetic compatibility data and trains a neural network
to predict compatibility scores between pet pairs.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from models.compatibility_model import CompatibilityModel


class CompatibilityDataset(Dataset):
    """PyTorch dataset for pet compatibility data"""

    def __init__(self, df, breed_to_idx, temp_to_idx):
        self.df = df
        self.breed_to_idx = breed_to_idx
        self.temp_to_idx = temp_to_idx

        self.size_map = {'small': 'small', 'medium': 'medium', 'large': 'large'}
        self.energy_map = {'low': 'low', 'medium': 'medium', 'high': 'high'}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Pet 1 features
        pet1_breed = torch.tensor(self.breed_to_idx.get(row['pet1_breed'], 0), dtype=torch.long)
        pet1_size = row['pet1_size']
        pet1_energy = row['pet1_energy']
        pet1_temp = torch.tensor(self.temp_to_idx.get(row['pet1_temperament'], 0), dtype=torch.long)
        pet1_age = torch.tensor(row['pet1_age'], dtype=torch.float32)
        pet1_social = torch.tensor(row['pet1_social'], dtype=torch.float32)
        pet1_weight = torch.tensor(row['pet1_weight'], dtype=torch.float32)

        # Pet 2 features
        pet2_breed = torch.tensor(self.breed_to_idx.get(row['pet2_breed'], 0), dtype=torch.long)
        pet2_size = row['pet2_size']
        pet2_energy = row['pet2_energy']
        pet2_temp = torch.tensor(self.temp_to_idx.get(row['pet2_temperament'], 0), dtype=torch.long)
        pet2_age = torch.tensor(row['pet2_age'], dtype=torch.float32)
        pet2_social = torch.tensor(row['pet2_social'], dtype=torch.float32)
        pet2_weight = torch.tensor(row['pet2_weight'], dtype=torch.float32)

        # Target
        target = torch.tensor(row['compatibility_score'], dtype=torch.float32)

        return {
            'pet1_breed': pet1_breed,
            'pet1_size': pet1_size,
            'pet1_energy': pet1_energy,
            'pet1_temp': pet1_temp,
            'pet1_age': pet1_age,
            'pet1_social': pet1_social,
            'pet1_weight': pet1_weight,
            'pet2_breed': pet2_breed,
            'pet2_size': pet2_size,
            'pet2_energy': pet2_energy,
            'pet2_temp': pet2_temp,
            'pet2_age': pet2_age,
            'pet2_social': pet2_social,
            'pet2_weight': pet2_weight,
            'target': target
        }


def collate_fn(batch):
    """Custom collate function to handle batching"""
    return {
        'pet1_breed': torch.stack([item['pet1_breed'] for item in batch]),
        'pet1_size': [item['pet1_size'] for item in batch],  # Keep as list of strings
        'pet1_energy': [item['pet1_energy'] for item in batch],  # Keep as list of strings
        'pet1_temp': torch.stack([item['pet1_temp'] for item in batch]),
        'pet1_age': torch.stack([item['pet1_age'] for item in batch]),
        'pet1_social': torch.stack([item['pet1_social'] for item in batch]),
        'pet1_weight': torch.stack([item['pet1_weight'] for item in batch]),
        'pet2_breed': torch.stack([item['pet2_breed'] for item in batch]),
        'pet2_size': [item['pet2_size'] for item in batch],  # Keep as list of strings
        'pet2_energy': [item['pet2_energy'] for item in batch],  # Keep as list of strings
        'pet2_temp': torch.stack([item['pet2_temp'] for item in batch]),
        'pet2_age': torch.stack([item['pet2_age'] for item in batch]),
        'pet2_social': torch.stack([item['pet2_social'] for item in batch]),
        'pet2_weight': torch.stack([item['pet2_weight'] for item in batch]),
        'target': torch.stack([item['target'] for item in batch])
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        # Move to device
        pet1_breed = batch['pet1_breed'].to(device)
        pet1_size = batch['pet1_size']
        pet1_energy = batch['pet1_energy']
        pet1_temp = batch['pet1_temp'].to(device)
        pet1_age = batch['pet1_age'].to(device)
        pet1_social = batch['pet1_social'].to(device)
        pet1_weight = batch['pet1_weight'].to(device)

        pet2_breed = batch['pet2_breed'].to(device)
        pet2_size = batch['pet2_size']
        pet2_energy = batch['pet2_energy']
        pet2_temp = batch['pet2_temp'].to(device)
        pet2_age = batch['pet2_age'].to(device)
        pet2_social = batch['pet2_social'].to(device)
        pet2_weight = batch['pet2_weight'].to(device)

        target = batch['target'].to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(
            pet1_breed, pet1_size, pet1_energy, pet1_temp,
            pet1_age, pet1_social, pet1_weight,
            pet2_breed, pet2_size, pet2_energy, pet2_temp,
            pet2_age, pet2_social, pet2_weight
        )

        # Calculate loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            pet1_breed = batch['pet1_breed'].to(device)
            pet1_size = batch['pet1_size']
            pet1_energy = batch['pet1_energy']
            pet1_temp = batch['pet1_temp'].to(device)
            pet1_age = batch['pet1_age'].to(device)
            pet1_social = batch['pet1_social'].to(device)
            pet1_weight = batch['pet1_weight'].to(device)

            pet2_breed = batch['pet2_breed'].to(device)
            pet2_size = batch['pet2_size']
            pet2_energy = batch['pet2_energy']
            pet2_temp = batch['pet2_temp'].to(device)
            pet2_age = batch['pet2_age'].to(device)
            pet2_social = batch['pet2_social'].to(device)
            pet2_weight = batch['pet2_weight'].to(device)

            target = batch['target'].to(device)

            # Forward pass
            output = model(
                pet1_breed, pet1_size, pet1_energy, pet1_temp,
                pet1_age, pet1_social, pet1_weight,
                pet2_breed, pet2_size, pet2_energy, pet2_temp,
                pet2_age, pet2_social, pet2_weight
            )

            # Calculate loss
            loss = criterion(output, target)

            total_loss += loss.item()
            num_batches += 1

            predictions.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())

    avg_loss = total_loss / num_batches

    # Calculate MAE
    mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))

    return avg_loss, mae


def main():
    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {DEVICE}")

    # Load data
    print("\nLoading data...")
    df = pd.read_csv('ml/data/compatibility_synthetic.csv')
    print(f"Loaded {len(df)} samples")

    # Load encodings
    with open('ml/data/breed_encoding.json', 'r') as f:
        breed_to_idx = json.load(f)

    # Create temperament encoding
    temperaments = df['pet1_temperament'].unique().tolist()
    temp_to_idx = {temp: idx for idx, temp in enumerate(temperaments)}

    print(f"Breeds: {len(breed_to_idx)}")
    print(f"Temperaments: {len(temp_to_idx)}")

    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

    # Create datasets
    train_dataset = CompatibilityDataset(train_df, breed_to_idx, temp_to_idx)
    val_dataset = CompatibilityDataset(val_df, breed_to_idx, temp_to_idx)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Create model
    print("\nCreating model...")
    model = CompatibilityModel(
        n_breeds=len(breed_to_idx),
        n_temperaments=len(temp_to_idx)
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_mae = evaluate(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val MAE: {val_mae:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'ml/models/compatibility_model.pth')
            print(f"  ✅ Saved best model (val_loss={val_loss:.4f})")

    print(f"\n🎉 Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save temperament encoding
    with open('ml/data/temperament_encoding.json', 'w') as f:
        json.dump(temp_to_idx, f, indent=2)
    print(f"✅ Saved temperament encodings to ml/data/temperament_encoding.json")


if __name__ == '__main__':
    main()
