"""
Training script for the Pet Energy State Model

This script loads synthetic energy state data and trains a neural network
to classify pet energy levels (low/medium/high).
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
from sklearn.metrics import classification_report, confusion_matrix
from models.energy_model import EnergyStateModel


class EnergyDataset(Dataset):
    """PyTorch dataset for pet energy state data"""

    def __init__(self, df, breed_to_idx):
        self.df = df
        self.breed_to_idx = breed_to_idx

        self.energy_map = {'low': 'low', 'medium': 'medium', 'high': 'high'}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Features
        age = torch.tensor(row['age'], dtype=torch.float32)
        breed = torch.tensor(self.breed_to_idx.get(row['breed'], 0), dtype=torch.long)
        base_energy_level = row['base_energy_level']
        hours_since = torch.tensor(row['hours_since_last_activity'], dtype=torch.float32)
        distance = torch.tensor(row['total_distance_24h'], dtype=torch.float32)
        duration = torch.tensor(row['total_duration_24h'], dtype=torch.float32)
        num_activities = torch.tensor(row['num_activities_24h'], dtype=torch.float32)
        hour = torch.tensor(row['hour_of_day'], dtype=torch.float32)
        day = torch.tensor(row['day_of_week'], dtype=torch.float32)

        # Target
        target = torch.tensor(row['energy_state_class'], dtype=torch.long)

        return {
            'age': age,
            'breed': breed,
            'base_energy_level': base_energy_level,
            'hours_since_last_activity': hours_since,
            'total_distance_24h': distance,
            'total_duration_24h': duration,
            'num_activities_24h': num_activities,
            'hour_of_day': hour,
            'day_of_week': day,
            'target': target
        }


def collate_fn(batch):
    """Custom collate function to handle batching"""
    return {
        'age': torch.stack([item['age'] for item in batch]),
        'breed': torch.stack([item['breed'] for item in batch]),
        'base_energy_level': [item['base_energy_level'] for item in batch],  # Keep as list of strings
        'hours_since_last_activity': torch.stack([item['hours_since_last_activity'] for item in batch]),
        'total_distance_24h': torch.stack([item['total_distance_24h'] for item in batch]),
        'total_duration_24h': torch.stack([item['total_duration_24h'] for item in batch]),
        'num_activities_24h': torch.stack([item['num_activities_24h'] for item in batch]),
        'hour_of_day': torch.stack([item['hour_of_day'] for item in batch]),
        'day_of_week': torch.stack([item['day_of_week'] for item in batch]),
        'target': torch.stack([item['target'] for item in batch])
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    correct = 0
    total = 0

    for batch in dataloader:
        # Move to device
        age = batch['age'].to(device)
        breed = batch['breed'].to(device)
        base_energy_level = batch['base_energy_level']
        hours_since = batch['hours_since_last_activity'].to(device)
        distance = batch['total_distance_24h'].to(device)
        duration = batch['total_duration_24h'].to(device)
        num_activities = batch['num_activities_24h'].to(device)
        hour = batch['hour_of_day'].to(device)
        day = batch['day_of_week'].to(device)
        target = batch['target'].to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(
            age, breed, base_energy_level, hours_since,
            distance, duration, num_activities, hour, day
        )

        # Calculate loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Calculate accuracy
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return total_loss / num_batches, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            age = batch['age'].to(device)
            breed = batch['breed'].to(device)
            base_energy_level = batch['base_energy_level']
            hours_since = batch['hours_since_last_activity'].to(device)
            distance = batch['total_distance_24h'].to(device)
            duration = batch['total_duration_24h'].to(device)
            num_activities = batch['num_activities_24h'].to(device)
            hour = batch['hour_of_day'].to(device)
            day = batch['day_of_week'].to(device)
            target = batch['target'].to(device)

            # Forward pass
            output = model(
                age, breed, base_energy_level, hours_since,
                distance, duration, num_activities, hour, day
            )

            # Calculate loss
            loss = criterion(output, target)

            total_loss += loss.item()
            num_batches += 1

            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = total_loss / num_batches

    return avg_loss, accuracy, all_predictions, all_targets


def main():
    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {DEVICE}")

    # Load data
    print("\nLoading data...")
    df = pd.read_csv('ml/data/energy_synthetic.csv')
    print(f"Loaded {len(df)} samples")

    # Load encodings
    with open('ml/data/breed_encoding.json', 'r') as f:
        breed_to_idx = json.load(f)

    print(f"Breeds: {len(breed_to_idx)}")
    print(f"Class distribution:")
    print(df['energy_state'].value_counts())

    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['energy_state'])
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

    # Create datasets
    train_dataset = EnergyDataset(train_df, breed_to_idx)
    val_dataset = EnergyDataset(val_df, breed_to_idx)

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
    model = EnergyStateModel(
        n_breeds=len(breed_to_idx),
        n_classes=3
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    print("\nStarting training...")
    best_val_acc = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_preds, val_targets = evaluate(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'ml/models/energy_model.pth')
            print(f"  ✅ Saved best model (val_acc={val_acc:.2f}%)")

    print(f"\n🎉 Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Final evaluation
    print("\nFinal Classification Report:")
    class_names = ['low', 'medium', 'high']
    print(classification_report(val_targets, val_preds, target_names=class_names))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(val_targets, val_preds)
    print(cm)


if __name__ == '__main__':
    main()
