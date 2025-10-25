"""
Training script for SimGNN similarity-based matching model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from models.simgnn import SimGNN, create_pet_graph

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = Path('ml/data')
MODEL_DIR = Path('ml/models/saved')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 60
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5
PATIENCE = 10

# Model architecture
INPUT_DIM = 10
HIDDEN_DIM = 128
NUM_LAYERS = 3
NUM_TENSORS = 16


def load_graph_data():
    """Load the social graph data"""
    print("Loading graph data...")
    pets_df = pd.read_csv(DATA_DIR / 'graph_pets.csv')
    edges_df = pd.read_csv(DATA_DIR / 'graph_edges.csv')
    labels_df = pd.read_csv(DATA_DIR / 'graph_labels.csv')

    print(f"Loaded {len(pets_df)} pets, {len(edges_df)} edges, {len(labels_df)} labels")
    return pets_df, edges_df, labels_df


def create_pet_features(pet, breed_to_idx, temp_to_idx):
    """Create feature vector for a pet"""
    features = []

    # Breed and temperament indices (normalized)
    breed_idx = breed_to_idx[pet['breed']] / len(breed_to_idx)
    temp_idx = temp_to_idx[pet['temperament']] / len(temp_to_idx)
    features.extend([breed_idx, temp_idx])

    # Size one-hot
    size_map = {'small': [1, 0, 0], 'medium': [0, 1, 0], 'large': [0, 0, 1]}
    features.extend(size_map[pet['size']])

    # Energy one-hot
    energy_map = {'low': [1, 0, 0], 'medium': [0, 1, 0], 'high': [0, 0, 1]}
    features.extend(energy_map[pet['energy']])

    # Age and weight (normalized)
    features.append(pet['age'] / 15.0)
    features.append(pet['weight'] / 100.0)

    return torch.tensor(features, dtype=torch.float)


def train_epoch(model, pairs, optimizer, criterion):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    # Process in batches
    for i in range(0, len(pairs), BATCH_SIZE):
        batch_pairs = pairs[i:i+BATCH_SIZE]

        optimizer.zero_grad()

        # Forward pass
        data1_list, data2_list, labels_list = [], [], []

        for data1, data2, label in batch_pairs:
            data1_list.append(data1)
            data2_list.append(data2)
            labels_list.append(label)

        # Batch the graphs
        from torch_geometric.data import Batch
        batch_data1 = Batch.from_data_list(data1_list)
        batch_data2 = Batch.from_data_list(data2_list)
        labels = torch.tensor(labels_list, dtype=torch.float).unsqueeze(1).to(DEVICE)

        # Predict
        predictions = model(batch_data1.to(DEVICE), batch_data2.to(DEVICE))

        # Compute loss
        loss = criterion(predictions, labels)

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, pairs, criterion):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for i in range(0, len(pairs), BATCH_SIZE):
        batch_pairs = pairs[i:i+BATCH_SIZE]

        data1_list, data2_list, labels_list = [], [], []
        for data1, data2, label in batch_pairs:
            data1_list.append(data1)
            data2_list.append(data2)
            labels_list.append(label)

        from torch_geometric.data import Batch
        batch_data1 = Batch.from_data_list(data1_list)
        batch_data2 = Batch.from_data_list(data2_list)
        labels = torch.tensor(labels_list, dtype=torch.float).unsqueeze(1).to(DEVICE)

        predictions = model(batch_data1.to(DEVICE), batch_data2.to(DEVICE))
        loss = criterion(predictions, labels)

        total_loss += loss.item()
        all_preds.extend(predictions.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    # Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    roc_auc = roc_auc_score(all_labels, all_preds)
    avg_precision = average_precision_score(all_labels, all_preds)
    accuracy = ((all_preds > 0.5) == all_labels).mean()

    return {
        'loss': total_loss / (len(pairs) // BATCH_SIZE + 1),
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'accuracy': accuracy,
    }


def train_simgnn():
    """Main training function"""
    print(f"Training SimGNN on device: {DEVICE}")
    print("=" * 80)

    # Load data
    pets_df, edges_df, labels_df = load_graph_data()

    # Create mappings
    breed_to_idx = {breed: idx for idx, breed in enumerate(pets_df['breed'].unique())}
    temp_to_idx = {temp: idx for idx, temp in enumerate(pets_df['temperament'].unique())}

    # Create pet graphs with features
    print("Creating pet graphs...")
    pet_graphs = {}

    for _, pet in pets_df.iterrows():
        pet_id = pet['pet_id']

        # Get friends
        friend_edges = edges_df[(edges_df['pet_a'] == pet_id) | (edges_df['pet_b'] == pet_id)]

        # Create simple star graph (pet at center, friends around)
        pet_features = create_pet_features(pet, breed_to_idx, temp_to_idx)

        # For simplicity, create a single-node graph for now
        # In practice, you'd include friend features
        data = Data(x=pet_features.unsqueeze(0), edge_index=torch.tensor([[0], [0]], dtype=torch.long))

        pet_graphs[pet_id] = data

    # Create training pairs from labels
    print("Creating training pairs...")
    pairs = []

    for _, label_row in labels_df.iterrows():
        pet1_id = label_row['pet_a']
        pet2_id = label_row['pet_b']
        compatibility = label_row['compatibility_score']

        if pet1_id in pet_graphs and pet2_id in pet_graphs:
            pairs.append((
                pet_graphs[pet1_id],
                pet_graphs[pet2_id],
                compatibility
            ))

    # Split data
    train_pairs, temp_pairs = train_test_split(pairs, test_size=0.3, random_state=42)
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.5, random_state=42)

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

    # Initialize model
    model = SimGNN(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_tensors=NUM_TENSORS,
    ).to(DEVICE)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 80)

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    print("\nStarting training...")
    print("=" * 80)

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_pairs, optimizer, criterion)
        val_metrics = evaluate(model, val_pairs, criterion)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_roc_auc': val_metrics['roc_auc'],
            'val_avg_precision': val_metrics['avg_precision'],
            'val_accuracy': val_metrics['accuracy'],
        })

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print()

        if val_metrics['loss'] < best_val_loss - 0.001:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_roc_auc': val_metrics['roc_auc'],
            }, MODEL_DIR / 'simgnn_best.pt')

            print(f"✓ Saved best model (epoch {epoch + 1})")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Final evaluation
    print("\n" + "=" * 80)
    checkpoint = torch.load(MODEL_DIR / 'simgnn_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_pairs, criterion)

    print("\nFinal Test Performance:")
    print(f"  Test Loss: {test_metrics['loss']:.4f}")
    print(f"  Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")

    # Save history
    pd.DataFrame(history).to_csv(MODEL_DIR / 'simgnn_history.csv', index=False)

    final_metrics = {
        'train_epochs': len(history),
        'best_val_loss': best_val_loss,
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'timestamp': datetime.now().isoformat(),
    }

    with open(MODEL_DIR / 'simgnn_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)

    print("\nTraining complete!")
    print("=" * 80)


if __name__ == '__main__':
    train_simgnn()
