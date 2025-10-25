"""
Graph Neural Network for Pet Social Compatibility

Uses Graph Attention Networks (GAT) to model the pet social graph and predict
compatibility based on:
- Pet features (breed, temperament, age, energy)
- Social graph structure (existing friendships, interactions)
- Temporal interaction history
- Community detection features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm
from torch_geometric.data import Data, Batch
import json
from typing import Dict, List, Optional, Tuple


class PetFeatureEncoder(nn.Module):
    """Encode pet features into a rich embedding"""

    def __init__(
        self,
        n_breeds: int = 15,
        n_temperaments: int = 9,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # Embeddings
        self.breed_embedding = nn.Embedding(n_breeds, embedding_dim)
        self.temperament_embedding = nn.Embedding(n_temperaments, embedding_dim)

        # Feature dimensions
        self.size_dim = 3  # small, medium, large
        self.energy_dim = 3  # low, medium, high
        self.numeric_dim = 4  # age, social_score, weight, activity_level

        # Total input dimension
        input_dim = embedding_dim * 2 + self.size_dim + self.energy_dim + self.numeric_dim

        # Feature encoder MLP
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.size_map = {'small': 0, 'medium': 1, 'large': 2}
        self.energy_map = {'low': 0, 'medium': 1, 'high': 2}

    def encode_categorical(self, value, mapping):
        """One-hot encode categorical value"""
        encoding = torch.zeros(len(mapping))
        if value in mapping:
            encoding[mapping[value]] = 1
        return encoding

    def forward(
        self,
        breed_idx: torch.Tensor,
        temperament_idx: torch.Tensor,
        size: str,
        energy: str,
        age: float,
        social_score: float,
        weight: float,
        activity_level: float,
    ) -> torch.Tensor:
        """Encode pet features into embedding"""
        # Embed categorical features
        breed_emb = self.breed_embedding(breed_idx)
        temp_emb = self.temperament_embedding(temperament_idx)

        # One-hot encode size and energy
        size_enc = self.encode_categorical(size, self.size_map)
        energy_enc = self.encode_categorical(energy, self.energy_map)

        # Normalize numeric features
        age_norm = age / 15.0
        weight_norm = weight / 100.0

        # Concatenate all features
        features = torch.cat([
            breed_emb,
            temp_emb,
            size_enc,
            energy_enc,
            torch.tensor([age_norm, social_score, weight_norm, activity_level])
        ])

        return self.encoder(features)


class GraphAttentionCompatibility(nn.Module):
    """
    Graph Attention Network for pet compatibility prediction.

    Architecture:
    1. Encode pet features into rich embeddings
    2. Apply multiple GAT layers to aggregate neighborhood information
    3. Use attention to learn which connections matter most
    4. Predict compatibility score between any two pets in the graph
    """

    def __init__(
        self,
        n_breeds: int = 15,
        n_temperaments: int = 9,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        gnn_layers: int = 3,
        attention_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Feature encoder
        self.feature_encoder = PetFeatureEncoder(
            n_breeds=n_breeds,
            n_temperaments=n_temperaments,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )

        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First GAT layer
        self.gat_layers.append(
            GATv2Conv(
                hidden_dim,
                hidden_dim // attention_heads,
                heads=attention_heads,
                dropout=dropout,
                edge_dim=16,  # Edge features (interaction strength, recency)
            )
        )
        self.batch_norms.append(BatchNorm(hidden_dim))

        # Middle GAT layers
        for _ in range(gnn_layers - 1):
            self.gat_layers.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // attention_heads,
                    heads=attention_heads,
                    dropout=dropout,
                    edge_dim=16,
                )
            )
            self.batch_norms.append(BatchNorm(hidden_dim))

        # Edge feature encoder (for interaction history)
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, 16),  # interactions_count, last_interaction_days, avg_duration
            nn.ReLU(),
        )

        # Compatibility predictor (combines two pet embeddings)
        self.compatibility_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Community detection head (optional, for clustering)
        self.community_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 8),  # 8 community clusters
        )

    def forward(
        self,
        x: torch.Tensor,  # Node features [num_nodes, feature_dim]
        edge_index: torch.Tensor,  # Graph connectivity [2, num_edges]
        edge_attr: Optional[torch.Tensor] = None,  # Edge features [num_edges, edge_dim]
        batch: Optional[torch.Tensor] = None,  # Batch assignment for multiple graphs
    ) -> torch.Tensor:
        """
        Forward pass through GNN

        Args:
            x: Node feature matrix
            edge_index: Graph connectivity in COO format
            edge_attr: Edge features (interaction history)
            batch: Batch vector for multiple graphs

        Returns:
            Node embeddings after graph convolutions
        """
        # Encode edge features if provided
        if edge_attr is not None:
            edge_features = self.edge_encoder(edge_attr)
        else:
            edge_features = None

        # Apply GAT layers
        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            # Graph attention convolution
            x_new = gat(x, edge_index, edge_attr=edge_features)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=0.2, training=self.training)

            # Residual connection (skip connection)
            if x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new

        return x

    def predict_compatibility(
        self,
        node_embeddings: torch.Tensor,
        pet1_idx: int,
        pet2_idx: int,
    ) -> float:
        """
        Predict compatibility score between two pets

        Args:
            node_embeddings: Graph node embeddings
            pet1_idx: Index of first pet
            pet2_idx: Index of second pet

        Returns:
            Compatibility score [0, 1]
        """
        # Get embeddings for both pets
        pet1_emb = node_embeddings[pet1_idx]
        pet2_emb = node_embeddings[pet2_idx]

        # Concatenate embeddings
        combined = torch.cat([pet1_emb, pet2_emb], dim=0)

        # Predict compatibility
        score = self.compatibility_predictor(combined)

        return score.item()

    def predict_community(
        self,
        node_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Predict community membership for clustering"""
        return self.community_head(node_embeddings)

    def get_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        """
        Extract attention weights from a specific GAT layer
        Useful for interpretability - see which connections the model focuses on
        """
        gat_layer = self.gat_layers[layer_idx]
        return gat_layer(x, edge_index, return_attention_weights=True)[1]


def create_pet_graph(
    pets: List[Dict],
    edges: List[Tuple[int, int]],
    edge_features: Optional[List[List[float]]] = None,
) -> Data:
    """
    Create PyTorch Geometric graph from pet data

    Args:
        pets: List of pet dictionaries with features
        edges: List of (pet1_idx, pet2_idx) tuples
        edge_features: Optional edge features (interaction history)

    Returns:
        PyTorch Geometric Data object
    """
    # Convert edges to tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Extract node features (simplified for demo)
    num_nodes = len(pets)
    node_features = torch.randn(num_nodes, 128)  # Placeholder

    # Convert edge features if provided
    if edge_features:
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        edge_attr = None

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


def load_model(model_path: str, device: str = 'cpu'):
    """Load trained GNN model"""
    model = GraphAttentionCompatibility()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


if __name__ == '__main__':
    # Test model creation
    print("Creating Graph Attention Network for pet compatibility...")
    model = GraphAttentionCompatibility(
        n_breeds=15,
        n_temperaments=9,
        hidden_dim=128,
        gnn_layers=3,
        attention_heads=4,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass with dummy data
    num_nodes = 10
    num_edges = 20

    x = torch.randn(num_nodes, 128)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 3)

    # Forward pass
    embeddings = model(x, edge_index, edge_attr)
    print(f"Output embedding shape: {embeddings.shape}")

    # Test compatibility prediction
    score = model.predict_compatibility(embeddings, 0, 1)
    print(f"Compatibility score between pet 0 and 1: {score:.3f}")

    # Test community detection
    communities = model.predict_community(embeddings)
    print(f"Community predictions shape: {communities.shape}")

    print("\n✅ GNN model created successfully!")
