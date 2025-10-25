"""
SimGNN: Similarity-based Graph Neural Network for Pet Matching
Uses graph matching networks to compute similarity between pet graphs

Based on "SimGNN: A Neural Network Approach to Fast Graph Similarity Computation"
Extended with attention mechanisms and multi-scale matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch


class GraphEncoder(nn.Module):
    """Encodes individual pet graphs into embeddings"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, batch):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]

        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        return x


class AttentionModule(nn.Module):
    """Attention mechanism for graph-level matching"""

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.context_weight = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings, batch):
        """
        Compute attention-weighted graph representation

        Args:
            embeddings: Node embeddings [num_nodes, hidden_dim]
            batch: Batch assignment [num_nodes]

        Returns:
            Graph embedding [batch_size, hidden_dim]
        """
        # Transform embeddings
        transformed = torch.tanh(self.context_weight(embeddings))

        # Compute attention scores
        scores = self.attention(transformed)  # [num_nodes, 1]

        # Apply softmax per graph
        unique_batch = batch.unique()
        weighted_embeddings = []

        for i in unique_batch:
            mask = batch == i
            graph_scores = scores[mask]
            graph_embeddings = embeddings[mask]

            # Softmax over nodes in this graph
            attn_weights = F.softmax(graph_scores, dim=0)

            # Weighted sum
            graph_embedding = (attn_weights * graph_embeddings).sum(dim=0)
            weighted_embeddings.append(graph_embedding)

        return torch.stack(weighted_embeddings)


class NeuralTensorNetwork(nn.Module):
    """Neural Tensor Network for computing similarity scores"""

    def __init__(self, hidden_dim: int, num_tensors: int = 16):
        super().__init__()

        self.num_tensors = num_tensors

        # Tensor layers for bilinear interactions
        self.tensors = nn.Parameter(torch.randn(num_tensors, hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.randn(num_tensors))

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(num_tensors, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, embedding1, embedding2):
        """
        Compute similarity score between two graph embeddings

        Args:
            embedding1: First graph embedding [batch_size, hidden_dim]
            embedding2: Second graph embedding [batch_size, hidden_dim]

        Returns:
            Similarity scores [batch_size, 1]
        """
        batch_size = embedding1.shape[0]

        # Compute tensor products
        tensor_products = []
        for k in range(self.num_tensors):
            # e1^T * M_k * e2
            # For each sample in batch: e1 @ M_k @ e2
            M_e2 = torch.matmul(embedding2, self.tensors[k].t())  # [batch, hidden_dim]
            product = (embedding1 * M_e2).sum(dim=1)  # [batch]

            tensor_products.append(product + self.bias[k])

        tensor_features = torch.stack(tensor_products, dim=1)  # [batch, num_tensors]

        # Pass through MLP
        similarity = self.mlp(tensor_features)

        return similarity


class SimGNN(nn.Module):
    """
    Complete SimGNN model for pet matching
    Computes similarity between pet social graphs
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_tensors: int = 16,
    ):
        super().__init__()

        self.encoder = GraphEncoder(input_dim, hidden_dim, num_layers)
        self.attention = AttentionModule(hidden_dim)
        self.ntn = NeuralTensorNetwork(hidden_dim, num_tensors)

        # Histogram features for fine-grained matching
        self.histogram_bins = 16
        self.histogram_mlp = nn.Sequential(
            nn.Linear(self.histogram_bins, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(16 + 1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def compute_histogram(self, embeddings1, embeddings2, batch1, batch2):
        """
        Compute histogram of pairwise node similarities

        Args:
            embeddings1: Node embeddings from graph 1
            embeddings2: Node embeddings from graph 2
            batch1: Batch assignment for graph 1
            batch2: Batch assignment for graph 2

        Returns:
            Histogram features [batch_size, histogram_bins]
        """
        unique_batch = batch1.unique()
        histograms = []

        for i in unique_batch:
            # Get nodes from both graphs in this batch
            mask1 = batch1 == i
            mask2 = batch2 == i

            nodes1 = embeddings1[mask1]  # [n1, hidden_dim]
            nodes2 = embeddings2[mask2]  # [n2, hidden_dim]

            # Compute pairwise cosine similarities
            similarities = F.cosine_similarity(
                nodes1.unsqueeze(1),  # [n1, 1, hidden_dim]
                nodes2.unsqueeze(0),  # [1, n2, hidden_dim]
                dim=2
            )  # [n1, n2]

            # Flatten and compute histogram
            flat_sims = similarities.flatten()
            hist = torch.histc(flat_sims, bins=self.histogram_bins, min=-1, max=1)
            hist = hist / (hist.sum() + 1e-8)  # Normalize

            histograms.append(hist)

        return torch.stack(histograms)

    def forward(self, data1, data2):
        """
        Forward pass for pair of graphs

        Args:
            data1: First graph batch (Data object with x, edge_index, batch)
            data2: Second graph batch (Data object with x, edge_index, batch)

        Returns:
            Similarity scores [batch_size, 1]
        """
        # Encode both graphs
        embeddings1 = self.encoder(data1.x, data1.edge_index, data1.batch)
        embeddings2 = self.encoder(data2.x, data2.edge_index, data2.batch)

        # Attention-weighted graph embeddings
        graph_embedding1 = self.attention(embeddings1, data1.batch)
        graph_embedding2 = self.attention(embeddings2, data2.batch)

        # Neural Tensor Network similarity
        ntn_similarity = self.ntn(graph_embedding1, graph_embedding2)

        # Histogram features
        histogram_features = self.compute_histogram(
            embeddings1, embeddings2, data1.batch, data2.batch
        )
        histogram_embedding = self.histogram_mlp(histogram_features)

        # Fuse features
        combined = torch.cat([ntn_similarity, histogram_embedding], dim=1)
        final_similarity = self.fusion(combined)

        return final_similarity


class SimGNNWithAttention(nn.Module):
    """
    Enhanced SimGNN with cross-graph attention
    Allows nodes from different graphs to attend to each other
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()

        # Graph encoder with attention
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.gat_layers.append(GATv2Conv(input_dim, hidden_dim // num_heads, heads=num_heads))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.gat_layers.append(GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Cross-graph attention
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # Similarity predictor
        self.similarity_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),  # Concat: g1, g2, g1*g2, |g1-g2|
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, data1, data2):
        """
        Forward pass with cross-graph attention

        Args:
            data1: First graph batch
            data2: Second graph batch

        Returns:
            Similarity scores [batch_size, 1]
        """
        # Encode both graphs
        x1, x2 = data1.x, data2.x

        for gat, bn in zip(self.gat_layers, self.batch_norms):
            x1 = gat(x1, data1.edge_index)
            x1 = bn(x1)
            x1 = F.relu(x1)

            x2 = gat(x2, data2.edge_index)
            x2 = bn(x2)
            x2 = F.relu(x2)

        # Pool to graph level
        g1 = global_mean_pool(x1, data1.batch) + global_max_pool(x1, data1.batch)
        g2 = global_mean_pool(x2, data2.batch) + global_max_pool(x2, data2.batch)

        # Cross-graph attention (allow graphs to attend to each other)
        g1_attn, _ = self.cross_attention(
            g1.unsqueeze(1), g2.unsqueeze(1), g2.unsqueeze(1)
        )
        g2_attn, _ = self.cross_attention(
            g2.unsqueeze(1), g1.unsqueeze(1), g1.unsqueeze(1)
        )

        g1_attn = g1_attn.squeeze(1)
        g2_attn = g2_attn.squeeze(1)

        # Combine features: original, attended, element-wise product, absolute difference
        combined = torch.cat([
            g1_attn,
            g2_attn,
            g1_attn * g2_attn,
            torch.abs(g1_attn - g2_attn)
        ], dim=1)

        # Predict similarity
        similarity = self.similarity_mlp(combined)

        return similarity


def create_pet_graph(pet_features, friends_features, friend_edges):
    """
    Create a PyG Data object representing a pet and their friend network

    Args:
        pet_features: Features of the main pet [feature_dim]
        friends_features: Features of friend pets [num_friends, feature_dim]
        friend_edges: Edge list between friends [2, num_edges]

    Returns:
        Data object representing the pet's social graph
    """
    # Combine pet and friend features
    all_features = torch.cat([pet_features.unsqueeze(0), friends_features], dim=0)

    # Adjust edge indices (shift by 1 since pet is node 0)
    adjusted_edges = friend_edges + 1

    # Add edges from pet to all friends
    pet_to_friends = torch.tensor([
        [0] * friends_features.shape[0],
        list(range(1, friends_features.shape[0] + 1))
    ], dtype=torch.long)

    # Combine all edges
    all_edges = torch.cat([pet_to_friends, adjusted_edges], dim=1)

    # Make edges bidirectional
    all_edges = torch.cat([all_edges, all_edges.flip(0)], dim=1)

    return Data(x=all_features, edge_index=all_edges)


if __name__ == '__main__':
    # Test SimGNN
    model = SimGNN(input_dim=10, hidden_dim=64, num_layers=2)

    # Create dummy data
    data1 = Data(
        x=torch.randn(20, 10),
        edge_index=torch.randint(0, 20, (2, 40)),
        batch=torch.zeros(20, dtype=torch.long)
    )

    data2 = Data(
        x=torch.randn(15, 10),
        edge_index=torch.randint(0, 15, (2, 30)),
        batch=torch.zeros(15, dtype=torch.long)
    )

    # Forward pass
    similarity = model(data1, data2)
    print(f"Similarity score: {similarity.item():.4f}")

    # Test enhanced version
    model_enhanced = SimGNNWithAttention(input_dim=10, hidden_dim=64, num_layers=2)
    similarity_enhanced = model_enhanced(data1, data2)
    print(f"Enhanced similarity score: {similarity_enhanced.item():.4f}")
