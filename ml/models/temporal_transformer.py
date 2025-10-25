"""
Temporal Transformer for Pet Activity Prediction

Uses Transformer architecture to model temporal patterns in pet activities:
- Activity sequences (walk → play → rest patterns)
- Time-of-day preferences
- Energy level fluctuations
- Social interaction timing
- Recommendation of optimal activity times

Architecture inspired by:
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Temporal Fusion Transformers" (Lim et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences
    Encodes both absolute time and relative time differences
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TimeEncoding(nn.Module):
    """
    Encode time-of-day and day-of-week information
    Uses cyclical encoding (sin/cos) to capture periodicity
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Learnable projection for time features
        self.time_proj = nn.Linear(4, d_model)  # hour_sin, hour_cos, day_sin, day_cos

    def forward(self, hour_of_day: torch.Tensor, day_of_week: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hour_of_day: Tensor of shape [batch_size, seq_len] with hours (0-23)
            day_of_week: Tensor of shape [batch_size, seq_len] with days (0-6)
        """
        # Convert to radians
        hour_rad = (hour_of_day / 24.0) * 2 * math.pi
        day_rad = (day_of_week / 7.0) * 2 * math.pi

        # Cyclical encoding
        hour_sin = torch.sin(hour_rad)
        hour_cos = torch.cos(hour_rad)
        day_sin = torch.sin(day_rad)
        day_cos = torch.cos(day_rad)

        # Stack and project
        time_features = torch.stack([hour_sin, hour_cos, day_sin, day_cos], dim=-1)
        return self.time_proj(time_features)


class TemporalMultiHeadAttention(nn.Module):
    """
    Multi-head attention with temporal bias
    Learns to attend more to recent events while maintaining long-term memory
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_relative_position: int = 32,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_relative_position = max_relative_position

        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Relative position embeddings
        self.relative_position_k = nn.Embedding(
            2 * max_relative_position + 1, self.d_k
        )
        self.relative_position_v = nn.Embedding(
            2 * max_relative_position + 1, self.d_k
        )

        self.dropout = nn.Dropout(dropout)

    def _get_relative_positions(self, seq_len: int) -> torch.Tensor:
        """Compute relative position matrix"""
        range_vec = torch.arange(seq_len)
        range_mat = range_vec.unsqueeze(0).expand(seq_len, seq_len)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        return distance_mat_clipped + self.max_relative_position

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = query.shape

        # Project to Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape for multi-head attention
        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Add relative position bias
        relative_positions = self._get_relative_positions(seq_len).to(query.device)
        rel_pos_k = self.relative_position_k(relative_positions)
        rel_pos_scores = torch.matmul(Q, rel_pos_k.transpose(-2, -1))
        scores = scores + rel_pos_scores

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Add relative position bias to values
        rel_pos_v = self.relative_position_v(relative_positions)
        rel_pos_context = torch.matmul(attn_weights, rel_pos_v)
        context = context + rel_pos_context

        # Reshape and project output
        context = rearrange(context, 'b h n d -> b n (h d)')
        output = self.out_proj(context)

        return output, attn_weights


class TransformerBlock(nn.Module):
    """Single Transformer encoder block with temporal attention"""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Temporal multi-head attention
        self.attention = TemporalMultiHeadAttention(
            d_model, num_heads, dropout
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x, attn_weights


class TemporalActivityTransformer(nn.Module):
    """
    Temporal Transformer for pet activity prediction

    Input: Sequence of past activities with features:
    - Activity type (walk, play, rest, etc.)
    - Duration, distance, intensity
    - Time of day, day of week
    - Energy level before/after
    - Social context (alone, with other pets, with owner)

    Output:
    - Predicted next activity type
    - Predicted optimal time for next activity
    - Predicted energy level
    - Recommended activity suggestions
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 100,
        num_activity_types: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Activity type embedding
        self.activity_embedding = nn.Embedding(num_activity_types, d_model)

        # Feature encoder for continuous features
        self.feature_encoder = nn.Linear(8, d_model)  # duration, distance, intensity, etc.

        # Time encoding
        self.time_encoding = TimeEncoding(d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Prediction heads
        self.activity_predictor = nn.Linear(d_model, num_activity_types)
        self.energy_predictor = nn.Linear(d_model, 3)  # low, medium, high
        self.duration_predictor = nn.Linear(d_model, 1)
        self.optimal_time_predictor = nn.Linear(d_model, 24)  # hour probabilities

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        activity_types: torch.Tensor,  # [batch, seq_len]
        features: torch.Tensor,  # [batch, seq_len, 8]
        hour_of_day: torch.Tensor,  # [batch, seq_len]
        day_of_week: torch.Tensor,  # [batch, seq_len]
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through temporal transformer

        Returns:
            next_activity: Predicted next activity type logits
            energy_state: Predicted energy state logits
            duration: Predicted activity duration
            optimal_time: Probabilities for each hour
            attention_maps: Attention weights for interpretability
        """
        batch_size, seq_len = activity_types.shape

        # Embed activity types
        act_emb = self.activity_embedding(activity_types)

        # Encode continuous features
        feat_emb = self.feature_encoder(features)

        # Encode time information
        time_emb = self.time_encoding(hour_of_day, day_of_week)

        # Combine embeddings
        x = act_emb + feat_emb + time_emb

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through transformer blocks
        attention_maps = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x, mask)
            attention_maps.append(attn_weights)

        # Use last token for prediction (or mean pooling)
        last_hidden = x[:, -1, :]

        # Make predictions
        next_activity = self.activity_predictor(last_hidden)
        energy_state = self.energy_predictor(last_hidden)
        duration = self.duration_predictor(last_hidden)
        optimal_time = self.optimal_time_predictor(last_hidden)

        # Stack attention maps
        attention_maps = torch.stack(attention_maps, dim=1)

        return next_activity, energy_state, duration, optimal_time, attention_maps

    def predict_next_activities(
        self,
        activity_sequence: torch.Tensor,
        features: torch.Tensor,
        hour_of_day: torch.Tensor,
        day_of_week: torch.Tensor,
        top_k: int = 5,
    ):
        """Get top-k recommended activities with probabilities"""
        self.eval()

        with torch.no_grad():
            next_act, energy, duration, time_probs, _ = self.forward(
                activity_sequence, features, hour_of_day, day_of_week
            )

            # Get top-k activities
            probs = F.softmax(next_act, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

            # Get optimal time
            time_dist = F.softmax(time_probs, dim=-1)
            optimal_hour = torch.argmax(time_dist, dim=-1)

            # Get energy prediction
            energy_probs = F.softmax(energy, dim=-1)
            predicted_energy = torch.argmax(energy_probs, dim=-1)

            return {
                'top_activities': top_indices.cpu().tolist(),
                'activity_probs': top_probs.cpu().tolist(),
                'optimal_hour': optimal_hour.cpu().tolist(),
                'predicted_energy': predicted_energy.cpu().tolist(),
                'expected_duration': duration.cpu().tolist(),
            }


if __name__ == '__main__':
    print("Creating Temporal Transformer for activity prediction...")

    model = TemporalActivityTransformer(
        d_model=256,
        num_layers=6,
        num_heads=8,
        d_ff=1024,
        num_activity_types=10,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    batch_size = 4
    seq_len = 20

    activity_types = torch.randint(0, 10, (batch_size, seq_len))
    features = torch.randn(batch_size, seq_len, 8)
    hour_of_day = torch.randint(0, 24, (batch_size, seq_len)).float()
    day_of_week = torch.randint(0, 7, (batch_size, seq_len)).float()

    next_act, energy, duration, optimal_time, attention = model(
        activity_types, features, hour_of_day, day_of_week
    )

    print(f"\nOutput shapes:")
    print(f"  Next activity logits: {next_act.shape}")
    print(f"  Energy state logits: {energy.shape}")
    print(f"  Duration prediction: {duration.shape}")
    print(f"  Optimal time distribution: {optimal_time.shape}")
    print(f"  Attention maps: {attention.shape}")

    # Test prediction
    predictions = model.predict_next_activities(
        activity_types, features, hour_of_day, day_of_week
    )
    print(f"\nPredictions:")
    print(f"  Top activities: {predictions['top_activities']}")
    print(f"  Optimal hours: {predictions['optimal_hour']}")

    print("\n✅ Temporal Transformer created successfully!")
