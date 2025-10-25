"""
Graph Diffusion Model for Pet Matching
Uses denoising diffusion probabilistic models to generate optimal matchings

Inspired by:
- "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- "Score-Based Generative Modeling through SDEs" (Song et al., 2021)
- Applied to graph matching problem
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for timesteps"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeConditionedGNN(nn.Module):
    """
    GNN conditioned on diffusion timestep
    Used for denoising in the diffusion process
    """

    def __init__(self, input_dim: int, hidden_dim: int, time_dim: int, num_layers: int = 3):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.time_proj = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.time_proj.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, edge_index, t):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            t: Timestep [batch_size]

        Returns:
            Denoised features [num_nodes, input_dim]
        """
        # Project input
        h = self.input_proj(x)

        # Time embedding
        t_emb = self.time_mlp(t)

        # GNN layers with time conditioning
        for conv, time_proj, norm in zip(self.convs, self.time_proj, self.norms):
            # Message passing
            h_new = conv(h, edge_index)

            # Add time conditioning (broadcast to all nodes)
            t_cond = time_proj(t_emb)
            # Assuming batch has all nodes from same timestep, expand t_cond
            h_new = h_new + t_cond.unsqueeze(0).expand(h_new.shape[0], -1)

            # Residual connection and normalization
            h = norm(h + h_new)
            h = F.gelu(h)

        # Project to output
        return self.output_proj(h)


class GraphDiffusionModel(nn.Module):
    """
    Diffusion model for graph-based pet matching

    Forward process: Gradually adds noise to matching scores
    Reverse process: Denoises to recover optimal matching
    """

    def __init__(
        self,
        pet_feature_dim: int,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_layers: int = 3,
        num_timesteps: int = 1000,
    ):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.pet_feature_dim = pet_feature_dim

        # Denoising network
        self.denoiser = TimeConditionedGNN(
            input_dim=pet_feature_dim * 2,  # Concatenated pair features
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_layers=num_layers
        )

        # Matching score predictor
        self.score_predictor = nn.Sequential(
            nn.Linear(pet_feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Diffusion schedule (linear beta schedule)
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)

        Args:
            x_start: Original features [batch_size, feature_dim]
            t: Timestep [batch_size]
            noise: Optional noise to add

        Returns:
            Noised features at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, x_t, t, edge_index):
        """
        Reverse diffusion step: p(x_{t-1} | x_t)

        Args:
            x_t: Noised features at timestep t
            t: Timestep
            edge_index: Graph connectivity

        Returns:
            Denoised features at timestep t-1
        """
        # Predict noise
        predicted_noise = self.denoiser(x_t, edge_index, t)

        # Compute x_{t-1}
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]

        # Reshape for broadcasting
        alpha_t = alpha_t.view(-1, 1)
        alpha_cumprod_t = alpha_cumprod_t.view(-1, 1)
        beta_t = beta_t.view(-1, 1)

        # Predicted x_0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)

        # Clip for stability
        pred_x0 = torch.clamp(pred_x0, -1, 1)

        # Sample x_{t-1}
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            alpha_cumprod_t_prev = self.alphas_cumprod[t - 1].view(-1, 1)

            # Posterior mean
            posterior_mean = (
                torch.sqrt(alpha_cumprod_t_prev) * beta_t / (1 - alpha_cumprod_t) * pred_x0 +
                torch.sqrt(alpha_t) * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * x_t
            )

            # Posterior variance
            posterior_variance = beta_t * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)

            return posterior_mean + torch.sqrt(posterior_variance) * noise
        else:
            return pred_x0

    def compute_loss(self, pet1_features, pet2_features, edge_index):
        """
        Compute diffusion training loss

        Args:
            pet1_features: Features of first pet [batch_size, feature_dim]
            pet2_features: Features of second pet [batch_size, feature_dim]
            edge_index: Graph connectivity

        Returns:
            Loss value
        """
        batch_size = pet1_features.shape[0]

        # Concatenate pair features
        pair_features = torch.cat([pet1_features, pet2_features], dim=1)

        # Sample random timestep for each example
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=pair_features.device)

        # Sample noise
        noise = torch.randn_like(pair_features)

        # Forward diffusion
        x_t = self.q_sample(pair_features, t, noise)

        # Predict noise
        predicted_noise = self.denoiser(x_t, edge_index, t)

        # MSE loss between predicted and actual noise
        loss = F.mse_loss(predicted_noise, noise)

        return loss

    @torch.no_grad()
    def sample(self, pet1_features, pet2_features, edge_index):
        """
        Generate matching scores via reverse diffusion

        Args:
            pet1_features: Features of first pet [batch_size, feature_dim]
            pet2_features: Features of second pet [batch_size, feature_dim]
            edge_index: Graph connectivity

        Returns:
            Matching scores [batch_size, 1]
        """
        batch_size = pet1_features.shape[0]
        device = pet1_features.device

        # Start from pure noise
        x_t = torch.randn(batch_size, self.pet_feature_dim * 2, device=device)

        # Reverse diffusion
        for t_idx in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            x_t = self.p_sample(x_t, t, edge_index)

        # Predict matching score from denoised features
        matching_score = self.score_predictor(x_t)

        return matching_score

    def forward(self, pet1_features, pet2_features, edge_index, return_score=True):
        """
        Forward pass - can be used for training or inference

        Args:
            pet1_features: Features of first pet
            pet2_features: Features of second pet
            edge_index: Graph connectivity
            return_score: If True, return matching score; else return loss

        Returns:
            Matching score or loss
        """
        if self.training:
            return self.compute_loss(pet1_features, pet2_features, edge_index)
        else:
            return self.sample(pet1_features, pet2_features, edge_index)


class ConditionalGraphDiffusion(nn.Module):
    """
    Conditional diffusion model that can incorporate user preferences
    and historical data for improved matching
    """

    def __init__(
        self,
        pet_feature_dim: int,
        condition_dim: int,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_layers: int = 3,
        num_timesteps: int = 1000,
    ):
        super().__init__()

        self.base_diffusion = GraphDiffusionModel(
            pet_feature_dim=pet_feature_dim + condition_dim,  # Concatenate conditions
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
        )

        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, condition_dim),
        )

    def forward(self, pet1_features, pet2_features, conditions, edge_index):
        """
        Args:
            pet1_features: Features of first pet
            pet2_features: Features of second pet
            conditions: Conditioning information (e.g., user preferences, feedback)
            edge_index: Graph connectivity

        Returns:
            Matching score or loss
        """
        # Encode conditions
        encoded_conditions = self.condition_encoder(conditions)

        # Concatenate with pet features
        conditioned_pet1 = torch.cat([pet1_features, encoded_conditions], dim=1)
        conditioned_pet2 = torch.cat([pet2_features, encoded_conditions], dim=1)

        return self.base_diffusion(conditioned_pet1, conditioned_pet2, edge_index)


class DiffusionMatchingOptimizer(nn.Module):
    """
    Uses diffusion to optimize matching assignments in a bipartite graph
    Solves the problem: given N pets and M potential matches, find optimal pairing
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_timesteps: int = 500,
    ):
        super().__init__()

        self.diffusion = GraphDiffusionModel(
            pet_feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_timesteps=num_timesteps
        )

    def optimize_matching(self, source_pets, target_pets, num_samples=10):
        """
        Find optimal matching between two sets of pets

        Args:
            source_pets: Features of source pets [N, feature_dim]
            target_pets: Features of target pets [M, feature_dim]
            num_samples: Number of diffusion samples to generate

        Returns:
            Matching matrix [N, M] with matching scores
        """
        N, M = source_pets.shape[0], target_pets.shape[0]
        device = source_pets.device

        # Create all pairs
        matching_scores = torch.zeros(N, M, device=device)

        for i in range(N):
            for j in range(M):
                pet1 = source_pets[i:i+1]
                pet2 = target_pets[j:j+1]

                # Create simple edge index for pair
                edge_index = torch.tensor([[0], [1]], device=device, dtype=torch.long)

                # Sample multiple times and average
                scores = []
                for _ in range(num_samples):
                    score = self.diffusion.sample(pet1, pet2, edge_index)
                    scores.append(score.item())

                matching_scores[i, j] = np.mean(scores)

        return matching_scores

    def get_optimal_assignment(self, matching_scores):
        """
        Get optimal one-to-one assignment using Hungarian algorithm
        (requires scipy)

        Args:
            matching_scores: Matching matrix [N, M]

        Returns:
            List of (source_idx, target_idx) pairs
        """
        from scipy.optimize import linear_sum_assignment

        # Convert to cost (1 - score)
        cost_matrix = 1 - matching_scores.detach().cpu().numpy()

        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        return list(zip(row_ind, col_ind))


if __name__ == '__main__':
    # Test diffusion model
    model = GraphDiffusionModel(pet_feature_dim=64, hidden_dim=128, num_timesteps=100)

    pet1 = torch.randn(4, 64)
    pet2 = torch.randn(4, 64)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    # Training mode
    model.train()
    loss = model(pet1, pet2, edge_index)
    print(f"Training loss: {loss.item():.4f}")

    # Inference mode
    model.eval()
    scores = model(pet1, pet2, edge_index)
    print(f"Matching scores: {scores.squeeze().tolist()}")

    # Test conditional diffusion
    conditional_model = ConditionalGraphDiffusion(
        pet_feature_dim=64,
        condition_dim=16,
        hidden_dim=128,
        num_timesteps=100
    )

    conditions = torch.randn(4, 16)  # User preferences
    conditional_model.train()
    cond_loss = conditional_model(pet1, pet2, conditions, edge_index)
    print(f"Conditional training loss: {cond_loss.item():.4f}")
