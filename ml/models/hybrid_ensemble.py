"""
Hybrid Ensemble Model for Pet Matching
Combines multiple approaches: GNN, SimGNN, and Diffusion
Uses adaptive weighting and meta-learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np


class AdaptiveWeightingModule(nn.Module):
    """
    Learns to weight different model predictions based on context
    """

    def __init__(self, num_models: int, context_dim: int):
        super().__init__()

        self.num_models = num_models

        # Context-dependent weight predictor
        self.weight_predictor = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_models),
            nn.Softmax(dim=-1)
        )

        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(num_models, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, predictions: torch.Tensor, context: torch.Tensor):
        """
        Args:
            predictions: Model predictions [batch_size, num_models]
            context: Context features [batch_size, context_dim]

        Returns:
            weighted_prediction: [batch_size, 1]
            weights: [batch_size, num_models]
            confidence: [batch_size, 1]
        """
        # Compute adaptive weights based on context
        weights = self.weight_predictor(context)

        # Weighted average of predictions
        weighted_prediction = (predictions * weights).sum(dim=1, keepdim=True)

        # Estimate confidence
        confidence = self.confidence_estimator(predictions)

        return weighted_prediction, weights, confidence


class UncertaintyEstimator(nn.Module):
    """
    Estimates prediction uncertainty using ensemble disagreement
    """

    def __init__(self, num_models: int):
        super().__init__()
        self.num_models = num_models

    def forward(self, predictions: torch.Tensor):
        """
        Args:
            predictions: [batch_size, num_models]

        Returns:
            uncertainty: [batch_size, 1]
        """
        # Variance across models
        variance = predictions.var(dim=1, keepdim=True)

        # Entropy-based uncertainty
        mean_pred = predictions.mean(dim=1, keepdim=True)
        epsilon = 1e-8
        entropy = -(mean_pred * torch.log(mean_pred + epsilon) +
                   (1 - mean_pred) * torch.log(1 - mean_pred + epsilon))

        # Combined uncertainty
        uncertainty = 0.5 * (variance + entropy)

        return uncertainty


class HybridEnsembleModel(nn.Module):
    """
    Ensemble of multiple matching models with adaptive weighting
    """

    def __init__(
        self,
        gat_model,
        simgnn_model,
        diffusion_model,
        context_dim: int = 32,
    ):
        super().__init__()

        self.gat_model = gat_model
        self.simgnn_model = simgnn_model
        self.diffusion_model = diffusion_model

        self.num_models = 3

        # Adaptive weighting
        self.weighting_module = AdaptiveWeightingModule(
            num_models=self.num_models,
            context_dim=context_dim
        )

        # Uncertainty estimation
        self.uncertainty_estimator = UncertaintyEstimator(num_models=self.num_models)

        # Context encoder (encodes pet pair into context)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, context_dim),
        )

    def extract_context(self, pet1_features, pet2_features):
        """
        Extract context from pet features for adaptive weighting

        Args:
            pet1_features: [batch_size, feature_dim]
            pet2_features: [batch_size, feature_dim]

        Returns:
            context: [batch_size, context_dim]
        """
        # Simple concatenation + feature difference
        context = torch.cat([
            pet1_features,
            pet2_features,
            torch.abs(pet1_features - pet2_features),
            pet1_features * pet2_features
        ], dim=1)

        # Encode to context dimension
        context = self.context_encoder(context)

        return context

    def forward(self, data1, data2, pet1_features, pet2_features, edge_index):
        """
        Args:
            data1: PyG Data for pet 1 (for SimGNN)
            data2: PyG Data for pet 2 (for SimGNN)
            pet1_features: Features for pet 1
            pet2_features: Features for pet 2
            edge_index: Edge index for diffusion

        Returns:
            final_prediction: [batch_size, 1]
            model_predictions: Dict with individual model outputs
            metadata: Dict with weights, confidence, uncertainty
        """
        # Get predictions from each model
        gat_pred = self.gat_model(data1.x, data1.edge_index, data1.edge_attr)
        simgnn_pred = self.simgnn_model(data1, data2)
        diffusion_pred = self.diffusion_model(pet1_features, pet2_features, edge_index)

        # Stack predictions
        all_predictions = torch.cat([gat_pred, simgnn_pred, diffusion_pred], dim=1)

        # Extract context for adaptive weighting
        context = self.extract_context(pet1_features, pet2_features)

        # Adaptive weighting
        final_prediction, weights, confidence = self.weighting_module(
            all_predictions, context
        )

        # Estimate uncertainty
        uncertainty = self.uncertainty_estimator(all_predictions)

        model_predictions = {
            'gat': gat_pred,
            'simgnn': simgnn_pred,
            'diffusion': diffusion_pred,
        }

        metadata = {
            'weights': weights,
            'confidence': confidence,
            'uncertainty': uncertainty,
        }

        return final_prediction, model_predictions, metadata


class AdaptiveFeedbackLoop(nn.Module):
    """
    Implements active learning / feedback loop system
    Identifies uncertain predictions and generates questions for users
    """

    def __init__(
        self,
        feature_dim: int,
        num_question_types: int = 10,
        uncertainty_threshold: float = 0.5,
    ):
        super().__init__()

        self.num_question_types = num_question_types
        self.uncertainty_threshold = uncertainty_threshold

        # Question generator - maps features to question importance
        self.question_generator = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_question_types),
            nn.Softmax(dim=-1)
        )

        # Feature updater - incorporates user responses
        self.feature_updater = nn.Sequential(
            nn.Linear(feature_dim + num_question_types, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

    def identify_uncertain_pairs(self, predictions, uncertainties):
        """
        Identify pairs that need additional information

        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates

        Returns:
            uncertain_indices: Indices of uncertain pairs
        """
        uncertain_mask = uncertainties.squeeze() > self.uncertainty_threshold
        uncertain_indices = torch.where(uncertain_mask)[0]

        return uncertain_indices

    def generate_questions(self, pet1_features, pet2_features):
        """
        Generate questions to ask users about uncertain pairs

        Args:
            pet1_features: Features of pet 1
            pet2_features: Features of pet 2

        Returns:
            question_priorities: [batch_size, num_question_types]
                Higher values = more important questions
        """
        pair_features = torch.cat([pet1_features, pet2_features], dim=1)
        question_priorities = self.question_generator(pair_features)

        return question_priorities

    def update_features(self, pet_features, user_responses):
        """
        Update pet features based on user responses

        Args:
            pet_features: Original features [batch_size, feature_dim]
            user_responses: User answers [batch_size, num_question_types]

        Returns:
            updated_features: [batch_size, feature_dim]
        """
        combined = torch.cat([pet_features, user_responses], dim=1)
        updated_features = self.feature_updater(combined)

        # Residual connection
        updated_features = pet_features + updated_features

        return updated_features

    def forward(
        self,
        pet1_features,
        pet2_features,
        uncertainties,
        user_responses=None
    ):
        """
        Complete feedback loop

        Args:
            pet1_features: Features of pet 1
            pet2_features: Features of pet 2
            uncertainties: Uncertainty estimates
            user_responses: Optional user responses to questions

        Returns:
            updated_pet1: Updated features for pet 1
            updated_pet2: Updated features for pet 2
            questions: Questions to ask (if no responses provided)
        """
        # Identify uncertain pairs
        uncertain_indices = self.identify_uncertain_pairs(None, uncertainties)

        if user_responses is None:
            # Generate questions for uncertain pairs
            questions = self.generate_questions(pet1_features, pet2_features)
            return pet1_features, pet2_features, questions
        else:
            # Update features with user responses
            updated_pet1 = self.update_features(pet1_features, user_responses)
            updated_pet2 = self.update_features(pet2_features, user_responses)
            return updated_pet1, updated_pet2, None


class MetaLearningMatcher(nn.Module):
    """
    Meta-learning approach for few-shot matching
    Learns to quickly adapt to new user preferences
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Fast adaptation network (like MAML)
        self.fast_weights = nn.ParameterDict({
            'w1': nn.Parameter(torch.randn(feature_dim * 2, hidden_dim)),
            'b1': nn.Parameter(torch.zeros(hidden_dim)),
            'w2': nn.Parameter(torch.randn(hidden_dim, 1)),
            'b2': nn.Parameter(torch.zeros(1)),
        })

    def forward(self, pet1_features, pet2_features, fast_weights=None):
        """
        Args:
            pet1_features: Features of pet 1
            pet2_features: Features of pet 2
            fast_weights: Adapted weights (if available)

        Returns:
            predictions: Matching predictions
        """
        if fast_weights is None:
            fast_weights = self.fast_weights

        # Concatenate features
        x = torch.cat([pet1_features, pet2_features], dim=1)

        # Forward pass with fast weights
        h = F.linear(x, fast_weights['w1'], fast_weights['b1'])
        h = F.relu(h)
        out = F.linear(h, fast_weights['w2'], fast_weights['b2'])
        out = torch.sigmoid(out)

        return out

    def adapt(self, support_pet1, support_pet2, support_labels, num_steps=5, alpha=0.01):
        """
        Adapt to new user preferences with few examples (MAML-style)

        Args:
            support_pet1: Pet 1 features from support set
            support_pet2: Pet 2 features from support set
            support_labels: Labels for support set
            num_steps: Number of adaptation steps
            alpha: Learning rate for adaptation

        Returns:
            adapted_weights: Weights adapted to user preferences
        """
        adapted_weights = {k: v.clone() for k, v in self.fast_weights.items()}

        for _ in range(num_steps):
            # Forward pass
            predictions = self.forward(support_pet1, support_pet2, adapted_weights)

            # Compute loss
            loss = F.binary_cross_entropy(predictions, support_labels)

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_weights.values(),
                create_graph=True
            )

            # Update weights
            adapted_weights = {
                k: v - alpha * g
                for (k, v), g in zip(adapted_weights.items(), grads)
            }

        return adapted_weights


class RichDataProber(nn.Module):
    """
    Generates diverse questions to collect richer data about pets
    Uses information gain to select most valuable questions
    """

    def __init__(self, feature_dim: int, num_probes: int = 20):
        super().__init__()

        self.num_probes = num_probes

        # Information gain estimator
        self.info_gain_network = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_probes),
            nn.Softmax(dim=-1)
        )

        # Probe embeddings (representing different questions)
        self.probe_embeddings = nn.Parameter(torch.randn(num_probes, 32))

    def estimate_information_gain(self, current_features, model_uncertainty):
        """
        Estimate information gain for each possible probe/question

        Args:
            current_features: Current pet features
            model_uncertainty: Current model uncertainty

        Returns:
            info_gains: Expected information gain for each probe
        """
        # Combine features with uncertainty
        combined = torch.cat([current_features, model_uncertainty], dim=1)

        # Estimate info gain
        info_gains = self.info_gain_network(current_features)

        return info_gains

    def select_next_probes(self, current_features, model_uncertainty, k=3):
        """
        Select top-k most informative probes

        Args:
            current_features: Current features
            model_uncertainty: Model uncertainty
            k: Number of probes to select

        Returns:
            probe_indices: Indices of selected probes
            probe_embeddings: Embeddings of selected probes
        """
        info_gains = self.estimate_information_gain(current_features, model_uncertainty)

        # Select top-k
        top_k_values, top_k_indices = torch.topk(info_gains, k, dim=1)

        # Get corresponding embeddings
        batch_size = current_features.shape[0]
        selected_embeddings = []

        for i in range(batch_size):
            batch_probes = self.probe_embeddings[top_k_indices[i]]
            selected_embeddings.append(batch_probes)

        selected_embeddings = torch.stack(selected_embeddings)

        return top_k_indices, selected_embeddings, top_k_values


# Question bank for feedback system
QUESTION_BANK = {
    0: "Does your pet enjoy playing with other dogs/cats?",
    1: "Is your pet comfortable with high-energy activities?",
    2: "Does your pet prefer indoor or outdoor activities?",
    3: "How does your pet react to new pets?",
    4: "Does your pet enjoy structured playtime?",
    5: "Is your pet food-motivated during play?",
    6: "Does your pet prefer one-on-one or group play?",
    7: "How vocal is your pet during play?",
    8: "Does your pet have any play preferences?",
    9: "How long can your pet play continuously?",
}


if __name__ == '__main__':
    print("Testing Hybrid Ensemble System...")

    # Create dummy models
    from gnn_compatibility import GraphAttentionCompatibility
    from simgnn import SimGNN
    from graph_diffusion import GraphDiffusionModel

    gat_model = GraphAttentionCompatibility(n_breeds=15, n_temperaments=9)
    simgnn_model = SimGNN(input_dim=10, hidden_dim=64)
    diffusion_model = GraphDiffusionModel(pet_feature_dim=64, hidden_dim=64, num_timesteps=100)

    # Create ensemble
    ensemble = HybridEnsembleModel(
        gat_model=gat_model,
        simgnn_model=simgnn_model,
        diffusion_model=diffusion_model,
        context_dim=32
    )

    print(f"✓ Hybrid ensemble created with {ensemble.num_models} models")

    # Test feedback loop
    feedback_loop = AdaptiveFeedbackLoop(feature_dim=64, num_question_types=10)
    print("✓ Adaptive feedback loop initialized")

    # Test meta-learning
    meta_learner = MetaLearningMatcher(feature_dim=64)
    print("✓ Meta-learning matcher created")

    # Test rich data prober
    prober = RichDataProber(feature_dim=64, num_probes=20)
    print("✓ Rich data prober initialized")

    print("\nAll hybrid ensemble components ready!")
