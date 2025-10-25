"""
Comprehensive model benchmarking and comparison system
Tests all models: GNN, SimGNN, Diffusion, and Hybrid Ensemble
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_recall_curve,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = Path('ml/data')
MODEL_DIR = Path('ml/models/saved')
RESULTS_DIR = Path('ml/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class ModelBenchmark:
    """Comprehensive benchmarking suite for pet matching models"""

    def __init__(self):
        self.results = {
            'models': {},
            'timestamp': datetime.now().isoformat(),
        }

    def load_test_data(self):
        """Load test dataset"""
        print("Loading test data...")
        pets_df = pd.read_csv(DATA_DIR / 'graph_pets.csv')
        labels_df = pd.read_csv(DATA_DIR / 'graph_labels.csv')

        return pets_df, labels_df

    def evaluate_model(self, model_name, predictions, labels):
        """
        Evaluate a model and compute comprehensive metrics

        Args:
            model_name: Name of the model
            predictions: Model predictions [N]
            labels: Ground truth labels [N]

        Returns:
            Dictionary of metrics
        """
        print(f"\nEvaluating {model_name}...")

        # Classification metrics
        roc_auc = roc_auc_score(labels, predictions)
        avg_precision = average_precision_score(labels, predictions)

        # Thresholded metrics
        binary_preds = (predictions > 0.5).astype(int)
        accuracy = accuracy_score(labels, binary_preds)
        f1 = f1_score(labels, binary_preds)

        # Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(labels, predictions)

        # Find optimal threshold (max F1)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

        # Calibration metrics
        mse = np.mean((predictions - labels) ** 2)
        mae = np.mean(np.abs(predictions - labels))

        # Distribution analysis
        pred_mean = predictions.mean()
        pred_std = predictions.std()

        metrics = {
            'roc_auc': float(roc_auc),
            'avg_precision': float(avg_precision),
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'optimal_threshold': float(optimal_threshold),
            'mse': float(mse),
            'mae': float(mae),
            'pred_mean': float(pred_mean),
            'pred_std': float(pred_std),
            'precision_recall': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist(),
            }
        }

        self.results['models'][model_name] = metrics

        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Avg Precision: {avg_precision:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")

        return metrics

    def compare_models(self):
        """Compare all models side-by-side"""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)

        if len(self.results['models']) == 0:
            print("No models to compare")
            return

        # Create comparison table
        comparison_df = pd.DataFrame(self.results['models']).T

        # Select key metrics
        key_metrics = ['roc_auc', 'avg_precision', 'accuracy', 'f1_score', 'mse']
        comparison_table = comparison_df[key_metrics]

        print("\n", comparison_table.to_string())

        # Find best model for each metric
        print("\n" + "-" * 80)
        print("BEST MODELS PER METRIC")
        print("-" * 80)

        for metric in key_metrics:
            if metric == 'mse':
                best_model = comparison_table[metric].idxmin()
                best_value = comparison_table[metric].min()
            else:
                best_model = comparison_table[metric].idxmax()
                best_value = comparison_table[metric].max()

            print(f"{metric:20s}: {best_model:15s} ({best_value:.4f})")

        # Overall winner (weighted score)
        weights = {
            'roc_auc': 0.3,
            'avg_precision': 0.3,
            'accuracy': 0.2,
            'f1_score': 0.15,
            'mse': -0.05,  # Negative because lower is better
        }

        scores = {}
        for model in comparison_table.index:
            score = sum(
                comparison_table.loc[model, metric] * weight
                for metric, weight in weights.items()
            )
            scores[model] = score

        winner = max(scores, key=scores.get)

        print("\n" + "=" * 80)
        print(f"OVERALL WINNER: {winner} (Score: {scores[winner]:.4f})")
        print("=" * 80)

        return comparison_table, scores

    def plot_comparisons(self):
        """Generate comparison visualizations"""
        print("\nGenerating comparison plots...")

        if len(self.results['models']) < 2:
            print("Need at least 2 models for comparison")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Comparison Dashboard', fontsize=16)

        # 1. ROC-AUC Comparison
        models = list(self.results['models'].keys())
        roc_aucs = [self.results['models'][m]['roc_auc'] for m in models]

        axes[0, 0].barh(models, roc_aucs, color='skyblue')
        axes[0, 0].set_xlabel('ROC-AUC')
        axes[0, 0].set_title('ROC-AUC Comparison')
        axes[0, 0].set_xlim([0, 1])

        # 2. Precision-Recall Curves
        for model_name in models:
            pr_data = self.results['models'][model_name]['precision_recall']
            axes[0, 1].plot(pr_data['recall'], pr_data['precision'],
                          label=model_name, linewidth=2)

        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Multiple Metrics Comparison
        metrics_to_plot = ['roc_auc', 'avg_precision', 'accuracy', 'f1_score']
        x = np.arange(len(models))
        width = 0.2

        for i, metric in enumerate(metrics_to_plot):
            values = [self.results['models'][m][metric] for m in models]
            axes[1, 0].bar(x + i * width, values, width, label=metric)

        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Multi-Metric Comparison')
        axes[1, 0].set_xticks(x + width * 1.5)
        axes[1, 0].set_xticklabels(models, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].set_ylim([0, 1])

        # 4. Error Analysis (MSE vs MAE)
        mses = [self.results['models'][m]['mse'] for m in models]
        maes = [self.results['models'][m]['mae'] for m in models]

        x_pos = np.arange(len(models))
        axes[1, 1].bar(x_pos - 0.2, mses, 0.4, label='MSE', color='coral')
        axes[1, 1].bar(x_pos + 0.2, maes, 0.4, label='MAE', color='lightgreen')

        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].set_title('Prediction Error Comparison')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(models, rotation=45)
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {RESULTS_DIR / 'model_comparison.png'}")

    def save_results(self):
        """Save benchmark results"""
        # Save JSON results
        with open(RESULTS_DIR / 'benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Saved results to {RESULTS_DIR / 'benchmark_results.json'}")

    def run_full_benchmark(self):
        """Run complete benchmarking pipeline"""
        print("\n" + "=" * 80)
        print("STARTING COMPREHENSIVE MODEL BENCHMARK")
        print("=" * 80)

        # Load test data
        pets_df, labels_df = self.load_test_data()

        # Here you would load each trained model and get predictions
        # For now, we'll create dummy predictions to demonstrate

        print("\nNote: To run full benchmark, load trained models and generate predictions")
        print("Example:")
        print("  model = torch.load(MODEL_DIR / 'gnn_best.pt')")
        print("  predictions = model.predict(test_data)")
        print("  benchmark.evaluate_model('GNN', predictions, labels)")

        # Compare models (if any have been evaluated)
        if len(self.results['models']) > 0:
            self.compare_models()
            self.plot_comparisons()

        # Save results
        self.save_results()

        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE")
        print("=" * 80)


class HybridModelAnalyzer:
    """Analyzes hybrid ensemble model behavior"""

    def __init__(self):
        self.weight_history = []

    def analyze_model_contributions(self, predictions_dict, weights_history):
        """
        Analyze how each model contributes to final predictions

        Args:
            predictions_dict: Dict of {model_name: predictions}
            weights_history: History of adaptive weights
        """
        print("\n" + "=" * 80)
        print("HYBRID MODEL ANALYSIS")
        print("=" * 80)

        # Average weights
        avg_weights = {}
        for model_name in predictions_dict.keys():
            avg_weights[model_name] = np.mean([w[model_name] for w in weights_history])

        print("\nAverage Model Weights:")
        for model, weight in sorted(avg_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model:15s}: {weight:.4f}")

        # Weight variance (stability)
        print("\nWeight Stability (lower = more stable):")
        for model_name in predictions_dict.keys():
            variance = np.var([w[model_name] for w in weights_history])
            print(f"  {model:15s}: {variance:.6f}")

        return avg_weights

    def analyze_uncertainty_patterns(self, uncertainties, predictions, labels):
        """Analyze when the model is uncertain"""
        print("\n" + "=" * 80)
        print("UNCERTAINTY ANALYSIS")
        print("=" * 80)

        # Correlation between uncertainty and error
        errors = np.abs(predictions - labels)
        correlation = np.corrcoef(uncertainties, errors)[0, 1]

        print(f"\nCorrelation(Uncertainty, Error): {correlation:.4f}")
        print(f"Mean Uncertainty: {uncertainties.mean():.4f}")
        print(f"Std Uncertainty: {uncertainties.std():.4f}")

        # Uncertainty thresholding
        high_uncertainty_mask = uncertainties > uncertainties.mean() + uncertainties.std()
        print(f"\nHigh Uncertainty Cases: {high_uncertainty_mask.sum()} ({100*high_uncertainty_mask.mean():.1f}%)")
        print(f"  Average Error (High Uncertainty): {errors[high_uncertainty_mask].mean():.4f}")
        print(f"  Average Error (Low Uncertainty): {errors[~high_uncertainty_mask].mean():.4f}")

        return correlation


if __name__ == '__main__':
    # Run benchmark
    benchmark = ModelBenchmark()

    # Example: Add some dummy results for demonstration
    # In practice, you'd load real model predictions

    print("\n" + "=" * 80)
    print("MODEL BENCHMARKING SYSTEM INITIALIZED")
    print("=" * 80)
    print("\nTo use:")
    print("1. Train your models (GNN, SimGNN, Diffusion, Hybrid)")
    print("2. Load each model and generate predictions on test set")
    print("3. Call benchmark.evaluate_model(name, predictions, labels)")
    print("4. Call benchmark.compare_models()")
    print("5. Call benchmark.plot_comparisons()")
    print("6. Call benchmark.save_results()")

    # benchmark.run_full_benchmark()
