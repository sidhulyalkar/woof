#!/usr/bin/env python3
"""
Real-time training monitoring dashboard
"""

import time
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

MODEL_DIR = Path('ml/models/saved')

def print_header():
    print("\n" + "="*100)
    print(" "*35 + "🤖 PETPATH ML TRAINING MONITOR")
    print("="*100)

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.2f}h"

def monitor_training():
    """Monitor training progress in real-time"""
    print_header()

    models = ['gat', 'simgnn', 'diffusion']

    while True:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training Status")
        print("-"*100)

        for model_name in models:
            history_file = MODEL_DIR / f'{model_name}_history.csv'
            metrics_file = MODEL_DIR / f'{model_name}_metrics.json'

            if history_file.exists():
                # Read latest training progress
                df = pd.read_csv(history_file)
                latest = df.iloc[-1]

                status = f"✓ {model_name.upper():10s} | "
                status += f"Epoch {int(latest['epoch']):3d} | "
                status += f"Val Loss: {latest['val_loss']:.4f} | "
                status += f"ROC-AUC: {latest['roc_auc']:.4f} | "
                status += f"Acc: {latest['accuracy']:.3f}"

                print(status)

            elif metrics_file.exists():
                # Training complete
                with open(metrics_file) as f:
                    metrics = json.load(f)

                status = f"✅ {model_name.upper():10s} | "
                status += f"COMPLETE | "
                status += f"Test ROC-AUC: {metrics['test_roc_auc']:.4f} | "
                status += f"Time: {format_time(metrics['training_time_seconds'])}"

                print(status)
            else:
                print(f"⏳ {model_name.upper():10s} | Waiting to start...")

        # Check if all complete
        all_complete = all(
            (MODEL_DIR / f'{m}_metrics.json').exists()
            for m in models
        )

        if all_complete:
            print("\n" + "="*100)
            print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
            print("="*100)

            # Show comparison
            print("\nFinal Comparison:")
            print("-"*100)

            for model_name in models:
                with open(MODEL_DIR / f'{model_name}_metrics.json') as f:
                    metrics = json.load(f)

                print(f"{model_name.upper():10s} | "
                      f"ROC-AUC: {metrics['test_roc_auc']:.4f} | "
                      f"Accuracy: {metrics['test_accuracy']:.4f} | "
                      f"Time: {format_time(metrics['training_time_seconds'])}")

            # Find winner
            best_model = max(models, key=lambda m:
                json.load(open(MODEL_DIR / f'{m}_metrics.json'))['test_roc_auc']
            )

            print("\n" + "="*100)
            print(f"🏆 BEST MODEL: {best_model.upper()}")
            print("="*100)

            break

        time.sleep(5)  # Update every 5 seconds

if __name__ == '__main__':
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
