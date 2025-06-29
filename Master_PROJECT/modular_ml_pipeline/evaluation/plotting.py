# evaluation/plotting.py

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import os

def plot_model_comparison_roc(results, output_dir="results"):
    """
    Compares the ROC curves of the trained models on a single plot.
    """
    print("--- Comparing Model ROC Curves ---")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    
    for model_name, (y_true, y_pred_proba, auc_score) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {auc_score:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.title('Model Comparison ROC Curves', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.5)
    
    save_path = os.path.join(output_dir, 'model_comparison_roc_curve.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Comparison plot saved to '{save_path}'.")
    plt.show()