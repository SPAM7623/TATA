"""
Utility Functions for Alpha Defect Detection Pipeline
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    auc, precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def print_header(title, width=80):
    """Print formatted header"""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def get_feature_columns(df, prefix='X'):
    """Extract feature columns by prefix"""
    return [col for col in df.columns if col.startswith(prefix)]


def calculate_metrics(y_true, y_pred, y_pred_proba=None, threshold=0.5):
    """
    Calculate comprehensive classification metrics

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Binary predictions
    y_pred_proba : array-like, optional
        Probability predictions for ROC-AUC
    threshold : float
        Classification threshold

    Returns:
    --------
    dict : Metrics dictionary
    """
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, accuracy_score
    )

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'threshold': threshold
    }

    if y_pred_proba is not None:
        # Handle different shape formats
        if len(y_pred_proba.shape) > 1:
            proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0]
        else:
            proba = y_pred_proba

        try:
            metrics['roc_auc'] = roc_auc_score(y_true, proba)
        except:
            metrics['roc_auc'] = np.nan

        try:
            precision, recall, _ = precision_recall_curve(y_true, proba)
            metrics['pr_auc'] = auc(recall, precision)
        except:
            metrics['pr_auc'] = np.nan

    return metrics


def print_metrics(metrics, decimal_places=4):
    """Print metrics in formatted table"""
    print("\nClassification Metrics:")
    print("-" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:15s}: {value:.{decimal_places}f}")
        else:
            print(f"  {key:15s}: {value}")


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Defect'],
                yticklabels=['Normal', 'Defect'])

    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    return plt.gcf()


def plot_roc_curve(y_true, y_pred_proba):
    """Plot ROC curve"""
    # Handle different shape formats
    if len(y_pred_proba.shape) > 1:
        proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0]
    else:
        proba = y_pred_proba

    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def plot_precision_recall_curve(y_true, y_pred_proba):
    """Plot Precision-Recall curve"""
    # Handle different shape formats
    if len(y_pred_proba.shape) > 1:
        proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0]
    else:
        proba = y_pred_proba

    precision, recall, _ = precision_recall_curve(y_true, proba)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.axhline(y=y_true.mean(), color='red', linestyle='--', label=f'Baseline ({y_true.mean():.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """
    Find optimal classification threshold

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Probability predictions
    metric : str
        Metric to optimize ('f1', 'recall', 'precision')

    Returns:
    --------
    float : Optimal threshold
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    # Handle different shape formats
    if len(y_pred_proba.shape) > 1:
        proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0]
    else:
        proba = y_pred_proba

    thresholds = np.arange(0.0, 1.01, 0.01)
    best_threshold = 0.5
    best_score = 0

    for threshold in thresholds:
        y_pred = (proba >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold


def plot_threshold_analysis(y_true, y_pred_proba):
    """
    Plot metric scores across different thresholds

    Useful for finding optimal decision boundary
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    # Handle different shape formats
    if len(y_pred_proba.shape) > 1:
        proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0]
    else:
        proba = y_pred_proba

    thresholds = np.arange(0.0, 1.01, 0.01)
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for threshold in thresholds:
        y_pred = (proba >= threshold).astype(int)
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        precision_scores.append(precision_score(y_true, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_true, y_pred, zero_division=0))

    plt.figure(figsize=(12, 6))
    plt.plot(thresholds, f1_scores, label='F1-Score', marker='o', linewidth=2)
    plt.plot(thresholds, precision_scores, label='Precision', marker='s', linewidth=2)
    plt.plot(thresholds, recall_scores, label='Recall', marker='^', linewidth=2)

    plt.axvline(x=0.5, color='red', linestyle='--', label='Default (0.5)')
    plt.xlabel('Classification Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Metrics vs Classification Threshold', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()

    return plt.gcf()


def print_classification_report(y_true, y_pred, target_names=None):
    """Print detailed classification report"""
    if target_names is None:
        target_names = ['Normal', 'Defect']

    report = classification_report(y_true, y_pred, target_names=target_names)
    print("\nDetailed Classification Report:")
    print(report)


class ModelPerformanceTracker:
    """Track and compare model performance across folds and iterations"""

    def __init__(self):
        self.results = []

    def add_result(self, fold, model_name, metrics):
        """Add fold result"""
        result = {
            'fold': fold,
            'model': model_name,
            **metrics
        }
        self.results.append(result)

    def get_summary(self):
        """Get summary statistics across folds"""
        df = pd.DataFrame(self.results)
        return df.groupby('model')[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].agg(
            ['mean', 'std']
        )

    def print_summary(self):
        """Print formatted summary"""
        summary = self.get_summary()
        print("\n" + "=" * 80)
        print("MODEL PERFORMANCE SUMMARY")
        print("=" * 80)
        print(summary.to_string())


def load_config(config_path):
    """Load YAML configuration file"""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
