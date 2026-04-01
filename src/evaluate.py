"""
Evaluation and testing for Buffalo Breed Recognition
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_recall_fscore_support
)
from tqdm import tqdm
import logging
import json

logger = logging.getLogger(__name__)


def evaluate_model(model, test_loader, device, class_names, save_dir='results'):
    """Comprehensive model evaluation"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    logger.info("Evaluating model on test set...")

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    logger.info(f"\n{'='*60}")
    logger.info("Test Set Results:")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, save_dir)

    # Classification Report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4
    )
    logger.info(f"\nClassification Report:\n{report}")

    # Save report
    with open(f"{save_dir}/classification_report.txt", 'w') as f:
        f.write(report)

    # Per-class accuracy
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        mask = np.array(all_labels) == i
        if mask.sum() > 0:
            class_acc = accuracy_score(
                np.array(all_labels)[mask],
                np.array(all_preds)[mask]
            )
            per_class_acc[class_name] = class_acc

    logger.info("\nPer-Class Accuracy:")
    for class_name, acc in sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {class_name}: {acc:.4f}")

    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'per_class_accuracy': {k: float(v) for k, v in per_class_acc.items()}
    }

    with open(f"{save_dir}/test_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics, cm


def plot_confusion_matrix(cm, class_names, save_dir='results'):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Confusion matrix saved to {save_dir}/confusion_matrix.png")


def plot_training_history(history, save_dir='plots'):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[0, 1].set_title('Accuracy Over Epochs',
                         fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Precision & Recall
    axes[1, 0].plot(history['val_precision'], label='Precision', linewidth=2)
    axes[1, 0].plot(history['val_recall'], label='Recall', linewidth=2)
    axes[1, 0].set_title('Precision & Recall', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # F1 Score
    axes[1, 1].plot(history['val_f1'], label='F1 Score',
                    linewidth=2, color='green')
    axes[1, 1].set_title('F1 Score Over Epochs',
                         fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_history.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Training history saved to {save_dir}/training_history.png")


def analyze_misclassifications(model, test_loader, device, class_names, save_dir='results', num_samples=100):
    """Analyze and save misclassified samples with detailed confusion analysis"""
    model.eval()

    misclassified = []
    confusion_pairs = {}  # Track which classes are confused with each other

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            # Find misclassified
            mask = predicted != labels
            if mask.sum() > 0:
                for i in range(len(labels)):
                    if mask[i]:
                        true_idx = labels[i].item()
                        pred_idx = predicted[i].item()
                        true_label = class_names[true_idx]
                        pred_label = class_names[pred_idx]
                        confidence = probs[i][pred_idx].item()

                        misclassified.append({
                            'true_label': true_label,
                            'predicted_label': pred_label,
                            'confidence': confidence,
                            'confidence_percent': f"{confidence*100:.2f}%",
                            'top_3_predictions': [
                                {
                                    'breed': class_names[idx],
                                    'confidence': probs[i][idx].item(),
                                    'confidence_percent': f"{probs[i][idx].item()*100:.2f}%"
                                }
                                for idx in torch.topk(probs[i], k=min(3, len(class_names))).indices.tolist()
                            ]
                        })

                        # Track confusion pairs
                        pair_key = f"{true_label} -> {pred_label}"
                        confusion_pairs[pair_key] = confusion_pairs.get(
                            pair_key, 0) + 1

    logger.info(f"\nMisclassification Analysis:")
    logger.info(f"  Total misclassified: {len(misclassified)}")

    # Analyze most common confusion pairs
    if confusion_pairs:
        logger.info(f"\nMost Common Confusion Pairs:")
        sorted_pairs = sorted(confusion_pairs.items(),
                              key=lambda x: x[1], reverse=True)
        for pair, count in sorted_pairs[:10]:
            logger.info(f"  {pair}: {count} times")

    # Save detailed misclassification report
    report = {
        'total_misclassified': len(misclassified),
        'confusion_pairs': confusion_pairs,
        'misclassified_samples': misclassified[:num_samples]
    }

    with open(f"{save_dir}/misclassifications.json", 'w') as f:
        json.dump(report, f, indent=4)

    logger.info(
        f"✓ Detailed misclassification report saved to {save_dir}/misclassifications.json")

    return misclassified
