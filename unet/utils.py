# elpv_segmentation/utils.py
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_metrics(true_masks, pred_masks, num_classes):
    true_masks_flat = true_masks.cpu().numpy().flatten()
    pred_masks_flat = pred_masks.cpu().numpy().flatten()
    
    cm = confusion_matrix(true_masks_flat, pred_masks_flat, labels=list(range(num_classes)))
    
    iou_per_class = []
    f1_per_class = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        iou_per_class.append(iou)
        f1_per_class.append(f1)
        
    mean_iou = np.nanmean([iou for iou in iou_per_class if not np.isnan(iou)])
    mean_f1 = np.nanmean([f1 for f1 in f1_per_class if not np.isnan(f1)])
    
    return mean_iou, mean_f1, iou_per_class, f1_per_class

def plot_training_history(history, history_save_path_str):
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss over Epochs"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True); plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history["val_iou"], label="Val IoU")
    plt.plot(history["val_f1"], label="Val F1-Score")
    plt.title("Validation IoU & F1-Score"); plt.xlabel("Epoch"); plt.ylabel("Metric"); plt.grid(True); plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history["lr"], label="Learning Rate")
    plt.title("Learning Rate over Epochs"); plt.xlabel("Epoch"); plt.ylabel("LR"); plt.grid(True); plt.legend()

    plt.tight_layout()
    plot_save_path = Path(history_save_path_str).stem + "_plots.png"
    plt.savefig(plot_save_path)
    print(f"Training plots saved to {plot_save_path}")
    # plt.show() # Uncomment if you want to display plots interactively