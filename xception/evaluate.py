# evaluate_and_plot.py

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-GUI environments where plots are saved
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import timm

# Assuming the elpv_dataset utility is in the python path or same directory
# If not, you might need to adjust sys.path or copy the utility
try:
    from elpv_dataset.utils import load_dataset
except ImportError:
    print("Warning: elpv_dataset.utils.load_dataset not found. "
          "The script will fail if data loading is attempted without this. "
          "Using dummy data for script structure testing if load_dataset is undefined.")
    def load_dataset(): # Dummy function if not found
        print("Using dummy load_dataset(). Please ensure the real one is available.")
        _images_np = np.random.rand(100, 300, 300).astype(np.float32)
        _proba_np = np.random.rand(100).astype(np.float32)
        _types_np = np.array(['mono'] * 50 + ['poly'] * 50)
        return _images_np, _proba_np, _types_np

# ——— Configuration ———
CONFIG = {
    "HISTORY_FILE_PATH": "training_history.json",
    "MODEL_PATH": "best_microcrack_detector.pth",
    "PLOT_SAVE_DIR": "evaluation_plots",
    "NUM_CLASSES": 1,
    "BATCH_SIZE": 32, # Should match training for consistency if any batch-dependent layers were an issue
    "RANDOM_SEED": 42, # Must be the same as used in training for reproducible test split
    "TRAIN_SPLIT_RATIO": 0.7, # Same as in training
    "VAL_SPLIT_RATIO": 0.15,  # Same as in training
    "TEST_SPLIT_RATIO": 0.15, # Same as in training
    # ImageNet mean/std for normalization (must match what was used in training for val/test)
    "NORM_MEAN": [0.485, 0.456, 0.406],
    "NORM_STD": [0.229, 0.224, 0.225],
}

# Create plot save directory if it doesn't exist
Path(CONFIG["PLOT_SAVE_DIR"]).mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ——— Dataset Class (copied from training script) ———
class ELPVArrayDataset(Dataset):
    def __init__(self, imgs, lbls, transform=None):
        self.imgs = imgs
        self.lbls = lbls
        self.transform = transform
    def __len__(self):
        return len(self.lbls)
    def __getitem__(self, i):
        img = self.imgs[i]
        if img.ndim == 2: 
            img = np.stack([img]*3, axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 1: 
            img = np.repeat(img, 3, axis=-1)
        
        if img.dtype != np.uint8: # Ensure uint8 for albumentations
            if img.max() <= 1.0 and img.min() >=0.0 : 
                 img = (img * 255).astype(np.uint8)
            elif img.max() <= 255 and img.min() >=0: 
                 img = img.astype(np.uint8)
            # else: you might want to raise an error or handle other cases

        if self.transform:
            img = self.transform(image=img)['image']
        lbl = torch.tensor([self.lbls[i]], dtype=torch.float32) # Label for BCEWithLogitsLoss
        return img, lbl

# ——— Test Transformation (copied from val_tf in training script) ———
test_tf = A.Compose([
    A.Normalize(mean=CONFIG["NORM_MEAN"], std=CONFIG["NORM_STD"]),
    ToTensorV2()
])

# ——— Function to Load Test Data (reproducing the split) ———
def get_test_data():
    print("Loading ELPV dataset for test set...")
    images_np, proba_np, _ = load_dataset()
    labels_np = (proba_np > 0.5).astype(np.int64) # Binarize labels

    idx = np.arange(len(labels_np))
    
    # Stratified split logic (same as training.py)
    # Ensure there are enough samples for each class to stratify
    if len(np.unique(labels_np)) > 1:
        _, temp_idx, _, temp_lbl_strat = train_test_split(
            idx, labels_np,
            test_size=(CONFIG["VAL_SPLIT_RATIO"] + CONFIG["TEST_SPLIT_RATIO"]),
            stratify=labels_np,
            random_state=CONFIG["RANDOM_SEED"]
        )
        # Calculate the proportion of the temporary set that should be validation
        # The remainder will be test. test_prop is for the second split.
        test_prop_for_split = CONFIG["TEST_SPLIT_RATIO"] / (CONFIG["VAL_SPLIT_RATIO"] + CONFIG["TEST_SPLIT_RATIO"])

        if len(np.unique(temp_lbl_strat)) > 1 :
            _, test_idx, _, _ = train_test_split(
                temp_idx, temp_lbl_strat,
                test_size=test_prop_for_split, # Proportion for the test set from the temp_idx
                stratify=temp_lbl_strat,
                random_state=CONFIG["RANDOM_SEED"]
            )
        else:
            print("Warning: Not enough class diversity for stratified val/test split part 2. Using non-stratified.")
            _, test_idx = train_test_split(
                temp_idx,
                test_size=test_prop_for_split,
                random_state=CONFIG["RANDOM_SEED"]
            )
    else: 
        print("Warning: Only one class present in the dataset for train/temp split. Using non-stratified split.")
        _, temp_idx = train_test_split(idx, test_size=(CONFIG["VAL_SPLIT_RATIO"] + CONFIG["TEST_SPLIT_RATIO"]), random_state=CONFIG["RANDOM_SEED"])
        test_prop_for_split = CONFIG["TEST_SPLIT_RATIO"] / (CONFIG["VAL_SPLIT_RATIO"] + CONFIG["TEST_SPLIT_RATIO"])
        _, test_idx = train_test_split(temp_idx, test_size=test_prop_for_split, random_state=CONFIG["RANDOM_SEED"])

    test_imgs, test_lbls = images_np[test_idx], labels_np[test_idx]
    print(f"Test samples: {len(test_imgs)} (0s: {np.sum(test_lbls == 0)}, 1s: {np.sum(test_lbls == 1)})")
    return test_imgs, test_lbls

# ——— Plotting Function ———
def plot_training_history(history_file_path, plot_save_dir):
    print(f"Loading training history from {history_file_path}...")
    if not Path(history_file_path).exists():
        print(f"ERROR: History file not found at {history_file_path}")
        return

    with open(history_file_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot 1: Loss (Train vs Val)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='.')
    plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='.')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(plot_save_dir) / "loss_curve.png")
    plt.close()
    print(f"Saved loss_curve.png to {plot_save_dir}")

    # Plot 2: Accuracy (Train vs Val)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'], label='Train Accuracy', marker='.')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy', marker='.')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(plot_save_dir) / "accuracy_curve.png")
    plt.close()
    print(f"Saved accuracy_curve.png to {plot_save_dir}")

    # Plot 3: Validation F1-Score
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_f1'], label='Validation F1-Score', marker='.', color='green')
    plt.title('Validation F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(plot_save_dir) / "val_f1_score_curve.png")
    plt.close()
    print(f"Saved val_f1_score_curve.png to {plot_save_dir}")

    # Plot 4: Validation Precision & Recall
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_precision'], label='Validation Precision', marker='.', color='purple')
    plt.plot(epochs, history['val_recall'], label='Validation Recall', marker='.', color='orange')
    plt.title('Validation Precision and Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(plot_save_dir) / "val_precision_recall_curve.png")
    plt.close()
    print(f"Saved val_precision_recall_curve.png to {plot_save_dir}")

    # Plot 5: Validation AUC
    if 'val_auc' in history and any(h is not None and h != -1.0 for h in history['val_auc']): # Check if AUC data is valid
        plt.figure(figsize=(10, 6))
        valid_auc_epochs = [e for e, auc_val in zip(epochs, history['val_auc']) if auc_val is not None and auc_val != -1.0]
        valid_auc_values = [auc_val for auc_val in history['val_auc'] if auc_val is not None and auc_val != -1.0]
        if valid_auc_values:
            plt.plot(valid_auc_epochs, valid_auc_values, label='Validation AUC', marker='.', color='cyan')
            plt.title('Validation AUC')
            plt.xlabel('Epochs')
            plt.ylabel('AUC')
            plt.legend()
            plt.grid(True)
            plt.savefig(Path(plot_save_dir) / "val_auc_curve.png")
            plt.close()
            print(f"Saved val_auc_curve.png to {plot_save_dir}")
        else:
            print("No valid AUC data to plot.")
    else:
        print("Validation AUC data not found or invalid in history file.")


    # Plot 6: Learning Rate
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['lr'], label='Learning Rate', marker='.', color='red')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(plot_save_dir) / "learning_rate_curve.png")
    plt.close()
    print(f"Saved learning_rate_curve.png to {plot_save_dir}")

# ——— Evaluation Function (adapted from training script's eval_epoch) ———
def perform_evaluation(model, loader, criterion, device):
    model.eval()
    epoch_loss_sum = 0
    all_probs_epoch = []
    all_targets_epoch = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if criterion:
                loss = criterion(out, y)
                epoch_loss_sum += loss.item() * x.size(0)
            
            probs_batch = torch.sigmoid(out)
            all_probs_epoch.extend(probs_batch.cpu().numpy().flatten())
            all_targets_epoch.extend(y.cpu().numpy().flatten())

    avg_loss = epoch_loss_sum / len(loader.dataset) if criterion and len(loader.dataset) > 0 else 0.0
    
    targets_np = np.array(all_targets_epoch, dtype=int)
    probs_np = np.array(all_probs_epoch)
    preds_np = (probs_np > 0.5).astype(int)

    accuracy = accuracy_score(targets_np, preds_np)
    precision = precision_score(targets_np, preds_np, zero_division=0)
    recall = recall_score(targets_np, preds_np, zero_division=0)
    f1 = f1_score(targets_np, preds_np, zero_division=0)
    
    auc = -1.0 
    if len(np.unique(targets_np)) > 1: # Check if more than one class in targets
        try:
            auc = roc_auc_score(targets_np, probs_np)
        except ValueError as e:
            print(f"Warning: Could not compute AUC: {e}") 
    else:
        print(f"Warning: Only one class present in targets. AUC cannot be computed robustly.")
        if len(targets_np) > 0: # if all targets are same, e.g. all 0s or all 1s
             # AUC is undefined or 0.5 depending on convention, sklearn may error or return 0.0
             # For safety, keeping -1 or checking if probs_np also has one value.
             pass


    return avg_loss, accuracy, precision, recall, f1, auc, targets_np, preds_np, probs_np

# ——— Main Script Logic ———
if __name__ == "__main__":
    # Part 1: Plot Training History
    plot_training_history(CONFIG["HISTORY_FILE_PATH"], CONFIG["PLOT_SAVE_DIR"])

    # Part 2: Evaluate on Test Set
    print("\n--- Test Set Evaluation ---")
    if not Path(CONFIG["MODEL_PATH"]).exists():
        print(f"ERROR: Model file not found at {CONFIG['MODEL_PATH']}. Skipping test evaluation.")
    else:
        # Set seed for reproducible data splitting (important for getting the same test set)
        np.random.seed(CONFIG["RANDOM_SEED"])
        torch.manual_seed(CONFIG["RANDOM_SEED"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(CONFIG["RANDOM_SEED"])

        # Load test data
        test_images, test_labels = get_test_data()
        test_dataset = ELPVArrayDataset(test_images, test_labels, transform=test_tf)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=2, pin_memory=True)

        # Initialize model
        # The timm warning "Mapping deprecated model name xception to current legacy_xception" is normal.
        model = timm.create_model('xception', pretrained=False, num_classes=CONFIG["NUM_CLASSES"]) # pretrained=False as we load weights
        model.load_state_dict(torch.load(CONFIG["MODEL_PATH"], map_location=DEVICE))
        model.to(DEVICE)

        # Define a criterion for loss calculation (optional for metrics, but good for consistency)
        # Using simple BCEWithLogitsLoss without pos_weight for eval loss reporting,
        # as pos_weight is mainly for training gradient adjustment.
        criterion_eval = nn.BCEWithLogitsLoss() 

        # Perform evaluation
        print("Evaluating model on the test set...")
        test_loss, test_acc, test_p, test_r, test_f1, test_auc, true_labels, pred_labels_binary, _ = perform_evaluation(
            model, test_loader, criterion_eval, DEVICE
        )
        
        print(f"\nTest Metrics:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Precision: {test_p:.4f}")
        print(f"  Recall: {test_r:.4f}")
        print(f"  F1-Score: {test_f1:.4f}")
        print(f"  AUC: {test_auc:.4f}")

        # Generate and save confusion matrix
        cm = confusion_matrix(true_labels, pred_labels_binary, labels=[0,1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Crack (0)","Crack (1)"])
        
        plt.figure(figsize=(8,6))
        disp.plot(cmap=plt.cm.Blues, ax=plt.gca()) # Pass ax to plot on current figure
        plt.title("Test Set Confusion Matrix")
        cm_save_path = Path(CONFIG["PLOT_SAVE_DIR"]) / "confusion_matrix_test_set.png"
        plt.savefig(cm_save_path)
        plt.close()
        print(f"Saved confusion_matrix_test_set.png to {cm_save_path}")

    print("\nEvaluation and plotting script finished.")