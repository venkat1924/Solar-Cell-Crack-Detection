import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
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

from elpv_dataset.utils import load_dataset 
import timm


CONFIG = {
    "DATASET_PATH": None,               
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 1e-4,              
    "WEIGHT_DECAY": 1e-4,
    "NUM_EPOCHS": 10000,
    "NUM_CLASSES": 1,                   
    "PATIENCE_EARLY_STOPPING": 100,
    "SCHEDULER_PATIENCE": 5,            
    "MODEL_BEST_SAVE_PATH": "best_microcrack_detector.pth",
    "MODEL_CHECKPOINT_DIR": "checkpoints", 
    "CHECKPOINT_EVERY_N_EPOCHS": 10,    
    "HISTORY_SAVE_PATH": "training_history.json",
    "TRAIN_SPLIT_RATIO": 0.7,
    "VAL_SPLIT_RATIO": 0.15,
    "TEST_SPLIT_RATIO": 0.15,
    "RANDOM_SEED": 42,
    "GRADIENT_CLIP_NORM": 1.0,          
}
assert abs(CONFIG["TRAIN_SPLIT_RATIO"] +
           CONFIG["VAL_SPLIT_RATIO"] +
           CONFIG["TEST_SPLIT_RATIO"] - 1.0) < 1e-9, \
    "Train/Val/Test ratios must sum to 1.0"

Path(CONFIG["MODEL_CHECKPOINT_DIR"]).mkdir(parents=True, exist_ok=True)

np.random.seed(CONFIG["RANDOM_SEED"])
torch.manual_seed(CONFIG["RANDOM_SEED"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["RANDOM_SEED"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

print("Loading ELPV dataset...")

try:
    images_np, proba_np, types_np = load_dataset()
except NameError: 
    print("Warning: load_dataset() not found. Using dummy data for script structure testing.")
    images_np = np.random.rand(100, 300, 300).astype(np.float32) 
    proba_np = np.random.rand(100).astype(np.float32)
    types_np = np.array(['mono'] * 50 + ['poly'] * 50)


labels_np = (proba_np > 0.5).astype(np.int64)
print(f"Dataset loaded. Total samples: {len(labels_np)}")
print(f"Class distribution: 0s={np.sum(labels_np == 0)}, 1s={np.sum(labels_np == 1)}")


idx = np.arange(len(labels_np))
if len(np.unique(labels_np)) > 1:
    train_idx, temp_idx, train_lbls_strat, temp_lbl_strat = train_test_split(
        idx, labels_np,
        test_size=(CONFIG["VAL_SPLIT_RATIO"] + CONFIG["TEST_SPLIT_RATIO"]),
        stratify=labels_np,
        random_state=CONFIG["RANDOM_SEED"]
    )
    val_prop_for_split = CONFIG["VAL_SPLIT_RATIO"] / (CONFIG["VAL_SPLIT_RATIO"] + CONFIG["TEST_SPLIT_RATIO"])

    if len(np.unique(temp_lbl_strat)) > 1 :
        val_idx, test_idx, _, _ = train_test_split(
            temp_idx, temp_lbl_strat,
            test_size=(1.0 - val_prop_for_split), 
            stratify=temp_lbl_strat,
            random_state=CONFIG["RANDOM_SEED"]
        )
    else: 
        print("Warning: Not enough class diversity for stratified val/test split. Using non-stratified.")
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1.0 - val_prop_for_split),
            random_state=CONFIG["RANDOM_SEED"]
        )
else: 
    print("Warning: Only one class present in the dataset. Using non-stratified split.")
    train_idx, temp_idx = train_test_split(idx, test_size=(CONFIG["VAL_SPLIT_RATIO"] + CONFIG["TEST_SPLIT_RATIO"]), random_state=CONFIG["RANDOM_SEED"])
    val_prop_for_split = CONFIG["VAL_SPLIT_RATIO"] / (CONFIG["VAL_SPLIT_RATIO"] + CONFIG["TEST_SPLIT_RATIO"])
    val_idx, test_idx = train_test_split(temp_idx, test_size=(1.0 - val_prop_for_split), random_state=CONFIG["RANDOM_SEED"])


train_imgs, train_lbls = images_np[train_idx], labels_np[train_idx]
val_imgs,   val_lbls   = images_np[val_idx],   labels_np[val_idx]
test_imgs,  test_lbls  = images_np[test_idx],  labels_np[test_idx]

print(f"Train samples: {len(train_imgs)} (0s: {np.sum(train_lbls == 0)}, 1s: {np.sum(train_lbls == 1)})")
print(f"Val samples: {len(val_imgs)} (0s: {np.sum(val_lbls == 0)}, 1s: {np.sum(val_lbls == 1)})")
print(f"Test samples: {len(test_imgs)} (0s: {np.sum(test_lbls == 0)}, 1s: {np.sum(test_lbls == 1)})")

train_tf = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.OneOf([
        A.GaussNoise(p=0.5),
        A.GaussianBlur(blur_limit=(3,5), p=0.5), 
    ], p=0.3), 
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), 
    ToTensorV2()
])
val_tf = A.Compose([ 
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

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
        
        if img.dtype != np.uint8:
            if img.max() <= 1.0 and img.min() >=0.0 : 
                 img = (img * 255).astype(np.uint8)
            elif img.max() <= 255 and img.min() >=0: 
                 img = img.astype(np.uint8)

        if self.transform:
            img = self.transform(image=img)['image']
        lbl = torch.tensor([self.lbls[i]], dtype=torch.float32)
        return img, lbl

train_ds = ELPVArrayDataset(train_imgs, train_lbls, transform=train_tf)
val_ds   = ELPVArrayDataset(val_imgs,   val_lbls,   transform=val_tf)
test_ds  = ELPVArrayDataset(test_imgs,  test_lbls,  transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"],
                          shuffle=True,  num_workers=2, pin_memory=True, drop_last=True) 
val_loader   = DataLoader(val_ds,   batch_size=CONFIG["BATCH_SIZE"],
                          shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=CONFIG["BATCH_SIZE"],
                          shuffle=False, num_workers=2, pin_memory=True)

model = timm.create_model('xception', pretrained=True,
                          num_classes=CONFIG["NUM_CLASSES"])
model.to(DEVICE)

num_pos_train = np.sum(train_lbls == 1)
num_neg_train = len(train_lbls) - num_pos_train

if num_pos_train > 0 and num_neg_train > 0:
    pos_weight_val = num_neg_train / num_pos_train
    print(f"Calculated pos_weight: {pos_weight_val:.4f} (Neg/Pos = {num_neg_train}/{num_pos_train})")
    pos_weight = torch.tensor([pos_weight_val], device=DEVICE)
else:
    print("Warning: num_pos_train or num_neg_train is zero. Using default pos_weight=1.0.")
    pos_weight = torch.tensor([1.0], device=DEVICE) 

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
optimizer = optim.AdamW(model.parameters(),
                        lr=CONFIG["LEARNING_RATE"],
                        weight_decay=CONFIG["WEIGHT_DECAY"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max', 
    factor=0.1,
    patience=CONFIG["SCHEDULER_PATIENCE"]
)

def train_epoch(m, loader, optim, crit, grad_clip_norm):
    m.train()
    epoch_loss_sum = 0
    all_preds = []
    all_targets = []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        out = m(x)
        loss = crit(out, y)
        loss.backward()
        
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip_norm)
            
        optim.step()

        epoch_loss_sum += loss.item() * x.size(0)
        preds_batch = (torch.sigmoid(out) > 0.5).long()
        all_preds.extend(preds_batch.cpu().numpy().flatten())
        all_targets.extend(y.cpu().numpy().flatten())
        
    avg_loss = epoch_loss_sum / len(loader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    return avg_loss, accuracy

def eval_epoch(m, loader, crit):
    m.eval()
    epoch_loss_sum = 0
    all_probs_epoch = []
    all_targets_epoch = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = m(x)
            loss = crit(out, y)
            epoch_loss_sum += loss.item() * x.size(0)
            
            probs_batch = torch.sigmoid(out)
            all_probs_epoch.extend(probs_batch.cpu().numpy().flatten())
            all_targets_epoch.extend(y.cpu().numpy().flatten())

    avg_loss = epoch_loss_sum / len(loader.dataset)
    
    targets_np = np.array(all_targets_epoch, dtype=int)
    probs_np = np.array(all_probs_epoch)
    preds_np = (probs_np > 0.5).astype(int)

    accuracy = accuracy_score(targets_np, preds_np)
    precision = precision_score(targets_np, preds_np, zero_division=0)
    recall = recall_score(targets_np, preds_np, zero_division=0)
    f1 = f1_score(targets_np, preds_np, zero_division=0)
    
    auc = -1.0 
    if len(np.unique(targets_np)) > 1:
        try:
            auc = roc_auc_score(targets_np, probs_np)
        except ValueError as e:
            print(f"Warning: Could not compute AUC during validation: {e}") 
    else:
        print(f"Warning: Only one class present in validation targets. AUC cannot be computed robustly.")

    return avg_loss, accuracy, precision, recall, f1, auc, targets_np, probs_np


best_val_f1 = 0
no_improvement_epochs = 0
history = { 
    "train_loss": [], "train_acc": [],
    "val_loss": [], "val_acc": [], "val_precision": [], "val_recall": [], "val_f1": [], "val_auc": [],
    "lr": []
}

print("Starting training...")
for e in range(CONFIG["NUM_EPOCHS"]):
    tr_loss, tr_acc = train_epoch(
        model, train_loader, optimizer, criterion, CONFIG["GRADIENT_CLIP_NORM"]
    )

    val_loss, val_acc, val_p, val_r, val_f1, val_auc, _, _ = eval_epoch(
        model, val_loader, criterion
    )

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_precision"].append(val_p)
    history["val_recall"].append(val_r)
    history["val_f1"].append(val_f1)
    history["val_auc"].append(val_auc)
    current_lr = optimizer.param_groups[0]['lr']
    history["lr"].append(current_lr)

    print(f"Epoch {e+1}/{CONFIG['NUM_EPOCHS']}: "
          f"LR: {current_lr:.1e} | "
          f"Train L: {tr_loss:.4f}, A: {tr_acc:.4f} | "
          f"Val L: {val_loss:.4f}, A: {val_acc:.4f}, P: {val_p:.4f}, R: {val_r:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

    scheduler.step(val_f1) 

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        no_improvement_epochs = 0
        torch.save(model.state_dict(), CONFIG["MODEL_BEST_SAVE_PATH"])
        print(f"  ↳ New best Val F1: {best_val_f1:.4f}. Model saved to {CONFIG['MODEL_BEST_SAVE_PATH']}")
    else:
        no_improvement_epochs += 1

    if (e + 1) % CONFIG["CHECKPOINT_EVERY_N_EPOCHS"] == 0:
        checkpoint_path = Path(CONFIG["MODEL_CHECKPOINT_DIR"]) / f"epoch_{e+1:03d}_f1_{val_f1:.4f}.pth"
        torch.save({
            'epoch': e + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_f1': best_val_f1,
            'val_loss': val_loss, 
            'val_f1': val_f1
        }, checkpoint_path)
        print(f"  ↳ Periodic checkpoint saved to {checkpoint_path}")

    if no_improvement_epochs >= CONFIG["PATIENCE_EARLY_STOPPING"]:
        print(f"Early stopping at epoch {e+1} due to no improvement in Val F1 for {CONFIG['PATIENCE_EARLY_STOPPING']} epochs.")
        break

with open(CONFIG["HISTORY_SAVE_PATH"], 'w') as f:
    json.dump(history, f, indent=4)
print(f"Training history saved to {CONFIG['HISTORY_SAVE_PATH']}")

plt.figure(figsize=(18, 10))
plt.subplot(2,3,1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.title("Loss"); plt.grid(True); plt.legend()

plt.subplot(2,3,2)
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.title("Accuracy"); plt.grid(True); plt.legend()

plt.subplot(2,3,3)
plt.plot(history["val_f1"], label="Val F1")
plt.title("Validation F1-Score"); plt.grid(True); plt.legend()

plt.subplot(2,3,4)
plt.plot(history["val_precision"], label="Val Precision")
plt.plot(history["val_recall"], label="Val Recall")
plt.title("Validation P & R"); plt.grid(True); plt.legend()

plt.subplot(2,3,5)
plt.plot(history["val_auc"], label="Val AUC")
plt.title("Validation AUC"); plt.grid(True); plt.legend()

plt.subplot(2,3,6)
plt.plot(history["lr"], label="Learning Rate")
plt.title("Learning Rate"); plt.grid(True); plt.legend()

plt.tight_layout(); plt.show()

if Path(CONFIG["MODEL_BEST_SAVE_PATH"]).exists():
    print(f"\nLoading best model from {CONFIG['MODEL_BEST_SAVE_PATH']} for final test evaluation...")
    model.load_state_dict(torch.load(CONFIG["MODEL_BEST_SAVE_PATH"])) 

    test_loss, test_acc, test_p, test_r, test_f1, test_auc, test_lbls, test_probs = eval_epoch(
        model, test_loader, criterion 
    )
    test_preds = (test_probs > 0.5).astype(int)

    print("\n--- Test Set Evaluation ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_p:.4f}")
    print(f"Test Recall: {test_r:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    print("\nTest Confusion Matrix:")
    cm = confusion_matrix(test_lbls, test_preds, labels=[0,1]) 
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Crack (0)","Crack (1)"])
    disp.plot(cmap=plt.cm.Blues); plt.title("Test Confusion Matrix"); plt.show()
else:
    print(f"ERROR: Best model not found at {CONFIG['MODEL_BEST_SAVE_PATH']}. Skipping test evaluation.")

print("\n--- End of Script ---")
