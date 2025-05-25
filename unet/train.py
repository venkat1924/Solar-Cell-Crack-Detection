# elpv_segmentation/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json

# Project specific imports
from config import CONFIG
from dataset import SolarDataset, get_transforms, create_dummy_data_if_needed
from model import UNetWithVGG16BN
from engine import train_epoch_segmentation, eval_epoch_segmentation
from utils import plot_training_history

def main():
    # --- Reproducibility & Device ---
    np.random.seed(CONFIG["RANDOM_SEED"])
    torch.manual_seed(CONFIG["RANDOM_SEED"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["RANDOM_SEED"])
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- Data Loading ---
    base_path = Path(CONFIG["DATA_BASE_PATH"])
    create_dummy_data_if_needed(base_path, CONFIG) # Create dummy data if real data not found

    # Get transforms
    # You can customize mean/std if your dataset statistics differ significantly
    # from ImageNet, or pass them via CONFIG if needed.
    data_transforms = get_transforms(input_size=CONFIG["INPUT_SIZE"])
    
    train_dataset = SolarDataset(root=base_path,
                                 image_folder=CONFIG["TRAIN_IMG_SUBDIR"],
                                 mask_folder=CONFIG["TRAIN_MASK_SUBDIR"],
                                 transforms=data_transforms, # Using same transform object for simplicity
                                 random_seed=CONFIG["RANDOM_SEED"])

    val_dataset = SolarDataset(root=base_path,
                               image_folder=CONFIG["VAL_IMG_SUBDIR"],
                               mask_folder=CONFIG["VAL_MASK_SUBDIR"],
                               transforms=data_transforms,
                               random_seed=CONFIG["RANDOM_SEED"])

    test_dataset = SolarDataset(root=base_path,
                                image_folder=CONFIG["TEST_IMG_SUBDIR"],
                                mask_folder=CONFIG["TEST_MASK_SUBDIR"],
                                transforms=data_transforms,
                                random_seed=CONFIG["RANDOM_SEED"])

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, 
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, 
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, 
                             num_workers=2, pin_memory=True)

    # --- Model, Loss, Optimizer ---
    model = UNetWithVGG16BN(n_classes=CONFIG["NUM_CLASSES"], pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], 
                            weight_decay=CONFIG["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                                     patience=CONFIG["SCHEDULER_PATIENCE"])

    # --- Training Loop ---
    best_val_iou = 0.0
    no_improvement_epochs = 0
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_f1": [], "lr": []}

    print("Starting ELPV segmentation training...")
    for e in range(CONFIG["NUM_EPOCHS"]):
        tr_loss = train_epoch_segmentation(
            model, train_loader, optimizer, criterion, DEVICE, CONFIG["GRADIENT_CLIP_NORM"]
        )
        val_loss, val_iou, val_f1, _, _ = eval_epoch_segmentation(
            model, val_loader, criterion, DEVICE, CONFIG["NUM_CLASSES"]
        )

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_f1"].append(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        history["lr"].append(current_lr)

        print(f"Epoch {e+1}/{CONFIG['NUM_EPOCHS']}: "
              f"LR: {current_lr:.1e} | "
              f"Train L: {tr_loss:.4f} | "
              f"Val L: {val_loss:.4f}, IoU: {val_iou:.4f}, F1: {val_f1:.4f}")

        scheduler.step(val_loss)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            no_improvement_epochs = 0
            torch.save(model.state_dict(), CONFIG["MODEL_BEST_SAVE_PATH"])
            print(f"  ↳ New best Val IoU: {best_val_iou:.4f}. Model saved to {CONFIG['MODEL_BEST_SAVE_PATH']}")
        else:
            no_improvement_epochs += 1

        if (e + 1) % CONFIG["CHECKPOINT_EVERY_N_EPOCHS"] == 0:
            chk_path = Path(CONFIG["MODEL_CHECKPOINT_DIR"]) / f"checkpoint_epoch_{e+1}.pth"
            torch.save({
                'epoch': e + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_iou': best_val_iou,
                'history': history # Save history in checkpoint too
            }, chk_path)
            print(f"  ↳ Checkpoint saved to {chk_path}")

        if no_improvement_epochs >= CONFIG["PATIENCE_EARLY_STOPPING"]:
            print(f"Early stopping at epoch {e+1} due to no improvement for {no_improvement_epochs} epochs.")
            break
    
    with open(CONFIG["HISTORY_SAVE_PATH"], 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {CONFIG['HISTORY_SAVE_PATH']}")

    # --- Plotting ---
    plot_training_history(history, CONFIG["HISTORY_SAVE_PATH"])

    # --- Final Evaluation on Test Set ---
    if Path(CONFIG["MODEL_BEST_SAVE_PATH"]).exists():
        print(f"\nLoading best model from {CONFIG['MODEL_BEST_SAVE_PATH']} for final test evaluation...")
        # Load best model state for testing
        model.load_state_dict(torch.load(CONFIG["MODEL_BEST_SAVE_PATH"], map_location=DEVICE))
        
        test_loss, test_iou, test_f1, iou_cls, f1_cls = eval_epoch_segmentation(
            model, test_loader, criterion, DEVICE, CONFIG["NUM_CLASSES"]
        )
        print("\n--- Test Set Evaluation ---")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Macro IoU (mIoU): {test_iou:.4f}")
        print(f"Test Macro F1-Score: {test_f1:.4f}")
        
        print("\nPer-class Metrics (Test Set):")
        class_names = CONFIG.get("CLASS_NAMES", [f"Class {i}" for i in range(CONFIG["NUM_CLASSES"])])
        for i in range(CONFIG["NUM_CLASSES"]):
            c_name = class_names[i] if i < len(class_names) else f"Class {i}"
            print(f"  {c_name}: IoU = {iou_cls[i]:.4f}, F1 = {f1_cls[i]:.4f}")
    else:
        print(f"Best model ({CONFIG['MODEL_BEST_SAVE_PATH']}) not found. Skipping test evaluation.")

    print("\n--- End of Script ---")

if __name__ == '__main__':
    main()