# elpv_segmentation/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import os # Added for os.path.exists for robustness

# Project specific imports
from config import CONFIG # Assuming CONFIG is loaded from config.py
from dataset import SolarDataset, get_transforms, create_dummy_data_if_needed
from model import UNetWithVGG16BN
from engine import train_epoch_segmentation, eval_epoch_segmentation
from utils import plot_training_history

def main():
    # --- Add resume option to CONFIG (Ideally, add this to your config.py) ---
    # If None, training starts from scratch. Otherwise, specify path to .pth file.
    #CONFIG["RESUME_CHECKPOINT_PATH"] = None 
    # Example: CONFIG["RESUME_CHECKPOINT_PATH"] = "checkpoints_elpv_orig_data/checkpoint_epoch_10.pth"

    # --- Reproducibility & Device ---
    np.random.seed(CONFIG["RANDOM_SEED"])
    torch.manual_seed(CONFIG["RANDOM_SEED"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["RANDOM_SEED"])
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- Data Loading ---
    base_path = Path(CONFIG["DATA_BASE_PATH"])
    create_dummy_data_if_needed(base_path, CONFIG)

    data_transforms = get_transforms(input_size=CONFIG["INPUT_SIZE"])
    
    train_dataset = SolarDataset(root=base_path,
                                 image_folder=CONFIG["TRAIN_IMG_SUBDIR"],
                                 mask_folder=CONFIG["TRAIN_MASK_SUBDIR"],
                                 transforms=data_transforms,
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
    # Initialize model, optimizer, and scheduler first
    model = UNetWithVGG16BN(n_classes=CONFIG["NUM_CLASSES"], pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], 
                            weight_decay=CONFIG["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                                     patience=CONFIG["SCHEDULER_PATIENCE"])

    # --- Checkpoint Resuming Logic ---
    start_epoch = 0
    best_val_iou = 0.0
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_f1": [], "lr": []}

    resume_path = CONFIG.get("RESUME_CHECKPOINT_PATH") # Use .get for safety
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming training from checkpoint: {resume_path}")
        try:
            saved = torch.load(resume_path, map_location=DEVICE, weights_only=False)

            # --- Case 1: full checkpoint dict ---
            if isinstance(saved, dict) and 'model_state_dict' in saved:
                print("Detected full checkpoint format. Loading model, optimizer, scheduler, and history.")
                model.load_state_dict(saved['model_state_dict'])
                optimizer.load_state_dict(saved['optimizer_state_dict'])
                
                if 'scheduler_state_dict' in saved:
                    scheduler.load_state_dict(saved['scheduler_state_dict'])
                else:
                    print("Warning: Scheduler state not found. Starting scheduler from scratch.")
                
                start_epoch     = saved.get('epoch', 0)
                best_val_iou    = saved.get('best_val_iou', 0.0)
                loaded_history  = saved.get('history', None)
                if loaded_history:
                    history = loaded_history
                    # Pad LR history if needed
                    lr_list = history.get('lr', [])
                    if len(lr_list) < start_epoch:
                        print(f"Padding LR history from {len(lr_list)} to {start_epoch} entries.")
                        lr_list.extend([optimizer.param_groups[0]['lr']] * (start_epoch - len(lr_list)))
                        history['lr'] = lr_list

            # --- Case 2: raw state_dict only ---
            else:
                print("Detected raw state_dict format. Loading only model weights.")
                model.load_state_dict(saved)
                # Leave optimizer, scheduler, history at their initialized defaults:
                start_epoch  = 0
                best_val_iou = 0.0

            print(f"Resumed from epoch {start_epoch}. Best Val IoU = {best_val_iou:.4f}.")

        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting training from scratch.")
            start_epoch = 0
            best_val_iou = 0.0
            history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_f1": [], "lr": []}
            # Re-initialize model, optimizer, scheduler to ensure clean state if loading failed partially
            model = UNetWithVGG16BN(n_classes=CONFIG["NUM_CLASSES"], pretrained=True).to(DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], 
                                    weight_decay=CONFIG["WEIGHT_DECAY"])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                                             patience=CONFIG["SCHEDULER_PATIENCE"])

    else:
        if resume_path: # resume_path was set but file does not exist
            print(f"Checkpoint not found at {resume_path}. Starting training from scratch.")
        else: # resume_path was None
            print("Starting training from scratch.")


    # --- Training Loop ---
    print(f"Starting ELPV segmentation training from epoch {start_epoch}...")
    for e in range(start_epoch, CONFIG["NUM_EPOCHS"]):
        current_epoch_num = e + 1 # For user display (1-indexed)
        
        tr_loss = train_epoch_segmentation(
            model, train_loader, optimizer, criterion, DEVICE, CONFIG["GRADIENT_CLIP_NORM"]
        )
        val_loss, val_iou, val_f1, _, _ = eval_epoch_segmentation(
            model, val_loader, criterion, DEVICE, CONFIG["NUM_CLASSES"]
        )

        # Append new results to history (even if history was loaded)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_f1"].append(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        history["lr"].append(current_lr)

        print(f"Epoch {current_epoch_num}/{CONFIG['NUM_EPOCHS']}: "
              f"LR: {current_lr:.1e} | "
              f"Train L: {tr_loss:.4f} | "
              f"Val L: {val_loss:.4f}, IoU: {val_iou:.4f}, F1: {val_f1:.4f}")

        scheduler.step(val_loss)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            no_improvement_epochs = 0 # Reset counter for early stopping when a new best is found
            torch.save({
                 'epoch': current_epoch_num,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict(),
                 'best_val_iou': best_val_iou,
                 'history': history
             }, CONFIG["MODEL_BEST_SAVE_PATH"])            
            print(f"  ↳ New best Val IoU: {best_val_iou:.4f}. Model saved to {CONFIG['MODEL_BEST_SAVE_PATH']}")
        else:
            # Increment no_improvement_epochs only if not resuming (or handle its loading)
            # For simplicity, if resuming, this counter effectively resets unless loaded.
            # If 'no_improvement_epochs' was saved in checkpoint, load it too.
            # Current checkpoint saves 'best_val_iou' but not 'no_improvement_epochs'.
            # Let's assume we reset it on resume or if it's not critical to resume this counter.
            if e == start_epoch and resume_path and os.path.exists(resume_path): # First epoch after resume
                 no_improvement_epochs = 0 # Reset or load if saved in checkpoint
            else:
                 no_improvement_epochs = getattr(scheduler, 'num_bad_epochs', 0) if hasattr(scheduler, 'num_bad_epochs') else (no_improvement_epochs + 1 if e > start_epoch else 0)


        if (current_epoch_num) % CONFIG["CHECKPOINT_EVERY_N_EPOCHS"] == 0:
            chk_path = Path(CONFIG["MODEL_CHECKPOINT_DIR"]) / f"checkpoint_epoch_{current_epoch_num}.pth"
            torch.save({
                'epoch': current_epoch_num, # Save the number of epochs completed
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_iou': best_val_iou,
                'history': history
            }, chk_path)
            print(f"  ↳ Checkpoint saved to {chk_path} after epoch {current_epoch_num}")

        # Early stopping logic (needs no_improvement_epochs to be consistent)
        # Consider loading/saving 'no_improvement_epochs' in checkpoint if strict continuation is needed.
        # For now, the reset logic above for 'no_improvement_epochs' is a simple approach.
        # A more robust way for early stopping counter:
        if 'no_improvement_epochs_counter' not in locals() and resume_path and os.path.exists(resume_path):
            # If resuming and you saved this counter in the checkpoint, load it.
            # no_improvement_epochs_counter = checkpoint.get('no_improvement_epochs_counter', 0)
            pass # For now, it resets based on scheduler.num_bad_epochs logic

        if val_iou <= best_val_iou: # If current val_iou is not better than best_val_iou
             current_no_improvement = getattr(scheduler, 'num_bad_epochs', 0) # For ReduceLROnPlateau
        else: # New best found
             current_no_improvement = 0
        
        if current_no_improvement >= CONFIG["PATIENCE_EARLY_STOPPING"]:
            print(f"Early stopping at epoch {current_epoch_num} due to no improvement for {current_no_improvement} epochs based on scheduler's count.")
            break
    
    # Save final history (contains all epochs, including resumed ones)
    with open(CONFIG["HISTORY_SAVE_PATH"], 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {CONFIG['HISTORY_SAVE_PATH']}")

    # --- Plotting ---
    plot_training_history(history, CONFIG["HISTORY_SAVE_PATH"])

    # --- Final Evaluation on Test Set ---
    if Path(CONFIG["MODEL_BEST_SAVE_PATH"]).exists():
        print(f"\nLoading best model from {CONFIG['MODEL_BEST_SAVE_PATH']} for final test evaluation...")
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