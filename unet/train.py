# elpv_segmentation/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import os

# Project specific imports
from config import CONFIG # Assuming CONFIG is loaded from config.py
from dataset import SolarDataset, get_transforms, create_dummy_data_if_needed # Ensure these are correctly implemented
from model import UNetWithConvNeXtBase # Using the ConvNeXt model
from engine import train_epoch_segmentation, eval_epoch_segmentation # Ensure these are correctly implemented
from utils import plot_training_history # Ensure this is correctly implemented

def main():
    # --- Create Output Directories (from CONFIG) ---
    # These paths are expected to be defined in CONFIG
    Path(CONFIG["MODEL_CHECKPOINT_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["MODEL_BEST_SAVE_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(CONFIG["HISTORY_SAVE_PATH"]).parent.mkdir(parents=True, exist_ok=True)

    # --- Reproducibility & Device ---
    np.random.seed(CONFIG["RANDOM_SEED"])
    torch.manual_seed(CONFIG["RANDOM_SEED"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["RANDOM_SEED"])
        # For full reproducibility, consider these, though they might impact performance:
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- Data Loading ---
    base_path = Path(CONFIG["DATA_BASE_PATH"])
    # Ensure this function correctly uses CONFIG if needed for parameters like dummy sample counts
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
                              num_workers=CONFIG["NUM_WORKERS"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                            num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                             num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)

    # --- Model, Loss, Optimizer ---
    model = UNetWithConvNeXtBase(n_classes=CONFIG["NUM_CLASSES"], pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss() # Add ignore_index if applicable, e.g., nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"],
                            weight_decay=CONFIG["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                     patience=CONFIG["SCHEDULER_PATIENCE"])

    # --- Checkpoint Resuming Logic ---
    start_epoch = 0
    best_val_iou = 0.0
    no_improvement_counter_es = 0
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_f1": [], "lr": []}

    resume_path_str = CONFIG.get("RESUME_CHECKPOINT_PATH") # Use .get for safety if key might be missing
    if resume_path_str and resume_path_str.lower() != 'none' and os.path.exists(resume_path_str):
        resume_path = Path(resume_path_str)
        print(f"Resuming training from checkpoint: {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location=DEVICE, weights_only=False)

            if 'model_state_dict' not in checkpoint:
                print("Checkpoint seems to be only model weights. Loading model_state_dict only.")
                model.load_state_dict(checkpoint)
            else:
                print("Loading full checkpoint (model, optimizer, scheduler, history, etc.).")
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                else:
                    print("Warning: Scheduler state not found in checkpoint. Initializing scheduler anew.")

                # 'epoch' in checkpoint is the last completed epoch (1-indexed)
                # start_epoch for range() should be 0-indexed next epoch to run
                start_epoch = checkpoint.get('epoch', 0) 
                best_val_iou = checkpoint.get('best_val_iou', 0.0)
                loaded_history = checkpoint.get('history', None)
                if loaded_history:
                    history = loaded_history
                    lr_list = history.get('lr', [])
                    if len(lr_list) < start_epoch: # start_epoch is number of completed epochs
                        print(f"Padding LR history from {len(lr_list)} to {start_epoch} entries.")
                        last_known_lr = optimizer.param_groups[0]['lr']
                        lr_list.extend([last_known_lr] * (start_epoch - len(lr_list)))
                        history['lr'] = lr_list
                
                no_improvement_counter_es = checkpoint.get('no_improvement_counter_es', 0)

            print(f"Resumed. Next epoch to run: {start_epoch + 1}. Best Val IoU: {best_val_iou:.4f}. No improvement count: {no_improvement_counter_es}")

        except Exception as e:
            print(f"Error loading checkpoint: {e}. Training from scratch.")
            start_epoch = 0
            best_val_iou = 0.0
            no_improvement_counter_es = 0
            history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_f1": [], "lr": []}
            model = UNetWithConvNeXtBase(n_classes=CONFIG["NUM_CLASSES"], pretrained=True).to(DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=CONFIG["WEIGHT_DECAY"])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=CONFIG["SCHEDULER_PATIENCE"])
    else:
        if resume_path_str and resume_path_str.lower() != 'none':
            print(f"Checkpoint not found at {resume_path_str}. Starting training from scratch.")
        else:
            print("No resume path specified or set to None. Starting training from scratch.")

    # --- Training Loop ---
    print(f"Starting ELPV segmentation training. Target epochs: {CONFIG['NUM_EPOCHS']}.")

    for epoch_idx in range(start_epoch, CONFIG["NUM_EPOCHS"]):
        current_epoch_num = epoch_idx + 1 # User-facing epoch number (1-indexed)

        tr_loss = train_epoch_segmentation(
            model, train_loader, optimizer, criterion, DEVICE, CONFIG.get("GRADIENT_CLIP_NORM") # Use .get for optional param
        )
        val_loss, val_iou, val_f1, _, _ = eval_epoch_segmentation( # Assuming eval returns these 5 values
            model, val_loader, criterion, DEVICE, CONFIG["NUM_CLASSES"]
        )

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

        # Prepare checkpoint data (will be updated before saving if it's a new best)
        checkpoint_data = {
            'epoch': current_epoch_num, # The epoch number that just completed
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_iou': best_val_iou, 
            'history': history,
            'no_improvement_counter_es': no_improvement_counter_es
        }

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            no_improvement_counter_es = 0 # Reset counter
            checkpoint_data['best_val_iou'] = best_val_iou # Update with new best
            checkpoint_data['no_improvement_counter_es'] = no_improvement_counter_es # Update with reset counter
            
            best_model_path = Path(CONFIG["MODEL_BEST_SAVE_PATH"])
            torch.save(checkpoint_data, best_model_path)
            print(f"  ↳ 🎉 New best Val IoU: {best_val_iou:.4f}. Model saved to {best_model_path}")
        else:
            no_improvement_counter_es += 1
            checkpoint_data['no_improvement_counter_es'] = no_improvement_counter_es # Update with incremented counter
            print(f"  ↳ No improvement in Val IoU for {no_improvement_counter_es} epochs (Early Stopping Patience: {CONFIG['PATIENCE_EARLY_STOPPING']}).")


        if (current_epoch_num % CONFIG["CHECKPOINT_EVERY_N_EPOCHS"] == 0) or (current_epoch_num == CONFIG["NUM_EPOCHS"]):
            chk_dir = Path(CONFIG["MODEL_CHECKPOINT_DIR"])
            chk_path = chk_dir / f"checkpoint_epoch_{current_epoch_num}.pth"
            # Save potentially updated checkpoint_data (if best_val_iou or no_improvement_counter_es changed)
            torch.save(checkpoint_data, chk_path)
            print(f"  ↳ 💾 Checkpoint saved to {chk_path}")

        if no_improvement_counter_es >= CONFIG["PATIENCE_EARLY_STOPPING"]:
            print(f"🛑 Early stopping triggered at epoch {current_epoch_num} due to no Val IoU improvement for {no_improvement_counter_es} epochs.")
            break
    
    # --- Save Final History ---
    history_path = Path(CONFIG["HISTORY_SAVE_PATH"])
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")

    # --- Plotting ---
    if history["val_iou"]: 
        plot_path_str = str(history_path.parent / f"{history_path.stem}_plot.png")
        plot_training_history(history, plot_path_str) 
        print(f"Training plot saved to {plot_path_str}")
    else:
        print("No training history to plot (e.g., training didn't complete any epochs).")

    # --- Final Evaluation on Test Set ---
    best_model_path = Path(CONFIG["MODEL_BEST_SAVE_PATH"])
    if best_model_path.exists():
        print(f"\nLoading best model from {best_model_path} for final test evaluation...")
        # Load the entire checkpoint first
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        # Then load the model's state dictionary
        model.load_state_dict(checkpoint['model_state_dict']) 
        model.eval() # Ensure model is in evaluation mode

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
            print(f"  Class '{c_name}': IoU = {iou_cls[i]:.4f}, F1 = {f1_cls[i]:.4f}")
    else:
        print(f"Best model ({best_model_path}) not found. Skipping test evaluation.")

    print("\n--- End of Script ---")

if __name__ == '__main__':
    main()
