# elpv_segmentation/config.py
from pathlib import Path

CONFIG = {
    # --- Data Paths ---
    "DATA_BASE_PATH": "/home/csegpuserver/elpv/unet/crack_segmentation", # IMPORTANT: Verify this
    "TRAIN_IMG_SUBDIR": "train/img",
    "TRAIN_MASK_SUBDIR": "train/ann",
    "VAL_IMG_SUBDIR": "val/img",
    "VAL_MASK_SUBDIR": "val/ann",
    "TEST_IMG_SUBDIR": "test_mix/img",
    "TEST_MASK_SUBDIR": "test_mix/ann",

    # --- Model & Training Hyperparameters ---
    "NUM_CLASSES": 5, # background, crack, cross crack, busbar, dark area
    "INPUT_SIZE": (256, 256), # Changed to tuple (Height, Width) for clarity, though (256) might work if your get_transforms expects int for square
    "BATCH_SIZE": 12,
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 1e-4,
    "NUM_EPOCHS": 5000,
    "GRADIENT_CLIP_NORM": 1.0,
    "SCHEDULER_PATIENCE": 10, # For ReduceLROnPlateau (LR reduction)
    "PATIENCE_EARLY_STOPPING": 100, # For early stopping the training run

    # --- Checkpointing & Resuming ---
    # Set to None to train from scratch, or path to a .pth file to resume
    "RESUME_CHECKPOINT_PATH": None, # Example: "checkpoints/elpv_convnext_intermediate/checkpoint_epoch_10.pth"
                                    # Or: "/home/csegpuserver/elpv/unet/best_elpv_unet_vgg16bn_orig_data.pth" IF you intend to resume from VGG16 weights (not recommended for full resume)

    # It's good practice to name checkpoints based on the new model
    "MODEL_BEST_SAVE_PATH": "checkpoints/elpv_convnext_best.pth",
    "MODEL_CHECKPOINT_DIR": "checkpoints/elpv_convnext_intermediate",
    "CHECKPOINT_EVERY_N_EPOCHS": 10,
    "HISTORY_SAVE_PATH": "checkpoints/elpv_convnext_training_history.json",

    # --- Reproducibility & Environment ---
    "RANDOM_SEED": 42,
    "NUM_WORKERS": 2, # Added: Number of workers for DataLoader

    # --- Miscellaneous ---
    "CLASS_NAMES": ["Background", "Crack", "Cross Crack", "Busbar", "Dark Area"],

    # --- Dummy Data Generation (if applicable, ensure these keys exist if create_dummy_data_if_needed uses them) ---
    # "DUMMY_NUM_TRAIN_SAMPLES": 20, (Example, if used by your dummy data function)
    # "DUMMY_NUM_VAL_SAMPLES": 10,   (Example)
    # "DUMMY_NUM_TEST_SAMPLES": 10,  (Example)
}

# Note: Directory creation for checkpoints is now handled in the training script.
# You can remove the Path(...).mkdir(...) line from here if it was solely for this.
# However, if DATA_BASE_PATH or other essential input directories need creation/validation,
# you might keep or add logic here or in a separate setup script.
