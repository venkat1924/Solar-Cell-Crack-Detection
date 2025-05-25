# elpv_segmentation/config.py
from pathlib import Path

CONFIG = {
    "DATA_BASE_PATH": "/home/csegpuserver/elpv/unet/crack_segmentation", # IMPORTANT: Change this
    "TRAIN_IMG_SUBDIR": "train/img",
    "TRAIN_MASK_SUBDIR": "train/ann",
    "VAL_IMG_SUBDIR": "val/img",
    "VAL_MASK_SUBDIR": "val/ann",
    "TEST_IMG_SUBDIR": "test_mix/img",      # Or "test_crack/img" or "testset/img"
    "TEST_MASK_SUBDIR": "test_mix/ann",     # Or "test_crack/ann" or "testset/ann"
    "BATCH_SIZE": 12,
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 1e-4,
    "NUM_EPOCHS": 50,
    "NUM_CLASSES": 5, # background, crack, cross crack, busbar, dark area
    "INPUT_SIZE": 256, # Used in FixResize, single int for square
    "PATIENCE_EARLY_STOPPING": 20,
    "SCHEDULER_PATIENCE": 10,
    "MODEL_BEST_SAVE_PATH": "best_elpv_unet_vgg16bn_orig_data.pth",
    "MODEL_CHECKPOINT_DIR": "checkpoints_elpv_orig_data",
    "CHECKPOINT_EVERY_N_EPOCHS": 10,
    "HISTORY_SAVE_PATH": "training_history_elpv_orig_data.json",
    "RANDOM_SEED": 42,
    "GRADIENT_CLIP_NORM": 1.0,
    "CLASS_NAMES": ["Background", "Crack", "Cross Crack", "Busbar", "Dark Area"]
}

# Create checkpoint directory if it doesn't exist
Path(CONFIG["MODEL_CHECKPOINT_DIR"]).mkdir(parents=True, exist_ok=True)