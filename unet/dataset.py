# elpv_segmentation/dataset.py
import os
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms.functional as F
import torchvision.transforms as T_vision # For InterpolationMode

def list_images(path):
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    return sorted([os.path.join(path, f) for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and os.path.splitext(f)[1].lower() in img_extensions])

class SolarDataset(VisionDataset):
    def __init__(self,
                 root,
                 image_folder,
                 mask_folder,
                 transforms,
                 mode="train",
                 random_seed=42):
        super().__init__(root, transforms)
        self.image_path = Path(self.root) / image_folder
        self.mask_path = Path(self.root) / mask_folder

        if not self.image_path.exists():
            raise OSError(f"{self.image_path} not found.")
        if not self.mask_path.exists():
            raise OSError(f"{self.mask_path} not found.")

        self.image_list = np.array(list_images(self.image_path))
        self.mask_list = np.array(list_images(self.mask_path))

        if len(self.image_list) == 0:
            raise FileNotFoundError(f"No images found in {self.image_path}")
        if len(self.mask_list) == 0:
            raise FileNotFoundError(f"No masks found in {self.mask_path}")

        img_basenames = {os.path.splitext(os.path.basename(p))[0] for p in self.image_list}
        mask_basenames = {os.path.splitext(os.path.basename(p))[0] for p in self.mask_list}

        if img_basenames != mask_basenames:
            print(f"Warning: Image and mask basenames do not perfectly match in {self.image_path} and {self.mask_path}.")
            common_basenames = sorted(list(img_basenames.intersection(mask_basenames)))
            
            img_map = {os.path.splitext(os.path.basename(p))[0]: p for p in self.image_list}
            mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in self.mask_list}
            
            self.image_list = np.array([img_map[bn] for bn in common_basenames if bn in img_map])
            self.mask_list = np.array([mask_map[bn] for bn in common_basenames if bn in mask_map])

            if len(self.image_list) != len(self.mask_list) or len(self.image_list) == 0:
                 raise ValueError(f"Could not reconcile image and mask lists. Found {len(self.image_list)} common items.")
            print(f"After filtering for common basenames, using {len(self.image_list)} pairs.")

        np.random.seed(random_seed)
        indices = np.arange(len(self.image_list))
        np.random.shuffle(indices)
        self.image_list = self.image_list[indices]
        self.mask_list = self.mask_list[indices]
        
        print(f"Initialized SolarDataset from {self.image_path} and {self.mask_path}: Found {len(self.image_list)} image-mask pairs.")

    def __len__(self):
        return len(self.image_list)

    def __getname__(self, index):
        image_name = os.path.splitext(os.path.split(self.image_list[index])[-1])[0]
        mask_name = os.path.splitext(os.path.split(self.mask_list[index])[-1])[0]
        if image_name == mask_name:
            return image_name
        else:
            print(f"Name mismatch at index {index}: Img='{image_name}', Mask='{mask_name}'")
            return False

    def __getraw__(self, index):
        if not self.__getname__(index):
            raise ValueError(f"Image doesn't match mask: {os.path.split(self.image_list[index])[-1]}")
        image = Image.open(self.image_list[index]).convert('RGB')
        mask = Image.open(self.mask_list[index]).convert('L')
        return image, mask

    def __getitem__(self, index):
        image, mask = self.__getraw__(index)
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return image, mask.long()

class Compose:
    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class FixResize:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, image, target):
        image = F.resize(image, self.size, interpolation=T_vision.InterpolationMode.BILINEAR)
        target = F.resize(target, self.size, interpolation=T_vision.InterpolationMode.NEAREST)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

def get_transforms(input_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Returns a Compose object for transformations.
    Can be extended to have different transforms for train/val/test.
    """
    # For now, same transforms for all, but you can customize here.
    # e.g., add augmentations for training_transforms_list
    # training_transforms_list = [
    #     # Add augmentations like RandomHorizontalFlip etc. here
    #     # Ensure they accept (image, target) or adapt them
    # ] + common_transforms_list
    
    common_transforms_list = [
        FixResize(input_size),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ]
    return Compose(common_transforms_list)

def create_dummy_data_if_needed(base_path, config):
    """Creates dummy data if the specified dataset path doesn't exist."""
    # Check for a key subdirectory to infer if data exists
    if not (Path(base_path) / config["TRAIN_IMG_SUBDIR"]).exists():
        print(f"Warning: Data path {base_path} or its subdirectories not found. Creating dummy data.")
        dummy_folders_info = {
            "train_img": Path(base_path) / config["TRAIN_IMG_SUBDIR"],
            "train_ann": Path(base_path) / config["TRAIN_MASK_SUBDIR"],
            "val_img": Path(base_path) / config["VAL_IMG_SUBDIR"],
            "val_ann": Path(base_path) / config["VAL_MASK_SUBDIR"],
            "test_img": Path(base_path) / config["TEST_IMG_SUBDIR"],
            "test_ann": Path(base_path) / config["TEST_MASK_SUBDIR"],
        }
        for folder_path in dummy_folders_info.values():
            folder_path.mkdir(parents=True, exist_ok=True)

        num_classes = config["NUM_CLASSES"]
        for i in range(20): # Dummy files for train
            Image.new("RGB", (300, 300), color="gray").save(dummy_folders_info["train_img"] / f"dummy_train_{i}.jpg")
            Image.new("L", (300, 300), color=i % num_classes).save(dummy_folders_info["train_ann"] / f"dummy_train_{i}.png")
        for i in range(10): # Dummy files for val
            Image.new("RGB", (300, 300), color="lightgray").save(dummy_folders_info["val_img"] / f"dummy_val_{i}.jpg")
            Image.new("L", (300, 300), color=i % num_classes).save(dummy_folders_info["val_ann"] / f"dummy_val_{i}.png")
        for i in range(10): # Dummy files for test
            Image.new("RGB", (300, 300), color="darkgray").save(dummy_folders_info["test_img"] / f"dummy_test_{i}.jpg")
            Image.new("L", (300, 300), color=i % num_classes).save(dummy_folders_info["test_ann"] / f"dummy_test_{i}.png")
        print("Dummy data created.")
    else:
        print(f"Found data at {base_path}. Skipping dummy data creation.")