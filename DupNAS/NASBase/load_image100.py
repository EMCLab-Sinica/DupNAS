# import kagglehub

# # Download the dataset
# path = kagglehub.dataset_download("ambityga/imagenet100")

# print("Path to dataset files:", path)


import sys, os
from os.path import dirname, realpath
#import kagglehub
import shutil
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.utils.data

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from settings import Settings

# Define dataset paths
DATASET_DIR = Settings.NAS_SETTINGS_PER_DATASET['IMAGE100']['TRAIN_DATADIR']  # Change if needed
#KAGGLEHUB_DIR = os.path.expanduser("~/.cache/kagglehub/datasets/ambityga/imagenet100/versions/8")


def load_image100_dataset(DATASET_DIR=DATASET_DIR):

    KAGGLEHUB_DIR = os.path.expanduser("/4TB/cyliu901/.cache/kagglehub/datasets/ambityga/imagenet100/versions/8")
    
    #Step 1: Download ImageNet-100 dataset
    if not os.path.exists(KAGGLEHUB_DIR):
        print("Downloading ImageNet-100 dataset from KaggleHub...")
        KAGGLEHUB_DIR = kagglehub.dataset_download("ambityga/imagenet100")
        #KAGGLEHUB_DIR = dataset_path
        print(f"Downloaded dataset to: {KAGGLEHUB_DIR}")

    else:
        print("Already saved ImageNet-100 dataset from KaggleHub.")
    
    # Step 2: Create target dataset directory
    os.makedirs(DATASET_DIR, exist_ok=True)
    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)


    marker = os.path.join(DATASET_DIR, ".prepared")
    if os.path.exists(marker):
        print("Dataset already prepared:", DATASET_DIR)
        return train_dir, val_dir

    def _install(src, dst):
        if not os.path.exists(dst):
            try:
                os.link(src, dst)         # hardlink (fast, no extra space)
            except OSError:
                shutil.copy2(src, dst)     # fallback

    # Step 3: Merge split train folders (train.X1 - train.X4) into a single train folder
    for i in range(1, 5):  # train.X1 to train.X4
        split_train_dir = os.path.join(KAGGLEHUB_DIR, f"train.X{i}")
        if os.path.exists(split_train_dir):
            print(f"Merging {split_train_dir} into {train_dir}...")
            for class_name in os.listdir(split_train_dir):
                class_path = os.path.join(split_train_dir, class_name)
                target_class_path = os.path.join(train_dir, class_name)
                os.makedirs(target_class_path, exist_ok=True)
                for img_file in os.listdir(class_path):
                    shutil.copy2(os.path.join(class_path, img_file), target_class_path)

    # Step 4: Move validation data (val.X) into the val folder
    val_src_dir = os.path.join(KAGGLEHUB_DIR, "val.X")
    if os.path.exists(val_src_dir):
        print(f"Moving {val_src_dir} into {val_dir}...")
        for class_name in os.listdir(val_src_dir):
            class_path = os.path.join(val_src_dir, class_name)
            target_class_path = os.path.join(val_dir, class_name)
            os.makedirs(target_class_path, exist_ok=True)
            for img_file in os.listdir(class_path):
                shutil.copy2(os.path.join(class_path, img_file), target_class_path)
    else:
        print("WARNING: val.X not found; validation will be empty.")

    open(marker, "w").close()
    print("Dataset setup complete!")
    return train_dir, val_dir


# Step 5: Define transformations for training and validation
# train_transform = transforms.Compose([
#     transforms.Resize(256),         # Resize to 256x256
#     transforms.RandomCrop(224),     # Random crop to 224x224
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# valid_transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Step 6: Load datasets using ImageFolder
# trainset = ImageFolder(root=train_dir, transform=train_transform)
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, pin_memory=True, num_workers=4)

# valset = ImageFolder(root=val_dir, transform=valid_transform)
# val_loader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)

# print(f"Train dataset size: {len(trainset)}")
# print(f"Validation dataset size: {len(valset)}")

# # Step 7: Test dataset loading
# for images, labels in train_loader:
#     print(f"Batch size: {images.shape}")  # Should be [batch_size, 3, 224, 224]
#     print(f"Labels: {labels[:10]}")
#     break
