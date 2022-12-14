from torchvision import transforms
from datasets.augumentations import RandomBoxes, RandomNoise, RandomBackground

MODEL_NEURONS = 16
BB_FACTOR = 150

# Data parameters
N_KEYPOINTS = 21
N_IMG_CHANNELS = 3
RAW_IMG_SIZE = 224
MODEL_IMG_SIZE = 128
RANDOM_CROP_SIZE = 180
DATA_DIR = "/data/wmucha/datasets/FreiHAND"
TRAIN_DATASET_MEANS = [0.4532, 0.4522, 0.4034]
TRAIN_DATASET_STDS =[0.2218, 0.2186, 0.2413]
DATASET_MEANS = [0.3950, 0.4323, 0.2954]
DATASET_STDS = [0.1966, 0.1734, 0.1836]

# Training parameters
EXPERIMENT_NAME = "checpoint_improved_aug_pytorch"
MAX_EPOCHS = 1000
BACTH_SIZE = 16
LEARNING_RATE = 0.1
DEVICE = 3
EARLY_STOPPING = 15
CONTINUE_FROM_CHECKPOINT = True
CHECKPOINT_DIR = "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/saved_models/fulldata_aug2_53"

# Testing parameters
TESTING_DEVICE = 0
TESTING_BATCH_SIZE = BACTH_SIZE
COLORMAP = {
    "thumb": {"ids": [0, 1, 2, 3, 4], "color": "g"},
    "index": {"ids": [0, 5, 6, 7, 8], "color": "c"},
    "middle": {"ids": [0, 9, 10, 11, 12], "color": "b"},
    "ring": {"ids": [0, 13, 14, 15, 16], "color": "m"},
    "little": {"ids": [0, 17, 18, 19, 20], "color": "r"},
}
# Image augumentations
TRAIN_IMG_TRANSFORM = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomRotation(degrees=(0, 30)),
                    transforms.RandomCrop(RANDOM_CROP_SIZE),
                    transforms.Resize(MODEL_IMG_SIZE),
                    # transforms.RandomHorizontalFlip(p=0.5),
                    # transforms.RandomVerticalFlip(p=0.5),
                    transforms.Normalize(mean=TRAIN_DATASET_MEANS, std=TRAIN_DATASET_STDS),
                ]
            )

TRAIN_HEATMAP_TRANSFORM = transforms.Compose(
                    [
                        transforms.RandomRotation(degrees=(0, 30)),
                        transforms.RandomCrop(RANDOM_CROP_SIZE),
                        transforms.Resize(MODEL_IMG_SIZE),
                        # transforms.RandomHorizontalFlip(p=0.5),
                        # transforms.RandomVerticalFlip(p=0.5),
                    ]
                )

VAL_IMG_TRANSFORM = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize(MODEL_IMG_SIZE),
                        transforms.Normalize(mean=TRAIN_DATASET_MEANS, std=TRAIN_DATASET_STDS),
                    ]
                )

VAL_HEATMAP_TRANSFORM = transforms.Compose(
                    [
                        transforms.Resize(MODEL_IMG_SIZE),
                    ]
                )