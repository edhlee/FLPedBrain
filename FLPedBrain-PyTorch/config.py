"""Configuration for CDS (Centralized Data Sharing) baseline training."""

import os

# Data paths
# Use relative path so it works on any machine after copying the project
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Site IDs for training (16 sites from FL)
# Original order (for reference)
# TRAIN_SITE_IDS_ORIGINAL = ['TM', 'PH', 'TO', 'UT', 'DU', 'CP', 'IN', 'ST', 'SE', 'CG', 'NY', 'CH', 'GO', 'BO', 'KC', 'DY']
TRAIN_SITE_IDS = ['TM', 'PH', 'TO', 'UT', 'DU', 'CP', 'IN', 'ST', 'SE', 'CG', 'NY', 'CH', 'GO', 'BO', 'KC', 'DY']

# GPU allocation for FL training (balanced by sample count)
# GPU 0: ST(328), GPU 1: SE(241), GPU 2: CG(150)
# GPU 3: UT(129)+TM(13)=142, GPU 4: IN(118)+DU(24)=142
# GPU 5: CP(96)+NY(26)+CH(14)=136, GPU 6: TO(92)+DY(28)+BO(19)=139
# GPU 7: GO(78)+PH(55)+KC(3)=136
FL_GPU_SITES = [
    ['ST'],                    # GPU 0: 328
    ['SE'],                    # GPU 1: 241
    ['CG'],                    # GPU 2: 150
    ['UT', 'TM'],              # GPU 3: 142
    ['IN', 'DU'],              # GPU 4: 142
    ['CP', 'NY', 'CH'],        # GPU 5: 136
    ['TO', 'DY', 'BO'],        # GPU 6: 139
    ['GO', 'PH', 'KC'],        # GPU 7: 136
]
VAL_SITE_IDS = ['TK', 'AU']

# Data files
NORMALS_TRAIN = os.path.join(DATA_DIR, "normals_400x256x256x64_train_for_fl_64.npy")
NORMALS_VAL = os.path.join(DATA_DIR, "normals_867x256x256x64_val_for_fl_64.npy")
COMBINED_TRAIN = os.path.join(DATA_DIR, "combined_data_uint8_train.npy")
COMBINED_VAL = os.path.join(DATA_DIR, "combined_data_uint8_val.npy")

# Model parameters
NUM_FRAMES = 64
FRAME_HEIGHT = 256
FRAME_WIDTH = 256
NUM_CLASSES = 5  # 0=Normal, 1=Ependymoma, 2=DIPG, 3=Medulloblastoma, 4=Pilocytic

# Class names (verified from data identifiers in .npy files)
CLASS_NAMES = ["Normal Controls", "Ependymoma", "DIPG", "Medulloblastoma", "Pilocytic"]

# Pretrained model settings
# Set USE_PRETRAINED=True to use I3D with ImageNet+Kinetics weights (like TF model)
# Set USE_PRETRAINED=False for random initialization baseline
USE_PRETRAINED = True
PRETRAINED_WEIGHTS = os.path.join(
    os.path.dirname(__file__), "pretrained_weights", "rgb_imagenet.pt"
)
FREEZE_ENCODER = False  # Set True to freeze pretrained encoder (faster training)

# Training parameters
BATCH_SIZE = 2  # 3D data is memory-intensive
BATCH_SIZE_EVAL = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
LR_DECAY_RATE = 0.5
LR_DECAY_STEPS = 100  # epochs

# Loss weights
DICE_LOSS_WEIGHT = 0.5  # Weight for segmentation loss vs classification loss

# Data augmentation
CROP_SIZE = 240  # Random crop from 256x256
AUGMENT_TRAIN = True

# Normals sampling (to match TF approach)
# TF uses train_x.shape[0]//4 normals per site, so ~1/4 of tumor samples
# Set to None to use all normals, or a fraction (e.g., 0.25) to match TF
NORMALS_FRACTION = 0.25  # Use 1/4 of normals relative to tumor samples (matches TF)

# Checkpointing
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
SAVE_EVERY = 5  # Save checkpoint every N epochs

# Device
DEVICE = "cuda"  # Will use Blackwell Pro 6000
