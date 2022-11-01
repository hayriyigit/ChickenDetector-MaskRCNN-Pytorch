import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 2
CHECKPOINT_FILE = "mask_rcnn.pth.tar"
PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = True
TRAIN_DIR = 'dataset/train'
VALID_DIR = 'dataset/valid'
TEST_DIR = 'dataset/test'
IMAGE_SIZE = [640,640]