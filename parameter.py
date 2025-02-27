REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 128
INPUT_DIM = 4
EMBEDDING_DIM = 128
NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value
K_SIZE = 20  # the number of neighboring nodes

USE_GPU = False  # do you want to collect training data using GPUs
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 1
NUM_META_AGENT = 32
LR = 1e-5
GAMMA = 1
DECAY_STEP = 256  # not use
SUMMARY_WINDOW = 32
FOLDER_NAME = 'ariadne'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
LOAD_MODEL = False  # do you want to load the model trained before
SAVE_IMG_GAP = 100
USE_WANDB = True
if USE_WANDB:
    WANDB_ID = ''
    WANDB_NOTES = ''
