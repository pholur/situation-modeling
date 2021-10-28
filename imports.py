# Critical to ensure reproducability
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import os
torch.manual_seed(0)
os.environ['TRANSFORMERS_CACHE'] = '/mnt/SSD2/pholur/cache/'
CORE_MODEL = 'roberta-large'
#CORE_MODEL = 'roberta-base'
#CORE_MODEL = 'distilbert-base-uncased'


# os.environ['PYTHONHASHSEED'] = '0'
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
# torch.set_deterministic(True)

g = torch.Generator()
g.manual_seed(0)

import random
random.seed(0)

import numpy as np
np.random.seed(0)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(0)
    random.seed(0)

# OPTIONS
FLAG = 1 # 1 for single entity in single sample, 0 for all entities in single sample
EPOCHS = 10#20#4
BATCH_SIZE = 64#128#16
AUG = [20,0,0] # has to be greater than 1
FRACTION = [0.8, 0.1, 0.1]
REEXTRACT = False
SAVE = False # works with reextract flag
LEARNING_RATE = 1e-6 #[[1e-6]] #1e-5 #1e-4 seems too high #5e-5 seems too high
FROZEN_LAYERS = -1#-6 #-3 # -2 # -1d

# SAVES
CHECKPOINT_PATH = '/mnt/SSD2/pholur/CTs/checkpoints/Day_1017_'
ROOT_NAME = 'Insider_Outsider_'
#INPUT_DATA_PATH = "/mnt/SSD2/pholur/CTs/conspiracy_0922.csv"
INPUT_DATA_PATH = "/mnt/SSD2/pholur/CTs/conspiracy_1017.csv"
RAW_TRAIN_DATA_PATH = "/mnt/SSD2/pholur/CTs/conspiracy_1017_raw_train.csv"
RAW_TEST_DATA_PATH = "/mnt/SSD2/pholur/CTs/conspiracy_1017_raw_test.csv"
RAW_VAL_DATA_PATH = "/mnt/SSD2/pholur/CTs/conspiracy_1017_raw_val.csv"

# INPUT_DATA_PATH = "/mnt/SSD2/pholur/CTs/conspiracy_1008.csv"
# RAW_TRAIN_DATA_PATH = "/mnt/SSD2/pholur/CTs/conspiracy_1008_raw_train.csv"
# RAW_TEST_DATA_PATH = "/mnt/SSD2/pholur/CTs/conspiracy_1008_raw_test.csv"
# RAW_VAL_DATA_PATH = "/mnt/SSD2/pholur/CTs/conspiracy_1008_raw_val.csv"
DEPOSIT_PATH = "./ShortTerm_Results/"

# CUDAs
preferred_cuda = "cuda:5"
preferred_cuda_test = "cuda:5"
device_for_aug = "cuda:4"

# Labels
labels = {"insider":0, "outsider":2, "idk":1}
reverse_labels = {0:"insider", 1:"idk", 2:"outsider"}

MAPPING = {"$AnswerA":0, "$AnswerB":2, "$AnswerC":1}
REVERSE_MAPPING = {0:"$AnswerA", 1:"$AnswerC", 2:"$AnswerB"}

# Mode
OPT = "test"
PAIR_TEST_WITH_LEAVE_OUT = True

MODEL_IN_TESTING = "/mnt/SSD2/pholur/CTs/checkpoints/Day_1017_Insider_Outsider_8_11961.pt"
#MODEL_IN_TESTING = "/mnt/SSD2/pholur/CTs/checkpoints/Day_1017_Insider_Outsider_0_1329.pt"
