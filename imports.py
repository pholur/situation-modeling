# Critical to ensure reproducability

import torch
import os
torch.manual_seed(0)
# os.environ['PYTHONHASHSEED'] = '0'
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
#torch.set_deterministic(True)

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
EPOCHS = 4
AUG = 20 # has to be greater than 1
REEXTRACT = False
LEARNING_RATE = 1e-5#1e-4 seems too high #5e-5 seems too high

# SAVES
CHECKPOINT_PATH = '/mnt/SSD2/pholur/CTs/checkpoints/'
ROOT_NAME = 'Insider_Outsider_'

# CUDAs
preferred_cuda = "cuda:1"
preferred_cuda_test = "cuda:1"
device_for_aug = "cuda:2"

# Labels
labels = {"insider":0, "outsider":2, "idk":1}
reverse_labels = {0:"insider", 1:"idk", 2:"outsider"}