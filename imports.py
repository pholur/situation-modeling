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
FLAG = 0