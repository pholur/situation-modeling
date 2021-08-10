### Torch

import torch
torch.manual_seed(0)

### Tokenizers etc..

from torch import cuda
device = 'cuda:0' if cuda.is_available() else 'cpu'

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

### Transformers

import transformers
from transformers import DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased",
    do_lower_case=True
)


def decide_set(text):
    inputs = tokenizer.encode_plus(
                    text,
                    None,
                    add_special_tokens=True,
                    max_length=max_len,
                    pad_to_max_length=True,
                    return_token_type_ids=True
                )
    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    ids_ = torch.tensor(ids, dtype=torch.long).to(device)
    mask_ = torch.tensor(mask, dtype=torch.long).to(device)
    return ids_.unsqueeze(0), mask_.unsqueeze(0), tokenizer.tokenize(text)

### Numpy

import numpy as np
np.random.seed(0)

import random
random.seed(0)

### Constants

max_len = 512
EPOCHS = 1
LEARNING_RATE = 5e-05
BATCH_SIZE = 8
SLOPE = 0.1
EXPONENTIAL_RATE = 350

### PATHS
DATASET_INPUT_PATH = "/mnt/SSD2/pholur/Concepts/numbers_float"
CHECKPOINT_PATH = "/mnt/SSD2/pholur/Concepts/checkpoints/"
#ROOT_NAME = "DistilBERT_numbers_350_"
ROOT_NAME = "DistilBERT_numbers_EXP_FLOAT_RANDOM_DOT_L1_FREEZE_3_"