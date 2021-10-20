import os
os.environ["MODEL_DIR"] = '/mnt/SSD2/pholur/CTs/model'
from imports import *

import nlpaug.augmenter.word as naw

aug = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased', action="insert", device=device_for_aug, aug_min=2, aug_max=5)

def data_aug(text, AUG=10):
    augmented_text = aug.augment(text, n=AUG)
    return augmented_text

# print(data_aug("I am a student."))
