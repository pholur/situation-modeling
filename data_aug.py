import os
os.environ["MODEL_DIR"] = '/mnt/SSD2/pholur/CTs/model'
from imports import *

import nlpaug.augmenter.word as naw

aug = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased', action="insert", device=device_for_aug)

def data_aug(text, AUG):
    augmented_text = aug.augment(text, n=AUG)
    return augmented_text

#data_aug("I am a student.")
#returns ['i am a literature student.', 'i am a business student.', 'i specifically am a student.', 'like i am a student.', 'again i am a student.', 'i am only a student.', 'i am a lab student.', 'i think am a student.', 'i sure am a student.', 'oh i am a student.']