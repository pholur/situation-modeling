from data_aug import data_aug
import torch
from torch.utils.data import Dataset, DataLoader
from imports import *
from shared_train_and_test_functions import tokenization
import pandas as pd
import sys
from collections import defaultdict
import re
# maybe we can make it a regression task?
MAPPING = {"$AnswerA":0, "$AnswerB":2, "$AnswerC":1}
REVERSE_MAPPING = {0:"$AnswerA", 1:"$AnswerC", 2:"$AnswerB"}



class ConspiracyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



def generate_full_labels(ordered_posts, dataset, train_encodings, flag = 0):
    # if flag is 1, each entity gets its own sample
    # if flag is 0, all entities are combined into one sample per post

    encoded_labels = []
    encoded_labels_other = []
    new_ordered_posts = []

    for i,(post, doc_offset) in enumerate(zip(ordered_posts, train_encodings.offset_mapping)):

        arr_offset = np.array(doc_offset)
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100

        list_of_labels = dataset[post]
        for (indices, label) in list_of_labels:

            if flag == 1:
                doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
            
            for index in indices:
                start_point = index[0]
                end_point = index[1]

                doc_enc_labels[(arr_offset[:,0] <= start_point) & (arr_offset[:,1] >= start_point)] = label
                doc_enc_labels[(arr_offset[:,0] <= end_point) & (arr_offset[:,1] >= end_point)] = label
                doc_enc_labels[(arr_offset[:,0] >= start_point) & (arr_offset[:,1] <= end_point)] = label        
            
            if flag == 1:
                new_ordered_posts.append(post)
                encoded_labels_other.append(doc_enc_labels.tolist())

        # print(doc_enc_labels)
        # print()
        # print(post)
        # print()
        # print(indices)
        # print()
        # print(arr_offset)
        # print()
        # print(list_of_labels)
        # sys.exit()
        if flag == 0:
            encoded_labels.append(doc_enc_labels.tolist())

    if flag == 0:
        return ordered_posts, encoded_labels
    elif flag == 1:
        return new_ordered_posts, encoded_labels_other



def get_data_processed(dataset, FLAG = 1):
    ordered_posts = list(dataset.keys())
    train_encodings = tokenization(ordered_posts)
    new_ordered_posts, labels = generate_full_labels(ordered_posts, dataset, train_encodings, FLAG)

    if FLAG == 1:
        print("Using separate samples per entity: Size = ", len(new_ordered_posts))
    else:
        print("Using one sample per entity: Size = ", len(new_ordered_posts))

    train_encodings = tokenization(new_ordered_posts)
    train_dataset = ConspiracyDataset(train_encodings, labels)
    return train_dataset


def get_data_loaded(data_class, batch_size=16, num_workers = 0):
    return DataLoader(
                        data_class,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        worker_init_fn=seed_worker,
                        generator=g,
                        pin_memory=True
                    )


def clean_text(text):
    text = text.replace("*", "") # many more of these to come but may remove matches so be careful!
    text = text.replace("(", "") # many more of these to come but may remove matches so be careful!
    text = text.replace(")", "") # many more of these to come but may remove matches so be careful!
    return text


def get_dataset_from_file(file_name, AUG):
    df = pd.read_csv(file_name)
    all_data = defaultdict(list)
    unique_posts_ids = df['PostID'].unique()
    count = 0
    for id_ in unique_posts_ids:
        temp_df = df[df['PostID'] == id_]
        main_post = str(temp_df['Post'].iloc[0]).lower()
        main_post = clean_text(main_post)
        print(count)
        count += 1
        # print(data_aug(main_post, AUG))
        total_augmented_posts = [main_post, *data_aug(main_post, AUG)]
        for post_re in total_augmented_posts:
            for (noun_phrase, sentiment) in zip(temp_df["NP"], temp_df["sentiment"]):
                try:
                    real_label = MAPPING[sentiment]
                    noun_phrase = str(noun_phrase).lower()
                    noun_phrase = clean_text(noun_phrase)
                    

                    if len(noun_phrase.split(" ")) < 2:
                        try:
                            # single word case, this still escapes some tokens that might need resolution.
                            # the last index is exclusive
                            indices = [(m.start(0)+1, m.end(0)) for m in re.finditer(" " + noun_phrase, post_re)]
                            indices.extend([(m.start(0), m.end(0)-1) for m in re.finditer(noun_phrase + " ", post_re)])
                        except:
                            indices = [(m.start(0), m.end(0)) for m in re.finditer(noun_phrase, post_re)]
                    else:
                        indices = [(m.start(0), m.end(0)) for m in re.finditer(noun_phrase, post_re)]
                except:
                    continue

                # all data will contain the posts as keys.
                # for each key, we will have a list of tuples.
                # each tuple will contain  a list of the start and end indices of the noun phrase in the post as the first element.
                # noun phrase is the second element.
                all_data[post_re].append((indices, real_label))

    return all_data


import pickle
def get_data(path, FLAG, AUG, re_extract = False, device_for_aug = "cuda:2"):
    
    if re_extract:
        dataset = get_dataset_from_file(path, AUG, device_for_aug)
        with open("/mnt/SSD2/pholur/CTs/Pickles/aug_dataset.pkl", 'wb') as f:
            pickle.dump(dataset, f)
    else:
        dataset = pickle.load(open("/mnt/SSD2/pholur/CTs/Pickles/aug_dataset.pkl", "rb"))

    train_dataset = get_data_processed(dataset, FLAG)
    train_loader = get_data_loaded(train_dataset)
    return train_loader