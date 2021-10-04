from transformers import DistilBertTokenizerFast, AutoTokenizer, BertTokenizer, BertTokenizerFast, PreTrainedTokenizerFast, RobertaTokenizerFast
from imports import *
tokenizer = RobertaTokenizerFast.from_pretrained(CORE_MODEL)
#tokenizer = DistilBertTokenizerFast.from_pretrained(CORE_MODEL)


def tokenization(train_texts, FLAG=False):

    def tokenize(text):
        return tokenizer(text, is_split_into_words=False, add_special_tokens=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
        #return tokenizer(text, is_split_into_words=False, return_offsets_mapping=True, padding=True)
    
    train_encodings = tokenize(train_texts)
    if FLAG:
        return train_encodings, tokenizer.convert_ids_to_tokens(train_encodings['input_ids'][0])
    return train_encodings