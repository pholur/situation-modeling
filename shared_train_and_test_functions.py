from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenization(train_texts, FLAG=False):

    def tokenize(text):
        return tokenizer(text, is_split_into_words=False, return_offsets_mapping=True, padding=True, truncation=True)
    
    train_encodings = tokenize(train_texts)
    if FLAG:
        return train_encodings, tokenizer.convert_ids_to_tokens(train_encodings['input_ids'][0])
    return train_encodings