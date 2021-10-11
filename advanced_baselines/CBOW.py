import torch
from torch import nn

import torchtext

from collections import defaultdict
import time

class CBoWTextClassifier(nn.Module):
    
    def __init__(self, text_field, class_field, emb_dim, dropout=0.5):
        super().__init__()        
        voc_size = len(text_field.vocab)
        n_classes = len(class_field.vocab)  

        # Embedding layer: we specify the vocabulary size and embedding dimensionality.        
        self.embedding = nn.Embedding(voc_size, emb_dim)

        # A linear output layer.
        self.top_layer = nn.Linear(emb_dim, n_classes)

        # A dropout layer to regularize the model.
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, docs):
        # The words in the documents are encoded as integers. The shape of the documents
        # tensor is (max_len, n_docs), where n_docs is the number of documents in this batch,
        # and max_len is the maximal length of a document in the batch.
        # The shape is now (max_len, n_docs, emb_dim).
        embedded = self.embedding(docs)
        # The shape is now (n_docs, emb_dim)
        cbow = embedded.mean(dim=0)
        cbow_drop = self.dropout(cbow)
        scores = self.top_layer(cbow_drop)
        return scores


class CBoWTextClassifier2(torch.nn.Module):
    
    def __init__(self, text_field, class_field, emb_dim, n_hidden=10, dropout=0.5):
        super().__init__()        
        voc_size = len(text_field.vocab)
        n_classes = len(class_field.vocab)       
        self.embedding = nn.Embedding(voc_size, emb_dim)
        self.hidden_layer = nn.Linear(emb_dim, n_hidden)
        self.top_layer = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, texts):
        embedded = self.embedding(texts)
        cbow = embedded.mean(dim=0)
        cbow_drop = self.dropout(cbow)
        hidden = torch.relu(self.hidden_layer(cbow_drop))
        scores = self.top_layer(hidden)
        return scores 


