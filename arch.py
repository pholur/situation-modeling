from transformers import DistilBertForTokenClassification, RobertaForTokenClassification, BertForTokenClassification

def return_token_model(model_name='distilbert-base-uncased', num_labels=3, device='cuda:0'):
    if model_name == 'distilbert-base-uncased':
        model = DistilBertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    elif model_name == 'roberta-base':
        model = RobertaForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    elif model_name == 'bert-base-uncased':
        model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)
    return model