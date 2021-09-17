from transformers import DistilBertForTokenClassification

def return_token_model(model_name='distilbert-base-uncased', num_labels=3, device='cuda:0'):
    model = DistilBertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)
    return model