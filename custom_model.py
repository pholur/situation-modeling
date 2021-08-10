from imports import *

def return_model(choice="pretrained"):

    class DistillBERTClass(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.distil_bert = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased", output_attentions=True)
        
        def forward(self, ids, mask):
            distilbert_output = self.distil_bert(ids, mask)
            hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
            return hidden_state
            
    model = DistillBERTClass()
    model.to(device)
    return model





