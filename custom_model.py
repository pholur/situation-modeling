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

def return_decoder(number_of_tags):
    class Decoder(nn.Module):
        def __init__(self):
            self.decoder = nn.Linear(768, 768)
            self.decoder_activation = nn.ReLU()
            self.decoder_dropout = nn.Dropout(p=0.1)
            self.decoder_output = nn.Linear(768, number_of_tags)

        def forward(self, input):
            input = self.decoder_dropout(input)
            input = self.decoder(input)
            input = self.decoder_activation(input)
            input = self.decoder_dropout(input)
            input = self.decoder_output(input)
            return input

    model = Decoder().to(device)
    return model




