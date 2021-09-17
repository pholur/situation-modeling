from imports import *
from arch import *
from data import *
import wandb
from transformers import AdamW

wandb.init(project="Situation Modeling")
labels = {"insider":0, "outsider":2, "idk":1}
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = return_token_model('distilbert-base-uncased', len(labels), device)
model.train()


PATH = "/mnt/SSD2/pholur/CTs/conspiracy.csv"
train_loader = get_data(PATH, FLAG)



## Maybe freeze a few layers y'know... ask Pavan if you want to do that.

optim = AdamW(model.parameters(), lr=5e-5)

index = 0
for epoch in range(20):
    for i,batch in enumerate(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        #labels = labels[torch.randperm(len(labels))]
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        wandb.log({'loss': loss, 'epoch': epoch, 'step': i, 'index': index})
        index += 1
        loss.backward()
        optim.step()

model.eval()