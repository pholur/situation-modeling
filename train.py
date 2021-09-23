from imports import *
from arch import *
from data import *
import wandb
from transformers import AdamW

wandb.init(project="Situation Modeling")
device = torch.device(preferred_cuda)

model = return_token_model('distilbert-base-uncased', len(labels), device)
model.train()


#PATH = "/mnt/SSD2/pholur/CTs/conspiracy_init.csv"
PATH = "/mnt/SSD2/pholur/CTs/conspiracy_0922.csv"
train_loader = get_data(PATH, FLAG, AUG, REEXTRACT)


## Maybe freeze a few layers y'know... ask Pavan if you want to do that.
modules = [model.distilbert.embeddings, *model.distilbert.transformer.layer[:3]] #Replace 3 by what you want
for module in modules:
    for param in module.parameters():
        param.requires_grad = False

optim = AdamW(model.parameters(), lr=LEARNING_RATE)

index = 0
for epoch in range(EPOCHS):
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
        print(i, loss)
        optim.step()

    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': loss,
                }, CHECKPOINT_PATH + ROOT_NAME + str(epoch) + "_" + str(index) + ".pt")


model.eval()