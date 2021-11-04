from imports import *
from arch import *
from data import *
import wandb
from transformers import AdamW

wandb.init(project="Situation Modeling")
device = torch.device(preferred_cuda)

model = return_token_model(CORE_MODEL, len(labels), device)
model.train()


PATH = INPUT_DATA_PATH
train_loader, val_loader = get_data(PATH, FLAG, AUG, REEXTRACT, FRACTION, OPT, SAVE)

print(model)
## Maybe freeze a few layers y'know... ask Pavan if you want to do that.
modules = [model.distilbert.embeddings, *model.distilbert.transformer.layer[:FROZEN_LAYERS]] #Replace 3 by what you want
#modules = [model.roberta.embeddings, *model.roberta.encoder.layer[:FROZEN_LAYERS]] #Replace 3 by what you want

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

        # #labels = labels[torch.randperm(len(labels))]
        # print(input_ids)
        # print(labels)
        # exit()
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        wandb.log({'loss': loss, 'epoch': epoch, 'step': i, 'index': index})
        index += 1
        loss.backward()
        print(i, loss)
        optim.step()

        if i % 10 == 0: # when should I compute validation loss?
            model.eval()
            val_loss = 0
            number_of_validation_batches = 0
            for j,val_batch in enumerate(val_loader):
                input_ids = val_batch['input_ids'].to(device)
                attention_mask = val_batch['attention_mask'].to(device)
                labels = val_batch['labels'].to(device)
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs[0]
                number_of_validation_batches += 1
            wandb.log({'val_loss': val_loss/number_of_validation_batches, 'epoch': epoch, 'step': i, 'index': index})
            model.train()

    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': loss,
                }, CHECKPOINT_PATH + ROOT_NAME + str(epoch) + "_" + str(index) + ".pt")

model.eval()