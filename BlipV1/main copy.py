import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
from typing import List
import json

# custom
from transform.randaugment import RandomAugment
from flickr8k_dataset import flickr8k_dataset
from data import create_dataset, create_loader
from models.blip import blip_decoder
from utils import cosine_lr_schedule
from configs.config import config


class RecordService:
    def __init__(self):
        pass

    # TODO save model
    def saveModel(self):
        pass

    # TODO save training progress

# 存圖
def saveTrainProgress(train_losses: List[int], valid_losses: List[int],
                      name: str, config: dict):
    losses_dict = {"train_losses": train_losses, "valid_losses": valid_losses}
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs,
             train_losses,
             label='Training Loss',
             color='blue',
             marker='o')
    plt.plot(epochs,
             valid_losses,
             label='Validation Loss',
             color='red',
             marker='x')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{config["progress_image"]}/{name}_progress.png')
    plt.close()

    with open(f'{config["progress_losses"]}/{name}_progress', 'w') as file:
        json.dump(losses_dict, file)


### 載入資料 ###
train_dataset = create_dataset(dataset='train', config=config, min_scale=0.5)
train_sampler = RandomSampler(train_dataset)

valid_dataset = create_dataset(dataset='valid', config=config, min_scale=0.5)
valid_sampler = RandomSampler(valid_dataset)

batch_size = config["batch_size"]
num_workers = 32
train_dataloader = create_loader(train_dataset, train_sampler, batch_size,
                                 num_workers)
valid_dataloader = create_loader(valid_dataset, valid_sampler, batch_size,
                                 num_workers)
print(
    f"train_dataset: {train_dataset.__len__()}, train_dataloader: {len(train_dataloader)}"
)
print(
    f"valid_dataset: {valid_dataset.__len__()}, valid_dataloader: {len(valid_dataloader)}"
)

### 建立模型 ###
print("Creating model")
model = blip_decoder(pretrained=config['pretrained'],
                     image_size=config['image_size'],
                     vit=config['vit'],
                     vit_grad_ckpt=config['vit_grad_ckpt'],
                     vit_ckpt_layer=config['vit_ckpt_layer'],
                     prompt=config['prompt'])

print('載入完畢')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = torch.optim.AdamW(params=model.parameters(),
                              lr=config['init_lr'],
                              weight_decay=config['weight_decay'])

# 記錄 losses
train_losses = []
valid_losses = []

print('開始訓練')
for epoch in range(0, config['max_epoch']):
    cosine_lr_schedule(optimizer, epoch, config['max_epoch'],
                       config['init_lr'], config['min_lr'])

    model.train()
    total_train_loss = 0.0
    for i, (image, caption) in enumerate(train_dataloader):
        image = image.to(device)
        loss = model(image, caption)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)
    print(f'[Epoch {epoch}] [Train] Average Loss: {avg_train_loss:.4f}')

    # valid 部份
    model.eval()
    total_valid_loss = 0.0
    with torch.no_grad():
        for i, (image, caption) in enumerate(valid_dataloader):
            image = image.to(device)
            loss = model(image, caption)
            total_valid_loss += loss.item()

    avg_valid_loss = total_valid_loss / len(valid_dataloader)
    valid_losses.append(avg_valid_loss)
    print(f'[Epoch {epoch}] [Valid] Average Loss: {avg_valid_loss:.4f}')

# TODO name
# {framework}_{dataset}_{epoch}
framework = "Blipv1"
dataset = config["train_dataset_name"]
batch_size = config["batch_size"]
max_epoch = config['max_epoch']
name = f"{framework}_{dataset}_{max_epoch}_{batch_size}"

# TODO saveTrainProgress
saveTrainProgress(train_losses=train_losses,
                  valid_losses=valid_losses,
                  name=name,
                  config=config)

# TODO model state name
torch.save(model.state_dict(), f'{config["saveModels"]}/{name}.pth')
