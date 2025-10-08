import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
from typing import List
import json
import os
from pathlib import Path
import shutil
from transformers import BlipForConditionalGeneration, BlipProcessor

# custom
from transform.randaugment import RandomAugment
from data import create_dataset, create_loader
from models.blip import blip_decoder
from utils import cosine_lr_schedule
from configs.config import config


class RecordService:

    def __init__(self, config, name):
        self.name = name
        self.record_path = f'{config["recordPath"]}/{self.name}'
        self.model_path = f"{self.record_path}/model.pth"
        self.loss_image_path = f"{self.record_path}/loss_progress.png"
        self.loss_dict_path = f"{self.record_path}/loss_progress.json"

        self.config_src = Path(
            "/home/chris/Desktop/VLM/BlipV1/configs/config.py")
        self.med_src = Path(
            "/home/chris/Desktop/VLM/BlipV1/configs/med_config.json")
        self.config_dest = Path(f"{self.record_path}/config.py")
        self.med_dest = Path(f"{self.record_path}/med_config.json")
        self.prepare()

    # TODO prepare
    def prepare(self):
        if not os.path.exists(self.record_path):
            os.makedirs(self.record_path, exist_ok=True)

    # TODO save config
    def saveConfig(self):

        def copy_file(src_path: str, dst_dir: str):
            src = Path(src_path)
            dst_dir = Path(dst_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / src.name)

        copy_file(src_path=self.config_src, dst_dir=self.config_dest)
        copy_file(src_path=self.med_src, dst_dir=self.med_dest)

    # TODO save model
    def saveModel(self, model):
        torch.save(model.state_dict(), f'{self.model_path}')

    # TODO save training progress
    def saveTrainProgress(self, train_losses: List[int],
                          valid_losses: List[int]):
        losses_dict = {
            "train_losses": train_losses,
            "valid_losses": valid_losses
        }
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
        plt.savefig(f'{self.loss_image_path}')
        plt.close()

        with open(f'{self.loss_dict_path}', 'w') as file:
            json.dump(losses_dict, file)


### 載入資料 ###
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base")
train_dataset = create_dataset(dataset='pretrain_train',
                               config=config,
                               min_scale=0.5,
                               processor=processor)
train_sampler = RandomSampler(train_dataset)

valid_dataset = create_dataset(dataset='pretrain_valid',
                               config=config,
                               min_scale=0.5,
                               processor=processor)
valid_sampler = RandomSampler(valid_dataset)

batch_size = config["batch_size"]
num_workers = 16
# train_dataloader = create_loader(train_dataset, train_sampler, batch_size,
#                                  num_workers)
# valid_dataloader = create_loader(valid_dataset, valid_sampler, batch_size,
#                                  num_workers)

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=True)

print(
    f"train_dataset: {train_dataset.__len__()}, train_dataloader: {len(train_dataloader)}"
)
print(
    f"valid_dataset: {valid_dataset.__len__()}, valid_dataloader: {len(valid_dataloader)}"
)

### 建立模型 ###
print("Creating model")
# model = blip_decoder(pretrained=config['pretrained'],
#                      image_size=config['image_size'],
#                      vit=config['vit'],
#                      vit_grad_ckpt=config['vit_grad_ckpt'],
#                      vit_ckpt_layer=config['vit_ckpt_layer'],
#                      prompt=config['prompt'])

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base")

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
    for i, inputs in enumerate(train_dataloader):
        pixel_values = inputs['pixel_values'].squeeze().to(device)
        input_ids = inputs['input_ids'].squeeze().to(device)
        decoder_input_ids = model.prepare_decoder_input_ids_from_labels(
            labels=input_ids)

        print('input_ids', input_ids.shape)
        print('pixel_values', pixel_values.shape)
        # input_ids = input_ids.to(device)
        # pixel_values = pixel_values.to(device)
        # caption = caption.to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values,
                        decoder_input_ids=decoder_input_ids,
                        labels=input_ids)
        loss = outputs.loss
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
            inputs = processor(text=caption,
                               images=image,
                               return_tensors="pt",
                               padding=True).to(device)
            input_ids = inputs['input_ids']
            pixel_values = inputs['pixel_values']

            loss = model(input_ids=input_ids,
                         pixel_values=pixel_values,
                         labels=input_ids)
            total_valid_loss += loss.item()

    avg_valid_loss = total_valid_loss / len(valid_dataloader)
    valid_losses.append(avg_valid_loss)
    print(f'[Epoch {epoch}] [Valid] Average Loss: {avg_valid_loss:.4f}')
#
name = "flickr30k_train_v1"
recordService = RecordService(config=config, name=name)
recordService.saveTrainProgress(
    train_losses=train_losses,
    valid_losses=valid_losses,
)
recordService.saveModel(model=model)
recordService.saveConfig()
