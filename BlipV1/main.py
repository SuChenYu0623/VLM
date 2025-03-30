import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


# custom
from transform.randaugment import RandomAugment
from flickr8k_dataset import flickr8k_dataset
from data import create_dataset, create_loader
from models.blip import blip_decoder
from utils import cosine_lr_schedule
from configs.config import config

# config = {
#     "image_root": 'flickr8k/images',
#     "caption_file": 'flickr8k/captions.csv',
#     "image_size": 224,

#     # 模型
#     "pretrained": False,
#     "vit": 'base',
#     "vit_grad_ckpt": False,
#     "vit_ckpt_layer": 0,
#     # "prompt": "a picture of ",
#     "prompt": "",

#     # 優化器
#     "init_lr": 1e-5,
#     "weight_decay": 0.05,

#     # 訓練
#     "max_epoch": 5,
#     "min_lr": 0

# }

### 載入資料 ###
train_dataset = create_dataset(dataset='flickr8k', config=config, min_scale=0.5)

dataset = train_dataset
sampler = RandomSampler(dataset)
batch_size = 10
num_workers = 4
train_dataloader = create_loader(dataset, sampler, batch_size, num_workers)
print(train_dataset.__len__())
print(len(train_dataloader))

### 建立模型 ###
print("Creating model")
model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                        vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                        prompt=config['prompt'])

print('載入完畢')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
model = model.to(device)

optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

print('開始訓練')
for epoch in range(0, config['max_epoch']):
    cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
    for i, (image, caption) in enumerate(train_dataloader):
        image = image.to(device)   
        loss = model(image, caption)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'[Epoch {epoch}] [{i}] {loss.item():.4f}')
            break

print('save model')
torch.save(model.state_dict(), 'saveModels/BlipV1.0.pth')