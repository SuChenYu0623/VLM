from PIL import Image
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from data import create_dataset, create_loader

# custom
from models.blip import blip_decoder
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from configs.config import config

# 初始化CIDEr和SPICE评分器
cider_scorer = Cider()
spice_scorer = Spice()




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# pretrained_path = 'saveModels/BlipV1.0.pth'
pretrained_path = '/home/chris/Desktop/VLM/BlipV1/record/coco_train_v1/model.pth'
model = blip_decoder(pretrained=pretrained_path, image_size=224, vit='base')
model.eval()
model = model.to(device)

train_dataset = create_dataset(dataset='valid', config=config, min_scale=0.5)
print(train_dataset.__len__())

with torch.no_grad():
    
    for i in range(train_dataset.__len__()):
        image, text = train_dataset[i]
        image = image.unsqueeze(0).to(device)
        caption = model.generate(image, sample=False, num_beams=1, max_length=40, min_length=5)
        candidate = {1: [caption[0]]} # 生成的結果
        reference = {1: []} # 答案
        for j in range(5):
            image, text = train_dataset[i+j]
            reference[1].append(text)
        print(reference)
        print('caption', caption)
        cider_score, _ = cider_scorer.compute_score(reference, candidate)
        spice_score, _ = spice_scorer.compute_score(reference, candidate)
        print(f"CIDEr Score: {cider_score}")
        print(f"SPICE Score: {spice_score}")

        i += 5
        break

