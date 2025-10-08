import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from transform.randaugment import RandomAugment
# custom
from models.blip import blip_decoder
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from transformers import BlipProcessor, BlipForConditionalGeneration


def load_captions(caption_file):
    """
    讀取 caption.txt，返回圖片名稱到標註的映射
    """
    df = pd.read_csv(caption_file)
    samples = list(df.itertuples(index=False, name=None))  
    return samples

class evaluate_dataset(Dataset):
    def __init__(self, transform, image_root, caption_file, max_words=30, prompt=''):
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        '''

        self.image_root = image_root
        # current_dir = os.path.dirname(__file__)
        # caption_file = os.path.join(current_dir, caption_file)
        print('os', os.getcwd())
        # print('caption_file', caption_file, image_root)
        self.samples = load_captions(caption_file)  # 讀取所有 (image, caption) 對
        self.transform = transform
        self.prompt = prompt
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        image_name, caption = self.samples[index]
        current_dir = os.path.dirname(__file__)
        # image_path = os.path.join(current_dir, self.image_root, image_name)        
        # image = Image.open(image_path).convert('RGB')   
        # image = self.transform(image)

        caption = self.prompt + caption
        # return image, caption
        return image_name, caption

def load_image(image, image_size, device):
    raw_image = Image.open(str(image)).convert('RGB')

    w, h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

class Evaluate_dataset(Dataset):
    def __init__(self, transform, image_root, caption_file, max_words=30, prompt=''):
        self.image_root = image_root
        self.caption_file = caption_file
        self.transform = transform
        self.samples = self.initDataset()
        self.samples_keys = list(self.samples.keys())

    def initDataset(self):
        reference = {}
        df = pd.read_csv(self.caption_file)
        for row in df.values:
            image_name = row[0]
            caption = row[1]
            if not image_name in reference:
                reference[image_name] = [caption]
            else:
                reference[image_name].append(caption)
        return reference

    def __len__(self):
        return len(list(self.samples.keys()))

    def __getitem__(self, index):
        image_name = self.samples_keys[index]
        captions = self.samples[image_name]
        current_dir = os.path.dirname(__file__)
        image_path = os.path.join(current_dir, self.image_root, image_name)        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        return image, captions, image_name


        

if __name__ == "__main__":
    config = {
        "image_root": '/home/chris/Desktop/VLM/datasets/flickr8k/images',
        "caption_file": '/home/chris/Desktop/VLM/datasets/flickr8k/captions.csv',
        "image_size": 224,
        "prompt": ''
    }

    image_root = config['image_root']
    caption_file = config['caption_file']
    image_size = config['image_size']
    prompt = config['prompt']
    transform = None
    dataset = evaluate_dataset(transform, image_root, caption_file, max_words=30, prompt='')
    print(dataset[0])
    # train_dataset = create_dataset(dataset='flickr8k', config=config, min_scale=0.5)

    reference = {1: ["a cat sitting on a bench", "a cat on the bench"]}
    candidate = {1: ["a cat is sitting on the bench"]}

    reference = {}
    candidate = {}
    df = pd.read_csv(caption_file)
    for row in df.values:
        image_name = row[0]
        caption = row[1]
        if not image_name in reference:
            reference[image_name] = [caption]
        else:
            reference[image_name].append(caption)
        
        
    print(len(reference.keys()))
    print(list(reference.keys())[0])
    
    # main
    min_scale = 0.5
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform_train = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])
    eval_dataset = Evaluate_dataset(transform_train, image_root, caption_file, max_words=30, prompt='')

    print(eval_dataset)
    print('eval:', eval_dataset.__len__())


    # 初始化CIDEr和SPICE评分器
    cider_scorer = Cider()
    spice_scorer = Spice()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #### pretrain model ####
    # processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    # model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
    # candidate = {}
    # reference = {}
    # cnt = 0
    # for i in range(eval_dataset.__len__()):
    #     image, captions, image_name = eval_dataset[i]
    #     image = image.unsqueeze(0).to(device)
    #     raw_image = Image.open(f"{image_root}/{image_name}").convert('RGB')
    #     inputs = processor(raw_image, return_tensors="pt").to("cuda")
    #     out = model.generate(**inputs)
    #     caption = processor.decode(out[0], skip_special_tokens=True)
    #     candidate[image_name] = [caption] # 生成的結果
    #     reference[image_name] = captions # 答案
    #     # print('caption:', caption)
        
    #     cnt += 1
    #     if cnt % 300 == 0:
    #         print('cnt:', cnt)
    #         # break

    # cider_score, _ = cider_scorer.compute_score(reference, candidate)
    # spice_score, _ = spice_scorer.compute_score(reference, candidate)
    # print(f"CIDEr Score: {cider_score}")
    # print(f"SPICE Score: {spice_score}")


    #### my model ####
    pretrained_path = 'saveModels/BlipV1.0.pth'
    model = blip_decoder(pretrained=pretrained_path, image_size=224, vit='base')
    model.eval()
    model = model.to(device)

    cnt = 0
    total_cider_score = 0
    total_spice_score = 0
    candidate = {}
    reference = {}
    with torch.no_grad():
        for i in range(eval_dataset.__len__()):
            image, captions, image_name = eval_dataset[i]
            image = image.unsqueeze(0).to(device)
            caption = model.generate(image, sample=False, num_beams=1, max_length=40, min_length=5)
            candidate[image_name] = caption # 生成的結果
            reference[image_name] = captions # 答案
            
            cnt += 1
            if cnt % 1000 == 0:
                print('cnt:', cnt)

    # print(f"Average CIDEr Score: {total_cider_score/cnt}")
    # print(f"Average SPICE Score: {total_spice_score/cnt}")
    cider_score, _ = cider_scorer.compute_score(reference, candidate)
    spice_score, _ = spice_scorer.compute_score(reference, candidate)
    print(f"CIDEr Score: {cider_score}")
    print(f"SPICE Score: {spice_score}")