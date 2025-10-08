import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


def load_captions(caption_file):
    """
    讀取 caption.txt，返回圖片名稱到標註的映射
    """
    df = pd.read_csv(caption_file)
    samples = list(df.itertuples(index=False, name=None))
    return samples


class flickr8k_dataset(Dataset):

    def __init__(self,
                 transform,
                 image_root,
                 caption_file,
                 max_words=30,
                 prompt=''):
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        '''

        self.image_root = image_root
        self.samples = load_captions(caption_file)  # 讀取所有 (image, caption) 對
        self.transform = transform
        self.prompt = prompt

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_name, caption = self.samples[index]
        if not caption:
            caption = "a picture"
            print('caption', caption, type(caption))
        try:
            current_dir = os.path.dirname(__file__)
            image_path = os.path.join(current_dir, self.image_root, image_name)
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)

            caption = self.prompt + caption
            return image, caption
        except Exception as e:
            print(f'{e}, image_name: {image_name}, caption: {caption}')


class pretrain_dataset(Dataset):

    def __init__(self,
                 transform,
                 processor,
                 image_root,
                 caption_file,
                 max_words=30,
                 prompt=''):
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        '''

        self.image_root = image_root
        self.samples = load_captions(caption_file)  # 讀取所有 (image, caption) 對
        self.transform = transform
        self.processor = processor
        self.prompt = prompt

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_name, caption = self.samples[index]
        if not caption:
            caption = "a picture"
            print('caption', caption, type(caption))
        try:
            current_dir = os.path.dirname(__file__)
            image_path = os.path.join(current_dir, self.image_root, image_name)
            image = Image.open(image_path).convert('RGB')
            
            caption = self.prompt + caption
            inputs = self.processor(text=caption,
                                    images=image,
                                    max_length=40,
                                    return_tensors="pt",
                                    padding="max_length",
                                    truncation=True)
            return inputs
        except Exception as e:
            print(f'{e}, image_name: {image_name}, caption: {caption}')
