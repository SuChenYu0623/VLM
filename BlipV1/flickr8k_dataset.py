import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


def load_captions(caption_file):
    """
    讀取 caption.txt，返回圖片名稱到標註的映射
    """
    df = pd.read_csv(caption_file)
    samples = list(df.itertuples(index=False, name=None))  
    return samples

class flickr8k_dataset(Dataset):
    def __init__(self, transform, image_root, caption_file, max_words=30, prompt=''):
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
        current_dir = os.path.dirname(__file__)
        image_path = os.path.join(current_dir, self.image_root, image_name)        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)

        caption = self.prompt + caption
        return image, caption