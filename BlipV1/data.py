import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


# custom
from transform.randaugment import RandomAugment
from flickr8k_dataset import flickr8k_dataset

def create_dataset(dataset, config, min_scale=0.5):
    '''
    輸入
    dataset
    config

    輸出
    dataset
    '''
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])        
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])
    
    if dataset == 'flickr8k':
        dataset = flickr8k_dataset(transform_train, config['image_root'], config['caption_file'], prompt=config['prompt'])
        return dataset
    
def create_loader(dataset, sampler, batch_size, num_workers, collate_fn=None):
    # 預設都開
    shuffle = False
    drop_last = True
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler, # 不同類型的取樣模式 依序抽 隨機抽 指定id抽
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    return loader       

if __name__ == '__main__':
    # config = {
    #     # "image_root": '..datasets/flickr8k/images',
    #     # "caption_file": '..datasets/flickr8k/captions.csv',
    #     "image_root": '/home/chris/Desktop/VLM/datasets/flickr8k/images',
    #     "caption_file": '/home/chris/Desktop/VLM/datasets/flickr8k/captions.csv',
    #     "image_size": 224,
    #     "prompt": "a photo of "
    # }
    from configs.config import config
    train_dataset = create_dataset(dataset='flickr8k', config=config, min_scale=0.5)

    dataset = train_dataset
    sampler = RandomSampler(dataset)
    batch_size = 50
    num_workers = 4
    train_dataloader = create_loader(dataset, sampler, batch_size, num_workers)
    print(train_dataset.__len__())
    print(len(train_dataloader))
