config = {
    "image_root": '/home/chris/Desktop/VLM/datasets/flickr8k/images',
    "caption_file": '/home/chris/Desktop/VLM/datasets/flickr8k/captions.csv',
    "image_size": 224,

    # 模型
    "pretrained": False,
    "vit": 'base',
    "vit_grad_ckpt": False,
    "vit_ckpt_layer": 0,
    # "prompt": "a picture of ",
    "prompt": "",

    # 優化器
    "init_lr": 1e-5,
    "weight_decay": 0.05,

    # 訓練
    "max_epoch": 5,
    "min_lr": 0
}