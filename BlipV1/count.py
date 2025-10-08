import pandas as pd

dataDict = {
    "flickr": {
        "train": "/home/chris/Desktop/VLM/datasets/flickr30k/captions.csv",
        "test": "/home/chris/Desktop/VLM/datasets/flickr8k/captions.csv",
    },
    "coco": {
        "train": "/home/chris/Desktop/VLM/datasets/newsDataset/train_fixed.csv",
        "test": "/home/chris/Desktop/VLM/datasets/newsDataset/test_fixed.csv",
    },
    "newsDataset": {
        "train": "/home/chris/Desktop/VLM/datasets/coco_train/train.csv",
        "test": "/home/chris/Desktop/VLM/datasets/coco_train/test.csv",
    }
}

for key in dataDict:
    print(f'=== {key} ===')
    for subKey in dataDict[key]:
        path = dataDict[key][subKey]

        df = pd.read_csv(path)
        print(len(df))
