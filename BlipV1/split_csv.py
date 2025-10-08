import pandas as pd


caption_path = '/home/chris/Desktop/VLM/datasets/coco_train/captions.csv'
df = pd.read_csv(caption_path)
print(df)

image_list = list(df['image'].values)
caption_list = list(df['caption'].values)

sampleDict = {}
for idx, image in enumerate(image_list):
    caption = caption_list[idx]
    if image in sampleDict:
        if len(sampleDict[image]) == 5:
            continue
        sampleDict[image].append(caption)
    else:
        sampleDict[image] = [caption]

print('sampleDict:', len(sampleDict))

_train_image_list = []
_train_caption_list = []
_test_image_list = []
_test_caption_list = []
for idx, key in enumerate(sampleDict):
    for caption in sampleDict[key]:
        if idx < 0.9 * len(sampleDict):
            _train_image_list.append(key)
            _train_caption_list.append(caption)
        else:
            _test_image_list.append(key)
            _test_caption_list.append(caption)


train_df = pd.DataFrame({
    'image': _train_image_list,
    'caption': _train_caption_list
    })
train_df.to_csv('/home/chris/Desktop/VLM/datasets/coco_train/train.csv', index=False)

test_df = pd.DataFrame({
    'image': _test_image_list,
    'caption': _test_caption_list
    })
test_df.to_csv('/home/chris/Desktop/VLM/datasets/coco_train/test.csv', index=False)