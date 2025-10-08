import pandas as pd

df1 = pd.read_csv(
    '/home/chris/Desktop/VLM/datasets/newsDataset/train_fixed.csv')
df2 = pd.read_csv(
    '/home/chris/Desktop/VLM/datasets/newsDataset/test_fixed.csv')


def parseSampleDict(df, limit=1000):
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

    return {key: sampleDict[key] for key in list(sampleDict.keys())[:limit]}


sampleDictDf1 = parseSampleDict(df1, limit=7000)
sampleDictDf2 = parseSampleDict(df2, limit=1000)
sampleDict = {**sampleDictDf1, **sampleDictDf2}
print('sampleDict:', len(sampleDict))
print('sampleDictDf1:', len(sampleDictDf1))
print('sampleDictDf2:', len(sampleDictDf2))


_image_list = []
_caption_list = []
for idx, key in enumerate(sampleDict):
    for caption in sampleDict[key]:
        _image_list.append(key)
        _caption_list.append(caption)

_df = pd.DataFrame({
    'image': _image_list,
    'caption': _caption_list
    })
_df.to_csv('/home/chris/Desktop/VLM/datasets/newsDataset/test_8k.csv', index=False)