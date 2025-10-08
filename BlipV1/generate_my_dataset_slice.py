import pandas as pd
from PIL import Image

image_root = '/home/chris/Desktop/VLM/datasets/newsDataset/images'
csv_path = '/home/chris/Desktop/VLM/datasets/newsDataset/train_fixed_full.csv'


df = pd.read_csv(csv_path)
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

_image_list = []
_caption_list = []
for idx, key in enumerate(sampleDict):
    try:
        raw_image = Image.open(f"{image_root}/{key}").convert('RGB')
        anwsers = sampleDict[key]

        # 未答標直接中斷
        anwsers_len = len(anwsers)
        if anwsers_len < 5:
            print(f'anwser 小於 5, len: {len(sampleDict[key])}, {key}')

        if idx % 100 == 0:
            print(f'idx: {idx}')
        
        for i in range(anwsers_len):
            _image_list.append(key)
            _caption_list.append(anwsers[i])
    
    except Exception as e:
        print(f'key: {key}, e: {e}')

# 建立 csv
output_df = pd.DataFrame({
    'image': _image_list,
    'caption': _caption_list
    })
output_df.to_csv('/home/chris/Desktop/VLM/datasets/newsDataset/train_fixed.csv', index=False)
print(output_df)


#/home/chris/Desktop/VLM/datasets/newsDataset/images/here-are-the-latest-developments_9fac4c0d
#/home/chris/Desktop/VLM/datasets/newsDataset/images/cad53830-7b93-42ee-96c7-4591bf114ab5_52b9edbe.webp
