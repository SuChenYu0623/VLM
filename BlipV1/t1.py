import json
import pandas as pd

file_path = '/home/chris/Desktop/VLM/datasets/coco/annotations_trainval2014/annotations/captions_train2014.json'


# 1. 讀入 COCO 格式的 JSON
with open(file_path, 'r', encoding='utf-8') as f:
    coco = json.load(f)

# 2. 建立 image_id -> file_name 的對應
id2file = {img['id']: img['file_name'] for img in coco['images']}

# 3. 迭代 captions (假設 annotations 裡每筆有 'caption' 欄位)
rows = []
for ann in coco['annotations']:
    img_file = id2file[ann['image_id']]
    caption = ann.get('caption', '')  # 如果 key 不同，請改成正確的欄位名稱
    rows.append({'image': img_file, 'caption': caption})

# 4. 輸出成 DataFrame，並存成 CSV
df = pd.DataFrame(rows)
df.to_csv('captions.csv', index=False, encoding='utf-8-sig')

# （選擇性）檢視前幾筆
print(df.head())
print(df)

    