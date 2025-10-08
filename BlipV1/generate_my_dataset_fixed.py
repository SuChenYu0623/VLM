import pandas as pd
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import random

banned_words = ['awe', 'a sunset scene in a city park', 'ad for a product', 'ad', 'awe, awe, awe, awe, awe,', 'bbc - bbc news', 'bbc - bbc news - bbc news -', 'bbc - bbc - bbc - b', 'a bbc bbc bbc bbc b', 'a satan satan satan satan s', 'a st ed a st ed a s']

image_root = '/home/chris/Desktop/VLM/datasets/newsDataset/images'
csv_path = '/home/chris/Desktop/VLM/datasets/newsDataset/test.csv'


df = pd.read_csv(csv_path)
image_list = list(df['image'].values)
caption_list = list(df['caption'].values)

sampleDict = {}
for idx, image in enumerate(image_list):
    caption = caption_list[idx]
    if caption in banned_words:
        continue
    if image in sampleDict:
        sampleDict[image].append(caption)
    else:
        sampleDict[image] = [caption]

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16, device_map="auto").to("cuda")

questions = [
    "What is this scene depicting?",
    "Please write a post base on this picture",
    "What do you think the person in the photo is feeling, and why?",
    "Describe the setting of this image as if you were writing a travel brochure.",
    "How does this image reflect the essence?",
    "If this photo were the cover of a magazine, what would the headline be?",
    "Write a short news caption for this photo that could appear in an online article.",
    "What might happen in the next five minutes after this moment?",
    "Imagine you are the person in the image — write a diary entry describing this moment.",
    "In what ways could this image be used in an advertisement? What product would it fit?",
    "If this were part of a video, what would you expect to see or hear next?",
    "Draft a concise news headline that summarizes the story behind this image.",
    "If this were a novel cover, what bold tagline would you choose?",
    "Write a catchy social‐media caption for this photo."
]


# for image, caption in df[['image', 'caption']].values:
#     image_list.append(image)
#     if caption in banned_words:
#         try:
#             raw_image = Image.open(f"{image_root}/{image}").convert('RGB')
#             # random.shuffle(questions)
#             for question in questions:
#                 inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)
#                 out = model.generate(**inputs)
#                 anwser = processor.decode(out[0], skip_special_tokens=True)
#                 if anwser in banned_words:
#                     continue
#                 if len(anwser) < 3:
#                     continue
#                 caption_list.append(anwser)
#                 print(image)
#                 print(question)
#                 print(anwser)

#                 print('-------\n')
#                 # break
#         except Exception as e:
#             print(image, caption, e)
#             caption_list.append('generate failed, plz remove.')
        
#         break
#     else:
#         caption_list.append(caption)

    
print('sampleDict:', len(sampleDict))

_image_list = []
_caption_list = []
for idx, key in enumerate(sampleDict):
    try:
        raw_image = Image.open(f"{image_root}/{key}").convert('RGB')
        anwsers = sampleDict[key]
        random.shuffle(questions)
        for question in questions:
            inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)
            out = model.generate(**inputs)
            anwser = processor.decode(out[0], skip_special_tokens=True)
            if len(anwsers) == 5:
                break
            if anwser in banned_words:
                continue
            if len(anwser.split(' ')) < 4:
                continue
            if anwser not in anwsers:
                anwsers.append(anwser)
            

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
output_df.to_csv('/home/chris/Desktop/VLM/datasets/newsDataset/test_fixed.csv', index=False)



#/home/chris/Desktop/VLM/datasets/newsDataset/images/here-are-the-latest-developments_9fac4c0d
#/home/chris/Desktop/VLM/datasets/newsDataset/images/cad53830-7b93-42ee-96c7-4591bf114ab5_52b9edbe.webp
