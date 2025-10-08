import pandas as pd
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import random


image_root = '/home/chris/Desktop/VLM/datasets/newsDataset/images'
csv_path = '/home/chris/Desktop/VLM/datasets/newsDataset/source_captions_fixed.csv'


processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16, device_map="auto").to("cuda")

df = pd.read_csv(csv_path)
image_list = list(df['image'].values)
caption_list = list(df['caption'].values)



sampleDict = {}
for idx, image in enumerate(image_list):
    caption = caption_list[idx]
    if image in sampleDict:
        sampleDict[image].append(caption)
    else:
        sampleDict[image] = [caption]

questions = [
    "What is this scene depicting?",
    "Please write a post base on this picture",
    "What do you think the person in the photo is feeling, and why?",
    "Describe the setting of this image as if you were writing a travel brochure.",
    "How does this image reflect the essence of childhood?",
    "How does this image reflect the essence of childhood?",
    "If this photo were the cover of a magazine, what would the headline be?",
    "What time of year do you think this photo was taken? Give reasons.",
    "What safety considerations should be taken into account in situations like the one in the photo?",
    "Write a short news caption for this photo that could appear in an online article.",
    "What might happen in the next five minutes after this moment?",
    "If this image were part of a storybook, what would the title and first sentence be?",
    "Imagine you are the person in the image — write a diary entry describing this moment.",
    "How might this scene be different if it were taken at sunset instead of during the day?",
    "What message or mood do you think the photographer was trying to capture?",
    "What sounds, smells, or sensations do you imagine in this scene?",
    "In what ways could this image be used in an advertisement? What product would it fit?",
    "How would you describe this moment using a metaphor or simile?",
    "If this were part of a video, what would you expect to see or hear next?",
    "What does this image remind you of from your own life or experiences?",
]


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
            if anwser not in anwsers:
                anwsers.append(anwser)
            
            if len(anwsers) == 5:
                break

        # 未答標直接中斷
        anwsers_len = len(anwsers)
        if anwsers_len < 5:
            print(f'anwser 小於 5, len: {len(sampleDict[key])}, {key}')

        if idx % 300 == 0:
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
output_df.to_csv('/home/chris/Desktop/VLM/datasets/newsDataset/captions.csv', index=False)



#/home/chris/Desktop/VLM/datasets/newsDataset/images/here-are-the-latest-developments_9fac4c0d
#/home/chris/Desktop/VLM/datasets/newsDataset/images/cad53830-7b93-42ee-96c7-4591bf114ab5_52b9edbe.webp
