from PIL import Image
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# custom
from models.blip import blip_decoder
from transformers import BlipProcessor, BlipForConditionalGeneration

def load_image(image, image_size, device):
    raw_image = Image.open(str(image)).convert('RGB')

    w, h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_path = 'saveModels/BlipV1.0.pth'
image = '/home/chris/Desktop/VLM/datasets/flickr8k/images/1002674143_1b742ab4b8.jpg'
text = 'A little girl covered in paint sits in front of a painted rainbow with her hands in a bowl .'
im = load_image(image, image_size=224, device=device)

model = blip_decoder(pretrained=pretrained_path, image_size=224, vit='base')
model.eval()
model = model.to(device)

print('im', im.shape)

with torch.no_grad():
    caption = model.generate(im, sample=False, num_beams=1, max_length=40, min_length=5)
    out = '' + caption[0]
    print('origin text:\n', text)
    print('my model out:\n', out)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
raw_image = Image.open(f"{image}").convert('RGB')
inputs = processor(raw_image, return_tensors="pt").to("cuda")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print('pretrain model out:\n', caption)
