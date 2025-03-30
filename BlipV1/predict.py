from PIL import Image
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# custom
from models.blip import blip_decoder

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
image = '/home/chris/Desktop/VLM/datasets/flickr8k/images/1000268201_693b08cb0e.jpg'
text = 'A child in a pink dress is climbing up a set of stairs in an entry way'
im = load_image(image, image_size=224, device=device)

model = blip_decoder(pretrained=pretrained_path, image_size=224, vit='base')
model.eval()
model = model.to(device)

print('im', im.shape)

with torch.no_grad():
    caption = model.generate(im, sample=False, num_beams=1, max_length=40, min_length=5)
    out = 'Caption: ' + caption[0]
    print('origin text', text)
    print('out', out)
