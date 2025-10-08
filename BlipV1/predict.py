from PIL import Image
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# custom
from models.blip import blip_decoder
from transformers import BlipProcessor, BlipForConditionalGeneration


class PredictBase():
    def __init__(self):
        # 使用模型
        pass

    # TODO 載入模型
    def load_model(self):
        ...

    # TODO 預測
    def predict(self, image_path):
        ...


class PredictModel(PredictBase):
    """
    自己訓練的模型
    模型架構與 BLIP 一致 (當前只有 VLM 部份)
    """
    def __init__(self, pretrained_path: str):
        # pretrained_path = 'saveModels/BlipV1.0.pth'
        self.image_size = 224
        self.model = self.load_model(pretrained_path=pretrained_path)
        pass
    
    # TODO load image
    def load_image(self, image_path: str, image_size: int, device: str):
        raw_image = Image.open(str(image_path)).convert('RGB')
        w, h = raw_image.size
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        image = transform(raw_image).unsqueeze(0).to(device)
        return image
    
    # TODO load model
    def load_model(self, pretrained_path: str):
        model = blip_decoder(pretrained=pretrained_path, image_size=224, vit='base')
        return model

    def predict(self, image_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = blip_decoder(pretrained=pretrained_path, image_size=224, vit='base')
        self.model.eval()
        self.model = self.model.to(device)

        im = self.load_image(image_path, self.image_size, device)
        with torch.no_grad():
            caption = self.model.generate(im, sample=False, num_beams=1, max_length=40, min_length=5)
            return caption[0]

class PredictPretrain(PredictBase):
    """
    預訓練訓練的模型
    """
    def __init__(self):
        self.model = self.load_model()
        self.processor = self.load_processor()

    def load_image(self, image_path: str, device: str):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        raw_image = Image.open(f"{image_path}").convert('RGB')
        inputs = self.processor(raw_image, return_tensors="pt").to(device)
        return inputs

    def load_model(self):
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
        return model
    
    def load_processor(self):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        return processor
    
    def predict(self, image_path: str):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inputs = self.load_image(image_path=image_path, device=device)
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_path = '/home/chris/Desktop/VLM/BlipV1/record/newsDatasetV2_50/model.pth'
image_path = '/home/chris/Desktop/VLM/datasets/newsDataset/images/c30q09638n8o_7d18fe15.webp'
text = 'a scene from a saturday evening in a city park'

# c30q09638n8o_7d18fe15.webp,Reuters Pakistani men in Lahore chant slogans at a rally expressing solidarity with the people of Kashmir.
# c30q09638n8o_7d18fe15.webp,Kashmir has been a source of conflict between India and Pakistan for decades
# c30q09638n8o_7d18fe15.webp,a scene from a saturday evening in a city park
# c30q09638n8o_7d18fe15.webp,a metaphor is a figure of speech
# c30q09638n8o_7d18fe15.webp,a man in a crowd of protesters in a protest against the pakistan

# main
print('origin text: \n', text)
predictModel = PredictModel(pretrained_path=pretrained_path)
caption = predictModel.predict(image_path=image_path)
print('blip v1:\n', caption)

predictPretrain = PredictPretrain()
caption = predictPretrain.predict(image_path=image_path)
print('blip pretrain:\n', caption)
