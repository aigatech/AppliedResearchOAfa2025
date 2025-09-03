import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageCaptioner:
    def __init__(self):
        self.processor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    def generate_caption_from_path(self, image_path):
        image = Image.open(image_path)
        return self.generate_caption_from_image(image)
    
    def generate_caption_from_image(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        inputs = self.processor(image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values=inputs['pixel_values'],
                max_new_tokens=50
            )

        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption

