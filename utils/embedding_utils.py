from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

def load_clip_model(model_name="openai/clip-vit-base-patch32"):
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

def get_image_embedding(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.get_image_features(**inputs)
    return output[0].cpu().numpy()

def get_text_embedding(model, processor, text):
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model.get_text_features(**inputs)
    return output[0].cpu().numpy()
