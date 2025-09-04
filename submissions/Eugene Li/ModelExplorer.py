from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time

model_path = r"C:\Users\Eugene\Documents\Python\training-folder"
image_path = r"C:\Users\Eugene\Documents\Python\bee.jpg"

model = AutoModelForImageClassification.from_pretrained(model_path)
checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)
image = Image.open(image_path).convert("RGB")
inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_idx = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_idx]
print("Predicted label: ", predicted_label)

# Surprise!
if predicted_label == "bee":
    img = Image.open("buzzzz.jpg")
    number_of_spam_images = 15
    print("GO GEORGIA TECH!!!!!!!!!!!!!!!")
    print("buzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")

    for _ in range(number_of_spam_images):
        width = np.random.randint(100, 1000)
        height = np.random.randint(100, 1000) 
        resized_img = img.resize((width, height))
        resized_img.show()
        time.sleep(0.5)
        