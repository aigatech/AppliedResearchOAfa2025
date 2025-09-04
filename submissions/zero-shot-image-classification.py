import torch
from matplotlib import pyplot as plt
from transformers import pipeline
from PIL import ImageDraw
from transformers.image_utils import load_image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

checkpoint = "iSEE-Laboratory/llmdet_large"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint, device_map="auto")
processor = AutoProcessor.from_pretrained(checkpoint)

url = "https://media.istockphoto.com/id/1094775212/photo/luxury-fashionable-clothing-and-stationery-items-flat-lay-on-white-background.jpg?s=612x612&w=0&k=20&c=qgSKJaydFrd3BarydzDUM74RiCy2fTdaNftK5cMSskc="
image = load_image(url)

text_labels = ["shoes", "camera", "shirt", "bottle", "hat", "cloth", "rubber band"]
inputs = processor(text=text_labels, images=image, return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
   outputs, threshold=0.50, target_sizes=[(image.height, image.width)], text_labels=text_labels,)[0]

draw = ImageDraw.Draw(image)

print(results)

scores = results["scores"]
text_labels = results["text_labels"]
boxes = results["boxes"]

for box, score, text_label in zip(boxes, scores, text_labels):
    xmin, ymin, xmax, ymax = box
    draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
    draw.text((xmin, ymin), f"{text_label}: {round(score.item(),2)}", fill="white")

image.show()