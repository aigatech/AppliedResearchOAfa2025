from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    DefaultDataCollator,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, ColorJitter, RandomHorizontalFlip
import evaluate
import numpy as np

# Adjust number of images by modifying ":xxx"         Can also change dataset to anything from HuggingFace
data = load_dataset("dummybrendan/animals", split="train[:1000]")  
data = data.train_test_split(test_size=0.2)

# Uncomment to check data if necessary
# print(data)
# print(data["train"][0])
# print(data["train"].features)

# Clean and format labels correctly
labels = list(set([t.strip() for t in data["train"]["text"]]))
# Sort labels to ensure consistency between different runs
labels.sort()

# Map labels to int IDs and vice versa
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)} 

# Add int ID to each dataset row
def add_int_ID(dataset_row):
    dataset_row["labels"] = label2id[dataset_row["text"].strip()]  # strip whitespace
    return dataset_row

data = data.map(add_int_ID)

checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

# Resize image
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)

transform = Compose([
    RandomResizedCrop(size),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ToTensor(),
    normalize,
])

# Convert image to RGB 
def transforms(dataset_rows):
    dataset_rows["pixel_values"] = [transform(img.convert("RGB")) for img in dataset_rows["image"]]
    del dataset_rows["image"]  # Only keep labels and pixel values
    return dataset_rows

data = data.with_transform(transforms)

data_collator = DefaultDataCollator()
accuracy = evaluate.load("accuracy")

def compute_metrics(prediction):
    predictions, labels = prediction
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

training_args = TrainingArguments(
    output_dir=r"C:\Users\Eugene\Documents\Python",
    remove_unused_columns=False,
    eval_strategy="epoch",  # Or evaluation_strategy, depending on version or Mac/Windows...:(
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    use_cpu=True,  # If computer doesn't have GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    compute_metrics=compute_metrics,
)

trainer.train()