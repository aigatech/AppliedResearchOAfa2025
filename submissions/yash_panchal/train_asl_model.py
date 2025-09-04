# train_asl_model.py (FINAL VERSION - ALL CHARACTERS)
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
import evaluate
import torch
from torchvision.transforms import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomRotation,
    ToTensor,
    Normalize,
    Resize,
    CenterCrop,
)

# Load the full dataset from the 'data' folder
dataset = load_dataset("imagefolder", data_dir="./data")

# Split the full dataset into training and validation sets
split_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
train_set = split_dataset['train']
val_set = split_dataset['test']

# (The rest of the script is the same as the optimized version)
labels = train_set.features["label"].names
model_name = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name)

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]
normalize = Normalize(mean=image_mean, std=image_std)

_train_transforms = Compose([
    RandomResizedCrop(size),
    RandomHorizontalFlip(),
    RandomRotation(10),
    ToTensor(),
    normalize,
])
_val_transforms = Compose([
    Resize(size),
    CenterCrop(size),
    ToTensor(),
    normalize,
])

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(img.convert("RGB")) for img in examples['image']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(img.convert("RGB")) for img in examples['image']]
    return examples

train_set.set_transform(train_transforms)
val_set.set_transform(val_transforms)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

model = ViTForImageClassification.from_pretrained(
    model_name, num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes=True,
)

metric = evaluate.load("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

training_args = TrainingArguments(
    output_dir="./asl-finetuned-model",
    per_device_train_batch_size=16,
    num_train_epochs=10,
    logging_steps=50,
    save_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    save_total_limit=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("Starting final training run on all collected letters...")
trainer.train()
trainer.save_model("./asl-finetuned-model")
print("✅ Final model saved to ./asl-finetuned-model")
