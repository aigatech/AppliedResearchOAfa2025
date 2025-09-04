# save_processor.py
from transformers import ViTImageProcessor

# This is the same processor we used in the training script
model_name = 'google/vit-base-patch16-224-in21k'

print(f"Loading processor from '{model_name}'...")
processor = ViTImageProcessor.from_pretrained(model_name)

# Save the processor's configuration to your fine-tuned model's directory
output_dir = './asl-finetuned-model'
processor.save_pretrained(output_dir)

print(f"✅ Processor configuration saved successfully to '{output_dir}'!")