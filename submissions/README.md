# Zero-Shot GeoGuessr with CLIP

## What it is
A GeoGuessr AI built with Hugging Face CLIP models.  
It takes random street-view-like photos from the [GeoGuessr Countries dataset](https://huggingface.co/datasets/deboradum/GeoGuessr-countries) and tries to guess the country **without any training** using zero-shot image classification.

- Uses `openai/clip-vit-base-patch32` by default  
- Compares each photo to prompt variants like *"a street scene in Japan"*  
- Aggregates scores across prompts per country  
- Reports **Top-1** and **Top-3** accuracy  
- Prints a few sample predictions

## How to run

1. **Install dependencies** (CPU works, GPU is faster):
   ```bash
   python -m pip install datasets transformers torch pillow tqdm
