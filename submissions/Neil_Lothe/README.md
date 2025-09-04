# Zero-Shot Topic Classifier

This tool classifies any text or batch of texts into user-defined topics without needing to fine-tune a model.

## Purpose and Features

Loads the `facebook/bart-large-mnli` model via HuggingFace’s zero-shot classification pipeline
Accepts a single text snippet or a CSV (`text` column)  
Returns labels with normalized confidence scores  
Optionally writes predictions to a JSON file

## How to run

1. Activate your virtual environment:  
   ```bash
   python -m venv venv or venv\Scripts\activate.bat
2. Install dependencies: pip install torch --index-url https://download.pytorch.org/whl/cpu transformers datasets


## Usage
You can use this program to classify a single snippet.
    For example:
    python zero_shot_cli.py --text "NASA announces new mission to explore Jupiter's moons." --labels "space, politics, sports" --threshold 0.2

    Sample Output:
    Text: NASA announces new mission to explore Jupiter's moons.
    space           0.9123
    politics        0.0543
    sports          0.0334

    Classify a csv batch:
    python zero_shot_cli.py --file sample_data.csv --labels "technology, politics, entertainment, cooking, science, sports, environment, business" --threshold 0.25 --output results.json
    
## Contact
Neil Lothe - neil.lothe@gmail.com