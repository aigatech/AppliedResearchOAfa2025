# AI Fact Checker with Wikipedia

## Project Title
AI Fact Checker with Wikipedia Evidence

## What it does
This application is an AI-powered fact-checking system that:
- Takes user claims as input (e.g., "The Eiffel Tower is in Paris")
- Extracts the main entity from the claim using SpaCy NLP
- Fetches relevant evidence from Wikipedia
- Analyzes the claim against the evidence using term overlap analysis
- Returns confidence scores for whether the claim is true, false, or uncertain

## LIMITATIONS
- Made in short period
- Only relies on wikipedia for fact checking
- Struggles with ambiguous entities

## How to run it

### Prerequisites
```bash
pip install gradio transformers wikipediaapi spacy torch
python -m spacy download en_core_web_md
```

### Running the application
```bash
cd Vedant_Lalit_Bhatt
python submission.py
```

The application will launch a Gradio web interface at `http://127.0.0.1:7864` where you can:
1. Enter any factual claim in the text box
2. Click submit to get fact-checking results
3. View the Wikipedia evidence used for verification

### Example claims to test:
- "dinosaurs lived in the cretaceous period"
- "donald trump is in the republican party" 
- "the eiffel tower is in paris"
- "strawberries are not berries"
