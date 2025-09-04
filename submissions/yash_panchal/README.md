# Project: ASL Alphabet Classifier

This is my submission for the AI@GT Applied Research Fall 2025 assessment. It is a real-time American Sign Language (ASL) letter translator that uses a custom-trained Vision Transformer model hosted on the Hugging Face Hub.

---
## What it Does
This project fine-tunes a pre-trained Vision Transformer (`google/vit-base-patch16-224-in21k`) on a custom-collected dataset of static ASL signs (this means that J and Z are not included because those involve motions). The final application uses a camera to classify these signs in real-time.

The fine-tuned model is hosted on the Hugging Face Hub and is downloaded automatically when the application is run.

Here are some resources to use to try on the application:

https://teachbesideme.com/asl-alphabet-printable-chart-and-flashcards/, https://sign.mt/?lang=en, https://www.youtube.com/watch?v=6_gXiBe9y9A&ab_channel=OurBergLife

**My model can be found here:** (https://huggingface.co/yashcpanchal/ASL_Alphabet_Classifier)

---
## How to Run It

**1. Prerequisites:**
* Python 3.11

**2. Setup:**
Create and activate a virtual environment, then install the required packages.
```bash
# Create and activate a virtual environment
# Copy and paste this into terminal
python -m venv .venv
# On Windows: .\.venv\Scripts\activate
# On Mac/Linux: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**3. Run the Application:**
This will run the translator. The first time you run it, it will download my custom fine-tuned model from the Hugging Face Hub.
```bash
python translator_custom.py
```
Press 'q' to quit the application or do ctrl-c in the terminal.

---
## Development Process
The repository also includes the scripts used to create the model:
* `collect_images.py`: To capture images for the dataset.
* `train_asl_model.py`: To fine-tune the ViT model on the collected data.