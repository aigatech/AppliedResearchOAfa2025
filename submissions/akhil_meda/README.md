**Harmful Comment Filter**

This project is a Python application designed to detect and neutralize harmful text using machine learning models from the Hugging Face library. It features an interactive graphical user interface (GUI) built with Gradio, allowing users to input text and receive a safe, rewritten, or censored version in real time.

**Key Features**

* Toxicity Detection: Identifies generally toxic language using a unitary/toxic-bert model.

* Self-Harm Detection: Flags content related to self-harm or suicide using a vibhorag101/roberta-base-suicide-prediction-phr model.

* Automated Rewriting: Attempts to rewrite harmful messages into safer alternatives with s-nlp/bart-base-detox.

* Hard Censorship: If the text can’t be rewritten after three attempts, replaces all words with asterisks (*) as a final measure.

* Interactive GUI: A simple, user-friendly Gradio interface to interact with the models.

**How It Works**

The app creates three Hugging Face pipelines: a toxicity classifier, a self-harm classifier, and a text rewriter.

When a user submits text, the app checks for harmful content with the two classifiers.

If flagged, it enters a loop that tries up to three rewrites using the detox model, re-checking after each attempt.

If the message is still harmful after all attempts, a regex-based hard censor masks every word with asterisks.

The GUI displays the final output and its status: Safe, Rewritten, Censored, or Harmful.

**Setup and Usage**


**Prerequisites**

* Python installed
* Libraries: transformers, gradio (and typically torch as a backend for transformers)

**Installation**

    pip install transformers gradio

if needed install 

    pip install torch

**Run the Application**

    python app.py

**Access the Interface**

Open the local URL shown in the terminal (usually http://127.0.0.1:7860) in your browser to use the app.

