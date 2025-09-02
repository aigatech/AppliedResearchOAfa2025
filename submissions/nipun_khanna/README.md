# Movie Review Analyzer

This project leverages sentiment analysis to classify movie reviews by predicted sentiment (Positive, Neutral, Negative, etc). There are two parts to this project:
* A training notebook which contains the code to fine-tune a DistilBERT sentiment analysis model on the IMDB movie review dataset
* An interface application file that launches an interface to test and predict movie review sentiment using the saved model

# How It Works
1. Simply enter a movie review in the interface
    * Ex. "This movie was awesome! The acting was great, plot was wonderful, and there were pythons!"

2. Recieve a corresponding sentiment phrases depending on the predicted score from the ML model
    * Ex. Score of 0.98 -> *Highly Positive* review


# How to Run

1. Download required packages: 
```
pip install datasets evaluate transformers numpy torch torchvision gradio python-dotenv
```
2. Run training.ipynb to train and save a model which will be saved to ``./local_movie_sentiment_classifier``
3. Create a ``.env`` file in the root directory with variable ``MODEL_PATH`` that includes the **absolute** path to the previously saved model
4. Run ``interface.py`` and open the localhost site to view the GUI