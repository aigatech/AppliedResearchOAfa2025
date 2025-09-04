# **Project Title: Financial Article Dissector**

**What it does:**  
The program uses FinBERT, which is based on the BERT language model. It is a NLP model trained for sentiment classification of financial text. This project dissects financial articles to determine the overall sentiment of the article and can be used to obtain a third-party view of a company's earnings or compare the biases between different articles. It breaks down articles into clusters of 500 words as to not go over the max input token length of 512. This also allows you see how the sentiment of the article changes between different parts. There are also 3 general descriptors outputed: neutral, positive and negative.

**How to Run:**  
Navigate to 'main.py' in the submissions folder and run `main.py'. When the program is run, enter a URL to a financial article when prompted.
