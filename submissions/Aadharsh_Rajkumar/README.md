# Project Name: Movie Review Analysis and Insights

# Summary: Using pre-trained models to run analysis on a labeled dataset

## What It Does
# Uses three HuggingFace pipelines to analyze IMDB movie reviews from the HuggingFace IMDB dataset. 
# 1. Sentiment Analysis Pipeline - For classifying movie reviews as positive or negative.
# 2. Named Entity Recognition Pipeline - For identifying common people and places from the reviews.
# 3. Summarization Pipeline - For generating a summary of each review. 

# Users can input reviews that they generate through the CLI to recieve model-produced sentiment analysis, entities, and summaries. 

# Users can save analysis on datasets into a CSV. 

# Users can view a plot showing the proportions of positive and negative reviews for each batch.

## How to Run
# 1. Install Dependencies - pip install -r requirements.txt

# 2. Analaysis of IMDB data - python main.py dataset --limit <> --save or python3 main.py dataset --limit <> --save
#   Input the number of reviews to analyze next to limit (default is 5). Example: python main.py dataset --limit <10>
#   The save tag is for saving results to a CSV.

# 3. Analyze Custom Review through CLI - python main.py custom