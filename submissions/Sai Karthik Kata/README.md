# Groove Genie 🧞‍♂️🎶  

Groove Genie is an **AI-powered DJ and music recommender system**.  
It takes in natural language queries like *"upbeat dance track"* or *"chill acoustic vibes"* and finds songs that match your mood.  

Under the hood, it uses:  
- **FAISS** for fast similarity search  
- **Sentence Transformers** for text embeddings  
- **Zero-shot classification** to interpret user intent  

It also supports filtering by **genres**, **year ranges**, and **popularity**.  

---

## Features
- Query music with natural language (e.g., *"relaxing piano for studying"*).  
- Filter results by genre(s), popularity, and year range.  
- Fast and scalable recommendation system powered by FAISS.  
- Runs locally.

---

## Ideas for the Future 
- Create a custom model that can more accurately interpret user query.
- Add voice input.
- Integrate into Spotify API.
  
---

## How to Run

### 1. Install Python
Make sure you have **Python 3.9+** installed.  

### 2. Install dependencies
Run the following command to install all required libraries:  
```bash
pip install pandas numpy faiss-cpu scikit-learn sentence-transformers transformers
```
### 3. Download the dataset
[Click Here for the Kaggle Dataset](https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks)
I recommend changing the name of the dataset to songs.csv for simplicity but if you prefer a different filename, 
update the SONGS_CSV_PATH global variable at the top of the Python file.

## 4. Run the file
Run the file 
```bash
python groove_genie.py
```
All customization optins can be changed through the CLI.
