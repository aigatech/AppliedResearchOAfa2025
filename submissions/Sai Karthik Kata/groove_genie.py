"""
Project Title: Groove Genie 🧞‍♂️🎶  
---------------------------------
What it does:  
Groove Genie is an AI-powered DJ and music recommender system.  
It uses NLP to understand text-based descriptions (e.g., "upbeat dance track" or "chill acoustic vibes")
and finds songs that match your mood. Under the hood, it combines FAISS for similarity search, 
Sentence Transformers for embeddings, and zero-shot classification for interpreting user intent.  
It supports filtering by genres, year ranges, and popularity.  

How to run it:  
1. Make sure you have Python 3.9+ installed.  
2. Install the required dependencies:  
   ```bash
   pip install pandas numpy faiss-cpu scikit-learn sentence-transformers transformers
   
3. The csv file required for this project can be found at:
https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks

I reccomend renaming the file to songs.csv since that what I used but this can be changed
by using the global variable at the top of the file.
"""


import pandas as pd
import numpy as np
import faiss
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pickle
import os
from typing import List, Dict, Tuple, Optional

"""USER INPUT NECESSARY"""
SONGS_CSV_PATH = "songs.csv"
build = True
save_index = False

class MusicRecommenderWithFaiss:
    def __init__(self):
       
        
        """
        Initialize the music recommender with Faiss indexing
        """
        self.feature_columns = [
            'danceability', 'energy', 'valence', 'acousticness', 
            'instrumentalness', 'speechiness', 'liveness', 'loudness', 'tempo'
        ]
        
        # Initialize components
        self.scaler = StandardScaler()
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.classifier = pipeline("zero-shot-classification", 
                                   model="facebook/bart-large-mnli")
        
        # Faiss index
        self.index = None
        self.df = None
        self.normalized_features = None
        
        # Metadata for efficient filtering 
        self.genre_to_ids = {}
        self.year_to_ids = {}
        
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the dataset
        """
        df = df.copy()
        
        # Handle missing values
        df[self.feature_columns] = df[self.feature_columns].fillna(df[self.feature_columns].median())
        
        # Normalize tempo and loudness
        if 'tempo' in df:
            df['tempo'] = df['tempo'] / 200.0  # Rough normalization
        if 'loudness' in df:
            df['loudness'] = (df['loudness'] + 60) / 60  # Normalize (-60 to 0)
        
        # Clip values to [0,1]
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = np.clip(df[col], 0, 1)
        
        return df
    
    def build_metadata_indices(self, df: pd.DataFrame):
        """
        Build metadata indices for efficient filtering
        """
        if 'genre' in df:
            for idx, genre in enumerate(df['genre'].fillna('unknown')):
                self.genre_to_ids.setdefault(genre, []).append(idx)
        
        if 'year' in df:
            for idx, year in enumerate(df['year'].fillna(0)):
                try:
                    year_decade = (int(year) // 10) * 10
                except ValueError:
                    continue
                self.year_to_ids.setdefault(year_decade, []).append(idx)
    
    def build_index(self, df: pd.DataFrame, index_type: str = "IVF"):
        """
        Build Faiss index from the dataset
        """
        print("Preprocessing dataset...")
        self.df = self.preprocess_dataset(df)
        
        print("Building metadata indices...")
        self.build_metadata_indices(self.df)
        
        print("Normalizing features...")
        # Extract features and scale
        feature_matrix = self.df[self.feature_columns].values.astype(np.float32)
        self.normalized_features = np.ascontiguousarray(
            self.scaler.fit_transform(feature_matrix).astype(np.float32)
        )
        
        # Normalize for cosine similarity
        faiss.normalize_L2(self.normalized_features)
        
        # Build Faiss index
        dimension = len(self.feature_columns)
        print(f"Building {index_type} index with dimension {dimension}...")
        
        if index_type == "Flat":
            self.index = faiss.IndexFlatIP(dimension)  
            
        elif index_type == "IVF":
            nlist = min(100, len(self.df) // 10) or 1
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.index.train(self.normalized_features)
            self.index.nprobe = min(10, nlist)  
            
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(dimension, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 100
        
        self.index.add(self.normalized_features)
        print(f"Index built successfully with {self.index.ntotal} vectors")
    
    def extract_music_attributes(self, text: str) -> Dict:
        """
        Extract music attributes from text using zero-shot classification
        """
        moods = ["danceability", "energy", "valence", "acousticness", 
                 "instrumentalness", "speechiness", "liveness", 
                 "loudness", "tempo"]
        
        try:
            mood_result = self.classifier(text, moods)
            return {'moods': dict(zip(mood_result['labels'], mood_result['scores']))}
        except Exception as e:
            print(f"Classification error: {e}")
            return {'moods': {m: 0.5 for m in moods}}
    
    def text_to_audio_features(self, text: str) -> np.ndarray:
        """
        Convert text description to audio feature vector
        """
        attributes = self.extract_music_attributes(text)
        moods = attributes['moods']
        
        features = {col: moods.get(col, 0.5) for col in self.feature_columns}
        
        feature_vector = np.array([features[col] for col in self.feature_columns])
        return feature_vector.astype(np.float32)
    
    def get_filtered_ids(self, genres: Optional[List[str]] = None, 
                     year_range: Tuple[int, int] = None, 
                     popularity_min: Optional[int] = None) -> Optional[List[int]]:
        """
        Get filtered song IDs based on constraints
        """
        valid_ids = set(range(len(self.df)))
        
        # --- Genre filtering ---
        if genres:
            genre_ids = set()
            for user_genre in genres:
                for g, ids in self.genre_to_ids.items():
                    if user_genre.lower().strip() in g.lower():
                        genre_ids.update(ids)
            valid_ids &= genre_ids
        
        # --- Year filtering ---
        if year_range:
            year_ids = set()
            start_decade = (year_range[0] // 10) * 10
            end_decade = (year_range[1] // 10) * 10
            for decade in range(start_decade, end_decade + 10, 10):
                if decade in self.year_to_ids:
                    for idx in self.year_to_ids[decade]:
                        if year_range[0] <= self.df.iloc[idx]['year'] <= year_range[1]:
                            year_ids.add(idx)
            valid_ids &= year_ids
        
        # --- Popularity filtering ---
        if popularity_min is not None:
            pop_ids = set(self.df[self.df['popularity'] >= popularity_min].index.tolist())
            valid_ids &= pop_ids
        
        return list(valid_ids) if valid_ids else None


    def recommend(self, user_text: str, n_recommendations: int = 10,
              genres: Optional[List[str]] = None, 
              year_range: Tuple[int, int] = None,
              popularity_min: int = None) -> pd.DataFrame:
        """
        Get music recommendations based on text input
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        print(f"Processing query: '{user_text}'")
        
        query_vector = self.text_to_audio_features(user_text).reshape(1, -1)
        query_normalized = self.scaler.transform(query_vector).astype(np.float32)
        faiss.normalize_L2(query_normalized)
        
        filtered_ids = self.get_filtered_ids(genres, year_range, popularity_min)
        if filtered_ids is not None and len(filtered_ids) == 0:
            return pd.DataFrame()
        
        if filtered_ids is None:
            similarities, indices = self.index.search(query_normalized, n_recommendations)
        else:
            if len(filtered_ids) < n_recommendations:
                n_recommendations = len(filtered_ids)
            subset_features = self.normalized_features[filtered_ids].copy()
            faiss.normalize_L2(subset_features)
            temp_index = faiss.IndexFlatIP(len(self.feature_columns))
            temp_index.add(subset_features)
            similarities, temp_indices = temp_index.search(query_normalized, n_recommendations)
            indices = np.array([[filtered_ids[idx] for idx in temp_indices[0]]])
        
        result_indices = indices[0]
        result_similarities = similarities[0]
        
        recommendations = self.df.iloc[result_indices].copy()
        recommendations['similarity_score'] = result_similarities
        recommendations['rank'] = range(1, len(recommendations) + 1)
                
        return recommendations[['rank', 'artist_name', 'track_name', 'genre', 'year', 
                                'popularity', 'similarity_score']]
        
    def read_index(self, filepath: str):
        self.index = faiss.read_index(f"{filepath}_faiss.index")
        with open(f"{filepath}_data.pkl", 'rb') as f:
            save_data = pickle.load(f)
        self.df = save_data['df']
        self.scaler = save_data['scaler']
        self.feature_columns = save_data['feature_columns']
        self.normalized_features = save_data['normalized_features']
        self.genre_to_ids = save_data['genre_to_ids']
        self.year_to_ids = save_data['year_to_ids']
        print(f"Index loaded from {filepath}")
    
    def save_index(self, filepath: str):
        faiss.write_index(self.index, f"{filepath}_faiss.index")
        save_data = {
            'df': self.df,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'normalized_features': self.normalized_features,
            'genre_to_ids': self.genre_to_ids,
            'year_to_ids': self.year_to_ids
        }
        with open(f"{filepath}_data.pkl", 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Index saved to {filepath}_faiss.index and {filepath}_data.pkl")

def main():
    df = pd.read_csv(SONGS_CSV_PATH)
    filename = "music_recommender"
    
    recommender = MusicRecommenderWithFaiss()

    if build:
        recommender.build_index(df, index_type="Flat")
    elif (os.path.exists(f"{filename}_faiss.index") and os.path.exists(f"{filename}_data.pkl")):
        recommender.read_index(filename)
    else:
        recommender.build_index(df, index_type="Flat")
        if(save_index):
            recommender.save_index(filename)

    query = ""
    while query != "q":
        query = input("Enter user query (use q to exit)\n")
        if query == "q":
            break
        if not query.strip():
            print("⚠️ Please enter a non-empty query.")
            continue
        
        # --- Ask user for genres ---
        genre_input = input("Enter genres (comma separated):\n")
        genres = [g.strip() for g in genre_input.split(",") if g.strip()] if genre_input else None

        # --- Toggle Year or Popularity Filtering ---
        filter_choice = input("Filter by (1) Year range, (2) Popularity, (3) Both, or (Enter to skip):\n").strip()

        year_range = None
        popularity_min = None

        if filter_choice == "1" or filter_choice == "3":
            try:
                year_min = int(input("Enter minimum year (e.g. 1990):\n"))
                year_max = int(input("Enter maximum year (e.g. 2020):\n"))
                year_range = (year_min, year_max)
            except ValueError:
                print("⚠️ Invalid year input, skipping year filter.")

        if filter_choice == "2" or filter_choice == "3":
            try:
                popularity_min = int(input("Enter minimum popularity (0–100):\n"))
            except ValueError:
                print("⚠️ Invalid popularity input, skipping popularity filter.")

        # --- Get Recommendations ---
        recommendations = recommender.recommend(
            query,
            n_recommendations=50,
            genres=genres,
            year_range=year_range,
            popularity_min=popularity_min
        )
        
        if recommendations.empty:
            print("\nNo recommendations found for the given filters.")
        else:
            print("\nRecommendations:")
            print(recommendations.to_string(index=False))

if __name__ == "__main__":
    main()
