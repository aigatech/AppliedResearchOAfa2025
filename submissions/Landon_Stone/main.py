from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from MoviesData import movies


# Load in the Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Compute vectors based on movie descriptions
movie_vectors = model.encode([movie['mood'] for movie in movies])

# Ask users for their mood
user_mood = input("Describe your mood, feelings, and day today: ")

# Convert mood to vectors
mood_embedding = model.encode([user_mood])

# Compute similarity between mood and movies
similarities = cosine_similarity(mood_embedding, movie_vectors)[0]

# Recommend top 3 movies based on similarity
top_indices = similarities.argsort()[-3:][::-1]

# Print the top 3 movies as well as similarity score
print("\nThese following movies should best match your current vibe: ")
for i in top_indices:
    print(f"{movies[i]['title']}: {movies[i]['description']}")
    print(f"Similarity Score: {round(similarities[i]*100, 2)}")

