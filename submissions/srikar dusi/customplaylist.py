import streamlit as st
from transformers import pipeline
import random

playlists = {
    "anger": [
        "https://open.spotify.com/playlist/37i9dQZF1EIgNZCaOGb0Mi"
    ],
    "fear": [
        "https://open.spotify.com/playlist/37i9dQZF1EIfMwRYymgnLH"
    ],
    "joy": [
        "https://open.spotify.com/playlist/37i9dQZF1EIdrEY58wGgwS"
    ],
    "love": [
        "https://open.spotify.com/playlist/37i9dQZF1EIfBRbPAJgykh"
    ],
    "sadness": [
        "https://open.spotify.com/playlist/37i9dQZF1EIg85EO6f7KwU"
    ],
    "surprise": [
        "https://open.spotify.com/playlist/37i9dQZF1EIgbSpJ037RGg"
    ]
}

def load_model():
    return pipeline(
        "text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True
    )

classifier = load_model()

st.title("🎵 Emotion-Based Spotify Playlist Generator!")

user_input = st.text_area("Enter your current mood or feelings:")

if st.button("Generate Playlist") and user_input.strip():
    results = classifier(user_input)[0]
    # Find top emotion
    top_emotion = max(results, key=lambda x: x['score'])['label'].lower()
    
    st.subheader(f"Top Emotion Detected: {top_emotion.capitalize()}")
    
    # Select playlists for top emotion
    final_playlists = playlists[top_emotion]
    
    st.subheader("Your Recommended Spotify Playlist 🎶")
    for i, link in enumerate(final_playlists, 1):
        st.markdown(f'<a href="{link}" target="_blank"><button>Open Playlist</button></a>', unsafe_allow_html=True)