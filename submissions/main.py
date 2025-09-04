from transformers import pipeline
import random

# get moood classifier
emotion_classifier = pipeline("text-classification", 
                              model="j-hartmann/emotion-english-distilroberta-base", 
                              return_all_scores=True)

# get the mood from user input
user_input = input("Describe your current mood in a few words: ")
emotions = emotion_classifier(user_input)[0]
mood = max(emotions, key=lambda x: x["score"])["label"]

# map mood categories to playlist generes
mood_to_genre = {
    "joy": "Upbeat Pop",
    "sadness": "Lo-fi Chill",
    "anger": "Heavy Metal",
    "love": "Romantic Acoustic",
    "fear": "Dark Synthwave",
    "surprise": "Experimental Indie"
}
playlist_genre = mood_to_genre.get(mood, "Eclectic Mix")

''' future area for expansion: suggesting songs for user based on mood
# placeholder songs
fake_songs = [
    "Lost in the Glow", "Neon Dreams", "Echoes of You",
    "Fading Skyline", "Parallel Hearts", "Starlight Drift"
]
playlist = random.sample(fake_songs, 3)
'''

# output results
print(f"🎶 Mood detected: {mood}")
print(f"🎧 Playlist theme: {playlist_genre}")

# generate album cover based on mood
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.to("cpu") # running on my cpu but could do cuda if it works

prompt = f"A dreamy album cover in {playlist_genre} style"
image = pipe(prompt).images[0]
image.save("cover.png") # saving image  locally


