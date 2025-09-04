# DJ Helper App

## Project Title
DJ Helper App – AI-assisted song analysis and recommendation for DJs

## What it does
This app uses AI models to analyze audio files (uploaded directly, not URLs) and generates the following information for each song:

- **Genre**: Identifies the music genre using an AI model.
- **Key**: Detects the musical key of the track.
- **BPM**: Estimates the tempo (beats per minute) of the song.
- **Embedding**: Generates a numerical representation of the audio to compare with other songs.
- **Related Songs**: Finds other songs with similar BPM and key from a dataset.

**Purpose:**  
As a home DJ, it can be difficult to find songs that match the mood or energy of other tracks. This app helps automate that process by analyzing BPM and key, making it easier to create smooth mixes and discover compatible songs.

## Why this approach
- **BPM and key analysis**: These are crucial for DJs to ensure songs mix harmoniously.
- **Embeddings**: Provide a way to numerically represent songs for more advanced similarity searches.
- **Lightweight dataset**: Focuses on sample `.wav` files, keeping the project small and manageable for testing and demonstration purposes.

## How to run it
1. Place your audio file (`.wav`) in the input.
2. The app outputs the genre, key, BPM, embedding, and related songs using a **Gradio interface** as the UI.
3. Results are displayed immediately in the web interface.

## Potential Improvements
- **Spotify API integration**: Connect to Spotify to fetch songs directly and avoid manual key detection.
- **Larger dataset**: Use a more extensive dataset of songs for better recommendations.
- **Additional audio features**: Include features like energy, loudness, or mood detection for more precise song matching.
- **Real-time DJ assistance**: Implement live BPM/key matching while mixing tracks.

## Requirements
- Python >= 3.9  
- Gradio  
- Transformers  
- Librosa  
- Torch  

## Notes
- Only `.wav` files are supported for analysis.
- The app is designed for small-scale testing and demonstration.
- Users can expand it by integrating with streaming services or using larger datasets.
