# Rhythm Score Calculator

## Overview
The **Rhythm Score Calculator** is an innovative audio analysis application that evaluates rhythmic performance by:
- Processing audio files or recordings
- Generating a score out of 100 based on rhythmic alignment

## Technical Components
| Component | Details |
|-----------|---------|
| Core Technology | **Facebook WAV2Vec2 Base Model** for audio embedding identification |
| Embedding Alignment | Matches audio to a beat grid at a specified BPM |
| Score Calculation | Generates rhythmic accuracy score through advanced processing |
| User Interface | **ChatGPT-generated** testing interface |

## Prerequisites
- Python 3.8+
- pip package manager

## Installation
```bash

# Install required dependencies
pip install torch transformers librosa numpy gradio soundfile

### Navigate to the directory that contains **main.py**

# Start the application
python main.py
```

###Then navigate to the provided local address to upload data and input a bpm. Then click Submit. The embedding identification part (w/ wav2vec2) works fine, but the score calculation is slightly chopped (still works but a 120 bpm file won't get 100 out of 100 for a 120 bpm setting).

