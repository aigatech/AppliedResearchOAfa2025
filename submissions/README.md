# Rhythm Score Calculator

## Overview
The **Rhythm Score Calculator** is an innovative audio analysis application that evaluates rhythmic performance by:
- Processing audio files or recordings
- Generating a score out of 100 based on rhythmic alignment
- Utilizing advanced machine learning techniques

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
# Clone the repository
git clone https://github.com/yourusername/rhythm-score-calculator.git
cd rhythm-score-calculator

# Install required dependencies
pip install torch transformers librosa numpy gradio soundfile

# Navigate to the project directory
cd rhythm-score-calculator

# Start the application
python main.py

