# Rhythm Score Calculator: Audio Rhythm Analysis Tool

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

## Installation Requirements
```bash
pip install torch transformers librosa numpy gradio soundfile
