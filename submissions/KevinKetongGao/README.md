# Toilet Paper Fullness Detector

Have you ever wondered exactly how full your toilet paper is just in case it's at that point where you might need a little bit extra for a particularly large #2?
Well, neither have I, but worry no more: here's the TOILET PAPER FULLNESS DETECTOR for ya!
All it does?: Detects toilet paper rolls and estimates fullness by analyzing paper-to-hole ratio.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python toilet_paper_detector.py
```

1. Upload image with toilet paper rolls facing camera
2. Click "Detect & Analyze"
3. View results with fullness percentages

## Features

- **Object Detection**: Labels all objects in frame (person, chair, bottle, etc.)
- **Toilet Paper Analysis**: Specifically analyzes detected rolls for fullness
- **Visual Labels**: Blue boxes for all objects, red boxes for toilet paper rolls
- **Debug Panel**: Shows all detected objects with confidence scores
- **Fullness Estimation**: Calculates percentage based on paper-to-hole ratio

## How It Works

- Uses DETR model to detect all objects in image
- Filters for circular/rectangular shapes as potential toilet paper rolls
- Analyzes white paper vs dark hole areas
- Calculates fullness percentage for each roll

## Requirements

- Python 3.8+
- Clear, well-lit images
- Toilet paper rolls facing camera