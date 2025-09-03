# FootFinder

A computer vision and LLM-powered web application that analyzes foot photos to provide measurements, foot characteristics, and personalized shoe recommendations.

## Features

- **Foot Analysis:** Automatic foot segmentation and measurement from top-down photos
- **Arch Classification:** Identifies flat, normal, or high arch types
- **Width Assessment:** Categorizes foot width as narrow, regular, or wide
- **Size Estimation:** Converts measurements to US men's and women's shoe sizes
- **Scale Calibration:** Uses manual foot measurements for accurate sizing
- **Shoe Recommendations:** AI-powered suggestions based on foot characteristics and intended use

## Installation & Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application:**
   ```bash
   python app.py
   ```

3. **Access the Web Interface:**
   Open your browser and go to the URL displayed in the terminal (typically `http://127.0.0.1:7860`)

## Usage

### Taking the Photo
- Position camera directly above your bare foot
- Ensure good lighting and contrast with the background
- Keep foot flat and straight

### Analysis Options
- **Foot Length:** Enter your measured foot length in inches for accurate sizing
- **Use Case:** Select your intended shoe usage (Walking, Running, Hiking, Court Sports, Cleats/Boots)
- **Sizing Reference:** Choose Men's, Women's, or Unspecified for size recommendations

### Results
The app provides:
- Overlay image with measurement lines at key foot positions
- Foot characteristics (arch type, width category, estimated sizes)
- Personalized shoe recommendations based on your foot profile and intended use

## Technical Details

### Computer Vision (cv_utils.py)
- Otsu thresholding for foot segmentation
- Contour detection and rotation for proper orientation
- Width measurements at forefoot (25%), midfoot (50%), and heel (85%) positions
- Manual measurement calibration for accurate sizing

### AI Recommendations (recommend.py)
- Uses Google's FLAN-T5-small model via Hugging Face Transformers
- Generates category-specific shoe recommendations (no brand names)
- Fallback system for robust recommendations
- CPU-only operation for broad compatibility

### Web Interface (app.py)
- Built with Gradio for easy deployment and use
- Responsive interface with input validation
- Comprehensive usage tips and disclaimers
- Real-time image processing and analysis

## Requirements

- Python 3.10+
- opencv-python-headless
- numpy
- transformers
- torch
- gradio

## Limitations & Disclaimers

- Measurements are approximate and depend on image quality
- Recommendations are general guidelines, not medical advice
- Professional fitting is recommended for optimal shoe selection
- Results may vary based on foot positioning and photo conditions

## File Structure

```
submissions/Vedant_Nahar_FootFit/
├── requirements.txt    # Python dependencies
├── app.py             # Gradio web application
├── cv_utils.py        # Computer vision functions
├── recommend.py       # LLM recommendation system
└── README.md          # This file
```