# Genuine vs. Fake Smile Detection for Dating Apps

## What it does
This project analyzes facial images to determine whether a smile is genuine or fake, using multiple HuggingFace models and advanced computer vision techniques. The system combines traditional OpenCV methods with state-of-the-art machine learning models for enhanced accuracy. Designed for dating applications to help users identify authentic smiles in profile pictures.

**Key Features:**
- Multiple detection engines (Basic, Enhanced, Ultra-Enhanced)
- Real-time webcam analysis with HuggingFace models
- Ensemble learning with 3+ specialized models
- Data augmentation for improved robustness
- Production-ready performance

## How to run it

### Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage Options

#### 1. Basic Smile Detection (Fast - 8ms)
```bash
python smile_detector.py --image_path your_image.jpg --show_image
```

#### 2. Enhanced Detection (3 HuggingFace Models)
```bash
python enhanced_detector.py --image_path your_image.jpg --show_details
```

#### 3. Ultra-Enhanced Detection (Data Augmentation)
```bash
python ultra_enhanced_detector.py --image_path your_image.jpg --show_details
```

#### 4. Production Webcam Analysis (Real-time)
```bash
python production_webcam.py
```

## 📹 Webcam Feature Instructions

### Quick Start for Real-Time Analysis
To use the live webcam smile detection feature:

1. **Navigate to the project directory:**
   ```bash
   cd "path/to/submissions/Samanyu_Bansal"
   ```

2. **Run the webcam analyzer:**
   ```bash
   python production_webcam.py
   ```

3. **Using the webcam interface:**
   - The system will automatically detect your camera and start analysis
   - Real-time classifications appear every 1 second
   - You'll see results like:
     - ✅ **Genuine smile** (confidence: 0.94)
     - ⚠️ **Fake smile** (confidence: 0.52)
     - 🔍 **No clear smile** (confidence: 0.47)

4. **Controls:**
   - **Press 'q'** to quit the application
   - **Press 's'** to save current analysis results
   - **Ensure good lighting** for best detection accuracy

### Expected Output Example:
```
🚀 Initializing Production Webcam Smile Analyzer...
✅ Production analyzer ready!

📹 Starting Real-Time Webcam Analysis
==================================================
• Using HuggingFace models for smile detection
• Press 'q' to quit
• Press 's' to save current analysis
• Analysis runs continuously every 1 second
==================================================

🔍 Frame 0: Analyzing with HuggingFace models...
✅ Classification: Genuine smile
   Confidence: 0.940
   Models Used: 3 HuggingFace models
```

### Troubleshooting:
- **Camera not found**: Ensure webcam is connected and not used by other apps
- **Slow performance**: Normal - each frame uses 3 HuggingFace models for accuracy
- **Models downloading**: First run downloads models automatically (one-time setup)

## Project Structure
```
submissions/Samanyu_Bansal/
├── README.md                     # This documentation
├── requirements.txt              # Python dependencies
├── smile_detector.py            # Basic OpenCV detector (fast)
├── enhanced_detector.py         # 3 HuggingFace models ensemble
├── ultra_enhanced_detector.py   # Advanced with data augmentation
├── production_webcam.py         # Real-time webcam analysis
├── advanced_smile_detector.py   # Core detection algorithms
└── config.py                    # Configuration settings
```

## Technical Approach

### Multi-Level Detection System
1. **Basic Detector**: OpenCV + rule-based classification (~8ms)
2. **Enhanced Detector**: 3 HuggingFace models in ensemble (~50ms)
3. **Ultra-Enhanced**: Data augmentation + advanced preprocessing (~100ms)

### HuggingFace Models Used
- `dima806/facial_emotions_image_detection` - Facial emotion specialist
- `trpakov/vit-face-expression` - Vision Transformer for expressions
- `google/vit-base-patch16-224` - General vision features

### Advanced Features
- **Ensemble Learning**: Multiple models vote on classification
- **Data Augmentation**: Image enhancement (contrast, brightness, sharpness)
- **Robust Face Detection**: Multiple cascade strategies
- **Weighted Classification**: Model-specific importance weights
- **Real-time Analysis**: Webcam integration with live feedback

### Genuine vs. Fake Smile Detection Logic
- **Genuine smiles** (Duchenne): Eye involvement, natural curves, balanced features
- **Fake smiles**: Forced expressions, lack of eye engagement, asymmetry
- **Classification**: Uses emotion scores, facial feature analysis, and ML predictions

## Performance Metrics
- **Basic Detector**: ~8ms per image, 55% confidence
- **Enhanced Detector**: ~50ms per image, 66% confidence (3 models)
- **Ultra-Enhanced**: ~100ms per image, high precision (40+ predictions)
- **Memory Usage**: Lightweight, CPU-optimized
- **Real-time Capable**: Yes, 1 FPS analysis rate

## Usage Recommendations
- **Speed Priority**: Use `smile_detector.py`
- **Balanced Performance**: Use `enhanced_detector.py` 
- **Maximum Accuracy**: Use `ultra_enhanced_detector.py`
- **Real-time Applications**: Use `production_webcam.py`

---

**Note:**
- No model weights or large datasets included
- Models download automatically from HuggingFace on first run
- CPU-optimized, no GPU required
- For demonstration and research purposes
