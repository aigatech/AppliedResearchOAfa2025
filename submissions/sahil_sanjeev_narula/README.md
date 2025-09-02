# AI Personality Quiz
A Streamlit-based personality assessment tool that combines traditional Big Five psychology with HuggingFace AI models for enhanced personality analysis.

## What It Does

This application provides a comprehensive personality assessment through:
- **10 carefully crafted questions** covering all Big Five personality dimensions
- **Real-time scoring** for Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism
- **AI-powered analysis** using HuggingFace's emotion classification model
- **Top trait identification** with visual metrics and rankings
- **Detailed explanations** of what each personality trait means
- **Personalized results** with examples and interpretations
- **Interactive interface** with progress tracking and navigation

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Environment
```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit transformers torch
```

### 4. Run the Application
```bash
streamlit run simple_personality_quiz.py
```

### 5. Open Your Browser
Navigate to `http://localhost:8501` to start the personality quiz