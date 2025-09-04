# 🎨 PromoKit Lite - AI Marketing Content Generator

## What it does

PromoKit Lite generates professional marketing flyers and promo videos from a prompt or website URL. It uses AI-powered copywriting and professional design templates to create:

- **Flyers** (PNG + PDF) with 4 professional templates
- **Promo videos** with captions and effects
- **AI copywriting** with multiple brand tones
- **Color palette extraction** from websites
- **Background image generation** (optional)

## How to run it

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
pip install -r requirements.txt
```

### Usage

**Web Interface (Recommended):**
```bash
python app.py
```
Open browser to `http://localhost:7860`

**Command Line:**
```bash
python scripts/quick_demo.py --prompt "Your business description" --tone friendly --template modern
```

### Environment Variables (Optional)
```bash
cp env.example .env
# Add HF_TOKEN for AI background images
```
