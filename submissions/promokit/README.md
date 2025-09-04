# 🎨 PromoKit Lite - AI Marketing Content Generator

Generate professional marketing flyers and promo videos from a prompt or website URL using AI-powered copywriting and design.

## ✨ Features

- **📄 Flyer Generation**: Create high-quality flyers in PNG and PDF formats
- **🎥 Video Creation**: Generate promo videos with captions and Ken Burns effects
- **🎨 Smart Color Palettes**: Automatic color extraction from websites or AI-generated palettes
- **🌐 Website Scraping**: Extract content and branding from any website
- **🤖 AI Copywriting**: Generate headlines, taglines, and CTAs using Phi-3 or Flan-T5
- **📊 Content Ranking**: Use semantic similarity to pick the best copy variants
- **🛡️ Safety Features**: Content filtering and URL validation
- **🎯 Multiple Tones**: Friendly, professional, playful, or bold brand voices

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd promokit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up environment variables:
```bash
cp env.example .env
# Edit .env and add your HF_TOKEN for AI background images
```

### Usage

#### Web Interface (Recommended)

Launch the Gradio web interface:
```bash
python app.py
```

Open your browser to `http://localhost:7860` and start generating content!

#### Command Line

Generate content from a prompt:
```bash
python scripts/quick_demo.py --prompt "Modern yoga studio in downtown Atlanta" --tone friendly
```

Generate content from a website:
```bash
python scripts/quick_demo.py --url "https://business.site" --tone professional --remote-image
```

## 📖 Documentation

### Core Modules

- **`core/branding.py`**: Website scraping, color extraction, and content summarization
- **`core/copywriter.py`**: AI-powered copy generation with Phi-3 and Flan-T5
- **`core/ranker.py`**: Semantic similarity ranking for copy variants
- **`core/flyer.py`**: PNG and PDF flyer rendering with accessibility features
- **`core/video.py`**: Video creation with slideshows and captions
- **`core/images_remote.py`**: AI background image generation via Hugging Face
- **`core/safety.py`**: Content safety and URL validation

### API Reference

#### Branding Module

```python
from core import branding

# Scrape website content
scraped = branding.scrape("https://example.com")

# Extract color palette
palette = branding.extract_palette(text, hex_colors)

# Summarize content
summary = branding.summarize("Business description")
```

#### Copywriter Module

```python
from core import copywriter

# Generate marketing copy
copy = copywriter.gen_copy(
    brief="Business description",
    tone="friendly",
    palette_hint=["#FF0000", "#00FF00", "#0000FF"]
)
```

#### Flyer Module

```python
from core import flyer

# Create flyer
png_path = flyer.render_png(
    headline="Amazing Service",
    tagline="Transform your experience",
    bullets=["Feature 1", "Feature 2", "Feature 3"],
    cta="Get Started",
    palette=["#FF0000", "#00FF00", "#0000FF"]
)

# Convert to PDF
pdf_path = flyer.render_pdf(png_path)
```

## 🎯 Use Cases

- **Small Businesses**: Create professional marketing materials quickly
- **Startups**: Generate content for pitch decks and social media
- **Agencies**: Rapidly prototype marketing campaigns
- **Content Creators**: Generate promotional content for social media
- **Event Organizers**: Create flyers and videos for events

## 🔧 Configuration

### Environment Variables

- `HF_TOKEN`: Hugging Face API token for AI background images
- `PROMOKIT_DEBUG`: Enable debug mode (true/false)

### Model Configuration

The system uses lightweight models optimized for CPU:

- **Copy Generation**: Microsoft Phi-3-mini-4k-instruct (fallback: Flan-T5-small)
- **Semantic Ranking**: all-MiniLM-L6-v2
- **Background Images**: SDXL Turbo (optional, via HF API)

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

Run specific tests:
```bash
python -m pytest tests/test_all.py::TestBranding
```

## 📁 Project Structure

```
promokit/
├── app.py                 # Gradio web interface
├── requirements.txt       # Python dependencies
├── env.example           # Environment variables template
├── README.md             # This file
├── TECHNICAL_PLAN.md     # Technical documentation
├── core/                 # Core modules
│   ├── __init__.py
│   ├── branding.py       # Website scraping & color extraction
│   ├── copywriter.py     # AI copy generation
│   ├── ranker.py         # Content ranking
│   ├── flyer.py          # Flyer rendering
│   ├── video.py          # Video creation
│   ├── images_remote.py  # AI background images
│   └── safety.py         # Content safety
├── assets/               # Static assets
│   ├── fonts/            # Font files
│   ├── icons/            # Icon files
│   └── templates/        # HTML/CSS templates
├── scripts/              # CLI tools
│   └── quick_demo.py     # Command-line interface
└── tests/                # Test suite
    └── test_all.py       # Comprehensive tests
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for model hosting and inference APIs
- Gradio for the web interface framework
- MoviePy for video processing
- Pillow for image manipulation
- WeasyPrint for PDF generation

## 🐛 Troubleshooting

### Common Issues

**"Error loading model"**: Ensure you have sufficient RAM and try using the Flan-T5 fallback.

**"PDF generation failed"**: Install WeasyPrint dependencies or use PNG-only mode.

**"Video creation failed"**: Ensure ffmpeg is installed on your system.

**"Background image generation failed"**: Check your HF_TOKEN or disable remote images.

### Performance Tips

- Use CPU-optimized models for better performance
- Disable remote image generation for faster processing
- Use shorter prompts for quicker generation
- Consider using the CLI for batch processing

## 📈 Roadmap

- [ ] Social media size templates (Instagram, Twitter, LinkedIn)
- [ ] A/B testing with multiple variants
- [ ] Logo upload and placement
- [ ] Advanced TTS integration
- [ ] Template library
- [ ] Analytics and metrics
- [ ] Multi-language support
