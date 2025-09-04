import pytest
import os
import tempfile
from core import branding, copywriter, ranker, flyer, video, safety

class TestBranding:
    def test_scrape_valid_url(self):
        """Test scraping a valid URL."""
        # This would need a real URL for testing
        # For now, test with a mock
        result = branding.scrape("https://example.com")
        assert isinstance(result, dict)
        assert "text" in result
        assert "candidate_hex_colors" in result
    
    def test_extract_palette(self):
        """Test palette extraction."""
        text = "This is a test with colors #FF0000 #00FF00 #0000FF"
        colors = ["#FF0000", "#00FF00", "#0000FF"]
        palette = branding.extract_palette(text, colors)
        assert len(palette) == 3
        assert all(color.startswith('#') for color in palette)
    
    def test_summarize(self):
        """Test text summarization."""
        text = "This is a professional business that provides excellent services to customers."
        result = branding.summarize(text)
        assert "summary" in result
        assert "tone" in result
        assert "keywords" in result
        assert result["tone"] in ["friendly", "professional", "playful", "bold"]

class TestCopywriter:
    def test_gen_copy(self):
        """Test copy generation."""
        brief = "A modern coffee shop in downtown"
        result = copywriter.gen_copy(brief, "friendly")
        assert "headline" in result
        assert "tagline" in result
        assert "bullets" in result
        assert "ctas" in result
        assert len(result["bullets"]) <= 3
        assert len(result["ctas"]) <= 2
    
    def test_ensure_json(self):
        """Test JSON extraction."""
        text = '{"headline": "Test", "tagline": "Test tagline"}'
        result = copywriter.ensure_json(text)
        assert isinstance(result, dict)
        assert "headline" in result

class TestRanker:
    def test_pick_best_headline(self):
        """Test headline ranking."""
        headlines = ["Get started today", "Sign up now", "Join our service"]
        result = ranker.pick_best_headline(headlines, "increase signups")
        assert result in headlines
    
    def test_score_variants(self):
        """Test variant scoring."""
        variants = ["Option A", "Option B", "Option C"]
        result = ranker.score_variants(variants, "test goal")
        assert len(result) == len(variants)
        assert all(isinstance(item, tuple) for item in result)

class TestFlyer:
    def test_render_png(self):
        """Test PNG rendering."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            result = flyer.render_png(
                "Test Headline",
                "Test Tagline",
                ["Bullet 1", "Bullet 2", "Bullet 3"],
                "Get Started",
                ["#FF0000", "#00FF00", "#0000FF"],
                out=tmp.name
            )
            assert result == tmp.name
            assert os.path.exists(result)
            os.unlink(result)
    
    def test_hex_to_rgb(self):
        """Test hex to RGB conversion."""
        rgb = flyer.hex_to_rgb("#FF0000")
        assert rgb == (255, 0, 0)

class TestVideo:
    def test_create_simple_video(self):
        """Test simple video creation."""
        # Create a test PNG first
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_png:
            flyer.render_png(
                "Test",
                "Test",
                ["Test"],
                "Test",
                ["#FF0000", "#00FF00", "#0000FF"],
                out=tmp_png.name
            )
            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_mp4:
                result = video.create_simple_video_with_captions(
                    tmp_png.name,
                    ["Test Caption"],
                    out=tmp_mp4.name
                )
                if result and os.path.exists(result):
                    os.unlink(result)
                os.unlink(tmp_png.name)

class TestSafety:
    def test_is_safe(self):
        """Test safety checks."""
        assert safety.is_safe("This is a safe text")
        assert not safety.is_safe("This contains VIOLENCE")
        assert not safety.is_safe("SCAM ALERT!!!")
    
    def test_validate_url(self):
        """Test URL validation."""
        assert safety.validate_url("https://example.com")
        assert not safety.validate_url("not-a-url")
        assert not safety.validate_url("javascript:alert('xss')")
    
    def test_sanitize_text(self):
        """Test text sanitization."""
        text = "<script>alert('xss')</script>Hello World"
        sanitized = safety.sanitize_text(text)
        assert "<script>" not in sanitized
        assert "Hello World" in sanitized

if __name__ == "__main__":
    pytest.main([__file__])
