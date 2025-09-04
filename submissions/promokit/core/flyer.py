from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import colorsys
import tempfile
from . import templates

# Try to import WeasyPrint, but provide fallback if not available
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError as e:
    print(f"WeasyPrint not available: {e}")
    WEASYPRINT_AVAILABLE = False
except OSError as e:
    print(f"WeasyPrint system dependencies missing: {e}")
    WEASYPRINT_AVAILABLE = False

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_contrast_ratio(color1, color2):
    """Calculate contrast ratio between two colors."""
    def get_luminance(color):
        # Convert to sRGB
        rgb = [c/255.0 for c in color]
        rgb = [((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92 for c in rgb]
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    
    lum1 = get_luminance(color1)
    lum2 = get_luminance(color2)
    
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    
    return (lighter + 0.05) / (darker + 0.05)

def get_readable_text_color(background_color):
    """Get a readable text color for a given background."""
    white = (255, 255, 255)
    black = (0, 0, 0)
    
    contrast_with_white = get_contrast_ratio(background_color, white)
    contrast_with_black = get_contrast_ratio(background_color, black)
    
    return white if contrast_with_white > contrast_with_black else black

def load_font(size, weight="normal"):
    """Load font with fallback to default system font."""
    try:
        # Try to load custom fonts
        font_paths = [
            "assets/fonts/Inter-Bold.ttf",
            "assets/fonts/Inter-Regular.ttf",
            "assets/fonts/OpenSans-Bold.ttf",
            "assets/fonts/OpenSans-Regular.ttf"
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
        
        # Fallback to default font
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()

def render_png(headline, tagline, bullets, cta, palette, bg_path=None, out="flyer.png", template_name="modern"):
    """Render flyer as PNG image with professional templates."""
    try:
        # Use professional templates for better quality
        if template_name and template_name in templates.get_available_templates():
            return templates.render_template_flyer(headline, tagline, bullets, cta, template_name, out)
        
        # Fallback to enhanced original method
        return render_enhanced_png(headline, tagline, bullets, cta, palette, bg_path, out)
        
    except Exception as e:
        print(f"Error rendering PNG: {e}")
        return create_fallback_flyer(out)

def render_enhanced_png(headline, tagline, bullets, cta, palette, bg_path=None, out="flyer.png"):
    """Original enhanced PNG rendering method."""
    try:
        W, H = 1080, 1350  # Instagram story dimensions
        
        # Create background
        if bg_path and os.path.exists(bg_path):
            # Load and resize background image
            bg_img = Image.open(bg_path).convert("RGB")
            bg_img = bg_img.resize((W, H), Image.Resampling.LANCZOS)
            
            # Add sophisticated overlay for readability
            overlay = Image.new("RGBA", (W, H), (0, 0, 0, 80))
            bg_img = Image.alpha_composite(bg_img.convert("RGBA"), overlay).convert("RGB")
        else:
            # Create enhanced gradient background using palette
            bg_img = create_enhanced_gradient_background(W, H, palette)
        
        # Create drawing context
        draw = ImageDraw.Draw(bg_img)
        
        # Parse colors
        primary_color = hex_to_rgb(palette[0])
        secondary_color = hex_to_rgb(palette[1])
        accent_color = hex_to_rgb(palette[2])
        
        # Get text colors with better contrast
        text_color = get_readable_text_color(primary_color)
        
        # Add subtle geometric elements for visual interest
        add_geometric_elements(draw, W, H, palette)
        
        # Layout parameters with better spacing
        padding = 100
        y_position = padding + 50
        
        # Render headline with enhanced styling
        headline_font = load_font(80, "bold")
        headline_bbox = draw.textbbox((0, 0), headline, font=headline_font)
        headline_width = headline_bbox[2] - headline_bbox[0]
        headline_x = (W - headline_width) // 2
        draw.text((headline_x, y_position), headline, font=headline_font, fill=text_color)
        y_position += 140
        
        # Render tagline with better typography
        tagline_font = load_font(42, "normal")
        tagline_bbox = draw.textbbox((0, 0), tagline, font=tagline_font)
        tagline_width = tagline_bbox[2] - tagline_bbox[0]
        tagline_x = (W - tagline_width) // 2
        draw.text((tagline_x, y_position), tagline, font=tagline_font, fill=text_color)
        y_position += 120
        
        # Render bullets with enhanced styling
        bullet_font = load_font(32, "normal")
        bullet_y_start = y_position
        for i, bullet in enumerate(bullets[:3]):  # Limit to 3 bullets
            bullet_text = f"• {bullet}"
            bullet_bbox = draw.textbbox((0, 0), bullet_text, font=bullet_font)
            bullet_width = bullet_bbox[2] - bullet_bbox[0]
            bullet_x = (W - bullet_width) // 2
            draw.text((bullet_x, bullet_y_start + i * 70), bullet_text, font=bullet_font, fill=text_color)
        
        # Render enhanced CTA button
        cta_y = H - 250
        cta_font = load_font(36, "bold")
        cta_text = cta if isinstance(cta, str) else cta[0] if cta else "Get Started"
        
        # Create enhanced button with gradient
        button_padding = 25
        cta_bbox = draw.textbbox((0, 0), cta_text, font=cta_font)
        button_width = cta_bbox[2] - cta_bbox[0] + 2 * button_padding
        button_height = cta_bbox[3] - cta_bbox[1] + 2 * button_padding
        button_x = (W - button_width) // 2
        button_y = cta_y - button_padding
        
        # Draw enhanced button with shadow
        draw_enhanced_button(draw, button_x, button_y, button_width, button_height, accent_color, text_color)
        
        # Draw CTA text
        cta_text_x = button_x + button_padding
        cta_text_y = button_y + button_padding
        draw.text((cta_text_x, cta_text_y), cta_text, font=cta_font, fill=text_color)
        
        # Save image with high quality
        bg_img.save(out, "PNG", quality=95, optimize=True)
        return out
        
    except Exception as e:
        print(f"Error rendering PNG: {e}")
        return create_fallback_flyer(out)

def create_enhanced_gradient_background(width, height, palette):
    """Create an enhanced gradient background with multiple color stops."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    
    # Create a more sophisticated gradient
    primary = hex_to_rgb(palette[0])
    secondary = hex_to_rgb(palette[1])
    accent = hex_to_rgb(palette[2])
    
    # Create diagonal gradient with multiple color stops
    for y in range(height):
        ratio = y / height
        
        # Use cubic interpolation for smoother gradients
        if ratio < 0.5:
            # First half: primary to secondary
            local_ratio = ratio * 2
            r = int(primary[0] * (1 - local_ratio) + secondary[0] * local_ratio)
            g = int(primary[1] * (1 - local_ratio) + secondary[1] * local_ratio)
            b = int(primary[2] * (1 - local_ratio) + secondary[2] * local_ratio)
        else:
            # Second half: secondary to accent
            local_ratio = (ratio - 0.5) * 2
            r = int(secondary[0] * (1 - local_ratio) + accent[0] * local_ratio)
            g = int(secondary[1] * (1 - local_ratio) + accent[1] * local_ratio)
            b = int(secondary[2] * (1 - local_ratio) + accent[2] * local_ratio)
        
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    return img

def add_geometric_elements(draw, width, height, palette):
    """Add subtle geometric elements for visual interest."""
    try:
        # Add subtle circles in corners
        circle_radius = 50
        circle_color = hex_to_rgb(palette[2])
        
        # Top-left corner
        draw.ellipse([20, 20, 20 + circle_radius*2, 20 + circle_radius*2], 
                    fill=circle_color, outline=None, width=0)
        
        # Bottom-right corner
        draw.ellipse([width - 20 - circle_radius*2, height - 20 - circle_radius*2, 
                     width - 20, height - 20], 
                    fill=circle_color, outline=None, width=0)
        
        # Add subtle lines
        line_color = hex_to_rgb(palette[1])
        for i in range(3):
            y = 200 + i * 300
            draw.line([(50, y), (width - 50, y)], fill=line_color, width=1)
            
    except Exception as e:
        print(f"Error adding geometric elements: {e}")

def draw_enhanced_button(draw, x, y, width, height, fill_color, text_color):
    """Draw an enhanced button with shadow and rounded corners."""
    try:
        # Draw shadow
        shadow_offset = 3
        shadow_color = (0, 0, 0, 100)
        draw.rectangle([x + shadow_offset, y + shadow_offset, 
                       x + width + shadow_offset, y + height + shadow_offset], 
                      fill=shadow_color, outline=None, width=0)
        
        # Draw main button with rounded corners effect
        draw.rectangle([x, y, x + width, y + height], 
                      fill=fill_color, outline=text_color, width=2)
        
        # Add subtle highlight
        highlight_color = tuple(min(255, c + 30) for c in fill_color)
        draw.rectangle([x + 2, y + 2, x + width - 2, y + height//3], 
                      fill=highlight_color, outline=None, width=0)
        
    except Exception as e:
        print(f"Error drawing enhanced button: {e}")
        # Fallback to simple button
        draw.rectangle([x, y, x + width, y + height], 
                      fill=fill_color, outline=text_color, width=2)

def render_pdf(png_path, out="flyer.pdf"):
    """Convert PNG to PDF using WeasyPrint."""
    if not WEASYPRINT_AVAILABLE:
        print("WeasyPrint not available, returning PNG path instead")
        return png_path
    
    try:
        if not os.path.exists(png_path):
            print(f"PNG file not found: {png_path}")
            return None
        
        # Create HTML template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ margin: 0; padding: 0; }}
                img {{ width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <img src="{png_path}" alt="Flyer">
        </body>
        </html>
        """
        
        # Convert to PDF
        HTML(string=html_content).write_pdf(out)
        return out
        
    except Exception as e:
        print(f"Error rendering PDF: {e}")
        # Fallback: return PNG path if PDF creation fails
        return png_path

def create_fallback_flyer(out_path):
    """Create a simple fallback flyer if rendering fails."""
    try:
        img = Image.new("RGB", (1080, 1350), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Simple text
        font = load_font(48)
        text = "Flyer Generation Failed"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (1080 - text_width) // 2
        text_y = 675  # Center vertically
        
        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))
        
        img.save(out_path, "PNG")
        return out_path
        
    except Exception as e:
        print(f"Error creating fallback flyer: {e}")
        return None
