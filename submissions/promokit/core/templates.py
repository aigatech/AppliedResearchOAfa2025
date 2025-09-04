"""
Professional flyer templates for PromoKit Lite
Provides high-quality, pre-designed layouts with better visual appeal
"""

from PIL import Image, ImageDraw, ImageFont
import os
import math

# Template configurations
TEMPLATES = {
    "modern": {
        "name": "Modern Business",
        "description": "Clean, professional layout with geometric elements",
        "colors": {
            "primary": "#2563EB",
            "secondary": "#1E40AF", 
            "accent": "#F59E0B",
            "text": "#1F2937",
            "light_text": "#6B7280"
        },
        "layout": "centered",
        "elements": ["geometric", "gradient", "button"]
    },
    "bold": {
        "name": "Bold Impact",
        "description": "High-contrast design with strong typography",
        "colors": {
            "primary": "#DC2626",
            "secondary": "#991B1B",
            "accent": "#F59E0B", 
            "text": "#FFFFFF",
            "light_text": "#F3F4F6"
        },
        "layout": "asymmetric",
        "elements": ["diagonal", "overlay", "button"]
    },
    "elegant": {
        "name": "Elegant Premium",
        "description": "Sophisticated design with subtle details",
        "colors": {
            "primary": "#7C3AED",
            "secondary": "#5B21B6",
            "accent": "#F59E0B",
            "text": "#1F2937", 
            "light_text": "#6B7280"
        },
        "layout": "grid",
        "elements": ["border", "pattern", "button"]
    },
    "playful": {
        "name": "Playful Creative",
        "description": "Fun, colorful design with organic shapes",
        "colors": {
            "primary": "#EC4899",
            "secondary": "#BE185D",
            "accent": "#8B5CF6",
            "text": "#1F2937",
            "light_text": "#6B7280"
        },
        "layout": "organic",
        "elements": ["circles", "gradient", "button"]
    }
}

def get_template(template_name="modern"):
    """Get template configuration"""
    return TEMPLATES.get(template_name, TEMPLATES["modern"])

def create_modern_layout(draw, width, height, colors, headline, tagline, bullets, cta):
    """Create modern business layout"""
    # Background gradient
    for y in range(height):
        ratio = y / height
        r = int(37 + (107 - 37) * ratio)  # Blue gradient
        g = int(99 + (154 - 99) * ratio)
        b = int(235 + (235 - 235) * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    # Geometric elements
    draw.rectangle([width-80, 0, width, 80], fill=colors["accent"], outline=None)
    draw.rectangle([0, height-80, 80, height], fill=colors["accent"], outline=None)
    
    # Content area with subtle overlay
    content_width = width - 120
    content_x = 60
    content_y = 80
    
    # Headline
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 72)
    except:
        font = ImageFont.load_default()
    
    draw.text((content_x, content_y), headline, fill=colors["text"], font=font)
    
    # Tagline
    try:
        tagline_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 36)
    except:
        tagline_font = ImageFont.load_default()
    
    draw.text((content_x, content_y + 100), tagline, fill=colors["light_text"], font=tagline_font)
    
    # Bullets
    bullet_y = content_y + 180
    for bullet in bullets:
        draw.text((content_x, bullet_y), f"• {bullet}", fill=colors["text"], font=tagline_font)
        bullet_y += 50
    
    # CTA Button
    button_width = 200
    button_height = 60
    button_x = content_x
    button_y = bullet_y + 40
    
    # Button background with shadow
    draw.rectangle([button_x+2, button_y+2, button_x+button_width+2, button_y+button_height+2], 
                  fill="#000000", outline=None)
    draw.rectangle([button_x, button_y, button_x+button_width, button_y+button_height], 
                  fill=colors["accent"], outline=None)
    
    # Button text
    try:
        button_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except:
        button_font = ImageFont.load_default()
    
    text_bbox = draw.textbbox((0, 0), cta, font=button_font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = button_x + (button_width - text_width) // 2
    text_y = button_y + (button_height - 24) // 2
    
    draw.text((text_x, text_y), cta, fill=colors["text"], font=button_font)

def create_bold_layout(draw, width, height, colors, headline, tagline, bullets, cta):
    """Create bold impact layout"""
    # Dark background
    draw.rectangle([0, 0, width, height], fill=colors["primary"], outline=None)
    
    # Diagonal accent
    points = [(width, 0), (width-200, 0), (width, 200)]
    draw.polygon(points, fill=colors["secondary"], outline=None)
    
    # Content
    content_x = 60
    content_y = 80
    
    # Large headline
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 84)
    except:
        font = ImageFont.load_default()
    
    draw.text((content_x, content_y), headline, fill=colors["text"], font=font)
    
    # Tagline
    try:
        tagline_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 42)
    except:
        tagline_font = ImageFont.load_default()
    
    draw.text((content_x, content_y + 120), tagline, fill=colors["light_text"], font=tagline_font)
    
    # Bullets with accent
    bullet_y = content_y + 200
    for i, bullet in enumerate(bullets):
        # Bullet point with accent color
        draw.text((content_x, bullet_y), "▶", fill=colors["accent"], font=tagline_font)
        draw.text((content_x + 30, bullet_y), bullet, fill=colors["text"], font=tagline_font)
        bullet_y += 60
    
    # Bold CTA
    button_width = 250
    button_height = 70
    button_x = content_x
    button_y = bullet_y + 40
    
    draw.rectangle([button_x, button_y, button_x+button_width, button_y+button_height], 
                  fill=colors["accent"], outline=None)
    
    try:
        button_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 28)
    except:
        button_font = ImageFont.load_default()
    
    text_bbox = draw.textbbox((0, 0), cta, font=button_font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = button_x + (button_width - text_width) // 2
    text_y = button_y + (button_height - 28) // 2
    
    draw.text((text_x, text_y), cta, fill=colors["text"], font=button_font)

def create_elegant_layout(draw, width, height, colors, headline, tagline, bullets, cta):
    """Create elegant premium layout"""
    # Light background
    draw.rectangle([0, 0, width, height], fill="#F8FAFC", outline=None)
    
    # Border frame
    border_width = 8
    draw.rectangle([border_width, border_width, width-border_width, height-border_width], 
                  fill=None, outline=colors["primary"], width=border_width)
    
    # Header accent
    draw.rectangle([0, 0, width, 120], fill=colors["primary"], outline=None)
    
    # Content
    content_x = 80
    content_y = 160
    
    # Elegant headline
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 76)
    except:
        font = ImageFont.load_default()
    
    draw.text((content_x, content_y), headline, fill=colors["text"], font=font)
    
    # Subtle tagline
    try:
        tagline_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 38)
    except:
        tagline_font = ImageFont.load_default()
    
    draw.text((content_x, content_y + 100), tagline, fill=colors["light_text"], font=tagline_font)
    
    # Elegant bullets
    bullet_y = content_y + 180
    for bullet in bullets:
        # Elegant bullet point
        draw.ellipse([content_x, bullet_y+8, content_x+12, bullet_y+20], 
                     fill=colors["accent"], outline=None)
        draw.text((content_x + 25, bullet_y), bullet, fill=colors["text"], font=tagline_font)
        bullet_y += 55
    
    # Elegant CTA
    button_width = 220
    button_height = 65
    button_x = content_x
    button_y = bullet_y + 50
    
    # Button with border
    draw.rectangle([button_x, button_y, button_x+button_width, button_y+button_height], 
                  fill=colors["accent"], outline=colors["primary"], width=3)
    
    try:
        button_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 26)
    except:
        button_font = ImageFont.load_default()
    
    text_bbox = draw.textbbox((0, 0), cta, font=button_font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = button_x + (button_width - text_width) // 2
    text_y = button_y + (button_height - 26) // 2
    
    draw.text((text_x, text_y), cta, fill=colors["text"], font=button_font)

def create_playful_layout(draw, width, height, colors, headline, tagline, bullets, cta):
    """Create playful creative layout"""
    # Gradient background
    for y in range(height):
        ratio = y / height
        r = int(236 + (139 - 236) * ratio)  # Pink to purple
        g = int(72 + (92 - 72) * ratio)
        b = int(153 + (246 - 153) * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    # Organic circles
    draw.ellipse([width-100, 50, width-20, 130], fill=colors["accent"], outline=None)
    draw.ellipse([20, height-120, 100, height-40], fill=colors["accent"], outline=None)
    
    # Content
    content_x = 80
    content_y = 100
    
    # Fun headline
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 78)
    except:
        font = ImageFont.load_default()
    
    draw.text((content_x, content_y), headline, fill=colors["text"], font=font)
    
    # Playful tagline
    try:
        tagline_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 40)
    except:
        tagline_font = ImageFont.load_default()
    
    draw.text((content_x, content_y + 110), tagline, fill=colors["light_text"], font=tagline_font)
    
    # Fun bullets
    bullet_y = content_y + 190
    for i, bullet in enumerate(bullets):
        # Colorful bullet points
        bullet_colors = [colors["accent"], colors["secondary"], colors["primary"]]
        draw.ellipse([content_x, bullet_y+5, content_x+15, bullet_y+20], 
                     fill=bullet_colors[i % len(bullet_colors)], outline=None)
        draw.text((content_x + 25, bullet_y), bullet, fill=colors["text"], font=tagline_font)
        bullet_y += 60
    
    # Playful CTA
    button_width = 240
    button_height = 75
    button_x = content_x
    button_y = bullet_y + 40
    
    # Rounded button effect
    draw.ellipse([button_x, button_y, button_x+button_height, button_y+button_height], 
                 fill=colors["accent"], outline=None)
    draw.ellipse([button_x+button_width-button_height, button_y, button_x+button_width, button_y+button_height], 
                 fill=colors["accent"], outline=None)
    draw.rectangle([button_x+button_height//2, button_y, button_x+button_width-button_height//2, button_y+button_height], 
                  fill=colors["accent"], outline=None)
    
    try:
        button_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 30)
    except:
        button_font = ImageFont.load_default()
    
    text_bbox = draw.textbbox((0, 0), cta, font=button_font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = button_x + (button_width - text_width) // 2
    text_y = button_y + (button_height - 30) // 2
    
    draw.text((text_x, text_y), cta, fill=colors["text"], font=button_font)

def render_template_flyer(headline, tagline, bullets, cta, template_name="modern", out="flyer.png"):
    """Render a professional flyer using templates"""
    width, height = 800, 1200
    
    # Create image
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Get template
    template = get_template(template_name)
    colors = template["colors"]
    
    # Apply template layout
    if template_name == "modern":
        create_modern_layout(draw, width, height, colors, headline, tagline, bullets, cta)
    elif template_name == "bold":
        create_bold_layout(draw, width, height, colors, headline, tagline, bullets, cta)
    elif template_name == "elegant":
        create_elegant_layout(draw, width, height, colors, headline, tagline, bullets, cta)
    elif template_name == "playful":
        create_playful_layout(draw, width, height, colors, headline, tagline, bullets, cta)
    else:
        create_modern_layout(draw, width, height, colors, headline, tagline, bullets, cta)
    
    # Save with high quality
    img.save(out, "PNG", quality=95, optimize=True)
    return out

def get_available_templates():
    """Get list of available templates"""
    return list(TEMPLATES.keys())

def get_template_info(template_name):
    """Get template information"""
    template = get_template(template_name)
    return {
        "name": template["name"],
        "description": template["description"],
        "colors": template["colors"]
    }
