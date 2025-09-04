import re
import os

def is_safe(text):
    """Basic text safety check for MVP."""
    if not text:
        return True
    
    text_lower = text.lower()
    
    # Simple keyword-based filtering
    unsafe_patterns = [
        r'\b(hack|exploit|crack|steal|cheat)\b',
        r'\b(violence|kill|murder|attack)\b',
        r'\b(drugs?|alcohol|illegal)\b',
        r'\b(sex|porn|adult|explicit)\b',
        r'\b(racist|hate|discriminate)\b',
        r'\b(scam|fraud|phish)\b'
    ]
    
    for pattern in unsafe_patterns:
        if re.search(pattern, text_lower):
            return False
    
    # Check for excessive caps (potential spam)
    if len(re.findall(r'[A-Z]', text)) > len(text) * 0.7:
        return False
    
    # Check for excessive punctuation
    if len(re.findall(r'[!?]{3,}', text)) > 0:
        return False
    
    return True

def sanitize_text(text):
    """Sanitize text for safe display."""
    if not text:
        return ""
    
    # Remove potentially dangerous HTML/script tags
    text = re.sub(r'<script.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<.*?>', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Limit length
    if len(text) > 1000:
        text = text[:1000] + "..."
    
    return text.strip()

def validate_url(url):
    """Basic URL validation."""
    if not url:
        return False
    
    # Check for basic URL format
    url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    if not re.match(url_pattern, url):
        return False
    
    # Check for potentially dangerous protocols
    dangerous_protocols = ['file://', 'javascript:', 'data:']
    for protocol in dangerous_protocols:
        if url.lower().startswith(protocol):
            return False
    
    return True

def check_content_safety(content_dict):
    """Check safety of content dictionary."""
    if not content_dict:
        return True
    
    # Check each text field
    text_fields = ['headline', 'tagline', 'bullets', 'ctas']
    
    for field in text_fields:
        if field in content_dict:
            content = content_dict[field]
            if isinstance(content, list):
                for item in content:
                    if not is_safe(str(item)):
                        return False
            elif isinstance(content, str):
                if not is_safe(content):
                    return False
    
    return True

def get_safe_fallback_content():
    """Get safe fallback content when safety check fails."""
    return {
        "headline": "Professional Service",
        "tagline": "Quality solutions for your needs",
        "bullets": ["Expert team", "Quality service", "Great value"],
        "ctas": ["Contact Us", "Learn More"],
        "palette": ["#2563EB", "#64748B", "#F59E0B"]
    }
