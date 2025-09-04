import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
import torch

# Initialize models
def get_model_and_tokenizer(model_id="google/flan-t5-small"):
    """Get model and tokenizer with fallback to Flan-T5."""
    try:
        if "phi" in model_id.lower():
            # For MVP, skip Phi-3 as it's too resource intensive
            print("Phi-3 model skipped for MVP (too resource intensive), using Flan-T5")
            tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
            model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
            return model, tokenizer, "seq2seq"
        else:
            # Use Flan-T5 by default
            tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
            model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
            return model, tokenizer, "seq2seq"
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        # Final fallback - use simple rule-based generation
        return None, None, "rule_based"

def gen_copy(brief, tone="friendly", palette_hint=None, model_id="google/flan-t5-small"):
    """Generate marketing copy based on brief and tone."""
    try:
        model, tokenizer, model_type = get_model_and_tokenizer(model_id)
        
        # Handle rule-based fallback
        if model_type == "rule_based":
            return generate_rule_based_copy(brief, tone, palette_hint)
        
        # Create prompt based on model type
        if model_type == "causal":
            prompt = f"""<|system|>
You are a concise marketing writer.
<|user|>
Based on the brief below, write short copy suited for a flyer.
Constraints: headline ≤ 8 words, tagline ≤ 12 words. Bullets: 3 items, short. CTAs: 2 items, imperative.
Return JSON exactly:
{{"headline":"...", "tagline":"...", "bullets":["...","...","..."], "ctas":["...","..."], "palette":["#RRGGBB","#RRGGBB","#RRGGBB"]}}
Brief: {brief}
Audience/tone hint: {tone}
<|assistant|>"""
        else:
            # Flan-T5 prompt with enhanced instructions
            prompt = f"""Create compelling marketing copy for a professional flyer.

Business: {brief}
Tone: {tone}

Requirements:
- Headline: Maximum 8 words, attention-grabbing
- Tagline: Maximum 12 words, compelling and memorable  
- Bullet points: 3 key benefits, concise and impactful
- Call-to-action: 2 action-oriented phrases

Return JSON format:
{{"headline":"...", "tagline":"...", "bullets":["...","...","..."], "ctas":["...","..."], "palette":["#RRGGBB","#RRGGBB","#RRGGBB"]}}

Output:"""
        
        # Generate text
        if model_type == "causal":
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=220,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract the assistant response
            if "<|assistant|>" in generated_text:
                generated_text = generated_text.split("<|assistant|>")[-1].strip()
        else:
            # Flan-T5 generation
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=220,
                    temperature=0.3,
                    do_sample=True
                )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse JSON from generated text
        result = ensure_json(generated_text)
        
        # Ensure we have all required fields
        required_fields = ["headline", "tagline", "bullets", "ctas"]
        for field in required_fields:
            if field not in result:
                result[field] = get_default_value(field, brief)
        
        # Add palette if not present
        if "palette" not in result or not result["palette"]:
            if palette_hint and len(palette_hint) >= 3:
                result["palette"] = palette_hint[:3]
            else:
                result["palette"] = get_default_palette(tone)
        
        return result
        
    except Exception as e:
        print(f"Error in gen_copy: {e}")
        return get_fallback_copy(brief, tone, palette_hint)

def ensure_json(text):
    """Extract and parse JSON from text robustly."""
    try:
        # Try to find JSON block
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(json_pattern, text)
        
        for match in json_matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # If no JSON found, try to extract key-value pairs
        return extract_key_value_pairs(text)
        
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {}

def extract_key_value_pairs(text):
    """Extract key-value pairs from text when JSON parsing fails."""
    result = {}
    
    # Extract headline
    headline_match = re.search(r'headline["\']?\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
    if headline_match:
        result["headline"] = headline_match.group(1)
    
    # Extract tagline
    tagline_match = re.search(r'tagline["\']?\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
    if tagline_match:
        result["tagline"] = tagline_match.group(1)
    
    # Extract bullets
    bullets = re.findall(r'bullets?["\']?\s*:\s*\[([^\]]+)\]', text, re.IGNORECASE)
    if bullets:
        bullet_items = re.findall(r'["\']([^"\']+)["\']', bullets[0])
        result["bullets"] = bullet_items[:3]  # Limit to 3
    
    # Extract CTAs
    ctas = re.findall(r'ctas?["\']?\s*:\s*\[([^\]]+)\]', text, re.IGNORECASE)
    if ctas:
        cta_items = re.findall(r'["\']([^"\']+)["\']', ctas[0])
        result["ctas"] = cta_items[:2]  # Limit to 2
    
    return result

def get_default_value(field, brief):
    """Get default values for missing fields."""
    defaults = {
        "headline": "Amazing Business",
        "tagline": "Transform your experience today",
        "bullets": ["Quality service", "Expert team", "Great value"],
        "ctas": ["Get Started", "Learn More"]
    }
    return defaults.get(field, "")

def get_default_palette(tone):
    """Get enhanced default color palette based on tone."""
    palettes = {
        "friendly": ["#3B82F6", "#64748B", "#F59E0B"],  # Blue, Gray, Orange
        "professional": ["#1F2937", "#6B7280", "#3B82F6"],  # Dark, Gray, Blue
        "playful": ["#EC4899", "#8B5CF6", "#F59E0B"],  # Pink, Purple, Orange
        "bold": ["#DC2626", "#1F2937", "#F59E0B"]  # Red, Dark, Orange
    }
    return palettes.get(tone, ["#2563EB", "#64748B", "#F59E0B"])

def generate_rule_based_copy(brief, tone, palette_hint):
    """Generate copy using simple rule-based approach when models fail."""
    # Extract key words from brief
    words = brief.lower().split()
    key_words = [word for word in words if len(word) > 3 and word not in ['with', 'that', 'this', 'they', 'have', 'will', 'from']]
    
    # Generate headline
    if key_words:
        headline = f"Amazing {key_words[0].title()}"
    else:
        headline = "Professional Service"
    
    # Generate tagline based on tone
    taglines = {
        "friendly": "Transform your experience today",
        "professional": "Excellence in every detail",
        "playful": "Where magic happens",
        "bold": "Unleash your potential"
    }
    tagline = taglines.get(tone, "Quality solutions for your needs")
    
    # Generate bullets
    bullets = [
        "Expert team and service",
        "Quality guaranteed",
        "Great value for money"
    ]
    
    # Generate CTAs
    ctas = ["Get Started", "Learn More"]
    
    # Get palette
    palette = palette_hint[:3] if palette_hint else get_default_palette(tone)
    
    return {
        "headline": headline,
        "tagline": tagline,
        "bullets": bullets,
        "ctas": ctas,
        "palette": palette
    }

def get_fallback_copy(brief, tone, palette_hint):
    """Generate fallback copy when model fails."""
    return {
        "headline": get_default_value("headline", brief),
        "tagline": get_default_value("tagline", brief),
        "bullets": get_default_value("bullets", brief),
        "ctas": get_default_value("ctas", brief),
        "palette": palette_hint[:3] if palette_hint else get_default_palette(tone)
    }
