import requests
import re
import json
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from readability import Document
import boilerpy3
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

def scrape(url):
    """Scrape website content and extract text, favicon, and candidate colors."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract main content using readability
        try:
            doc = Document(response.content)
            main_content = doc.summary()
        except Exception as e:
            print(f"error getting summary: {e}")
            main_content = ""
        
        # Fallback to boilerpy3 if readability fails
        if len(main_content) < 100:
            try:
                extractor = boilerpy3.LargestContentExtractor()
                doc = extractor.get_doc(response.content)
                main_content = doc.content
            except Exception as e:
                print(f"boilerpy3 fallback failed: {e}")
                main_content = ""
        
        # Extract favicon
        favicon_url = None
        favicon_link = soup.find('link', rel=lambda x: x and 'icon' in x.lower())
        if favicon_link:
            favicon_url = urljoin(url, favicon_link.get('href'))
        
        # Extract candidate hex colors
        candidate_hex_colors = []
        hex_pattern = r'#[0-9a-fA-F]{6}'
        hex_colors = re.findall(hex_pattern, response.text)
        
        # Count frequency and get top colors
        color_counts = {}
        for color in hex_colors:
            color_counts[color] = color_counts.get(color, 0) + 1
        
        # Get top 5 most frequent colors
        candidate_hex_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        candidate_hex_colors = [color for color, count in candidate_hex_colors]
        
        return {
            "text": main_content,
            "favicon_url": favicon_url,
            "candidate_hex_colors": candidate_hex_colors
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return {
            "text": "",
            "favicon_url": None,
            "candidate_hex_colors": []
        }

def extract_palette(text, candidate_hex_colors):
    """Extract a 3-color palette from text or candidate colors."""
    if candidate_hex_colors and len(candidate_hex_colors) >= 3:
        # Use the most frequent colors from the page
        return candidate_hex_colors[:3]
    
    # Fallback to LLM-generated palette
    return generate_palette_from_text(text)

def generate_palette_from_text(text):
    """Generate a palette using LLM based on text content."""
    try:
        # Use a simple approach for MVP - return a safe default palette
        # In a full implementation, you'd use an LLM here
        return ["#2563EB", "#64748B", "#F59E0B"]  # Blue, Gray, Orange
    except Exception:
        return ["#111111", "#444444", "#E8E8E8"]  # Safe fallback

def summarize(text_or_prompt):
    """Summarize text and extract tone and keywords."""
    try:
        # For MVP, use a simple approach
        # In production, you'd use an LLM here
        if len(text_or_prompt) > 200:
            summary = text_or_prompt[:200] + "..."
        else:
            summary = text_or_prompt
        
        # Simple tone detection
        text_lower = text_or_prompt.lower()
        if any(word in text_lower for word in ['fun', 'playful', 'creative', 'exciting']):
            tone = "playful"
        elif any(word in text_lower for word in ['professional', 'business', 'corporate', 'serious']):
            tone = "professional"
        elif any(word in text_lower for word in ['bold', 'strong', 'powerful', 'confident']):
            tone = "bold"
        else:
            tone = "friendly"
        
        # Extract keywords (simple approach)
        words = re.findall(r'\b\w+\b', text_or_prompt.lower())
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        keywords = [word for word in words if word not in common_words and len(word) > 3][:5]
        
        return {
            "summary": summary,
            "tone": tone,
            "keywords": keywords
        }
    except Exception as e:
        print(f"Error in summarize: {e}")
        return {
            "summary": text_or_prompt[:100] if text_or_prompt else "",
            "tone": "friendly",
            "keywords": []
        }
