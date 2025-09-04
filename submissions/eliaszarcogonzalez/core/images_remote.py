import requests
import os
import base64
from PIL import Image
import io
import numpy as np

def generate_bg(prompt, out="bg.png"):
    """Generate background image using Hugging Face Inference API."""
    try:
        token = os.getenv("HF_TOKEN")
        if not token:
            print("HF_TOKEN not set, skipping remote image generation")
            return None
        
        # Try multiple models for better quality
        models = [
            "stabilityai/stable-diffusion-2-1",  # High quality, good for backgrounds
            "runwayml/stable-diffusion-v1-5",    # Good balance of quality/speed
            "stabilityai/sdxl-turbo"            # Fast fallback
        ]
        
        for model_id in models:
            try:
                url = f"https://api-inference.huggingface.co/models/{model_id}"
                headers = {"Authorization": f"Bearer {token}"}
                
                # Create a safe, professional prompt
                safe_prompt = f"Professional, minimalist, {prompt[:100]}, high quality, clean design"
                
                payload = {
                    "inputs": safe_prompt,
                    "parameters": {
                        "num_inference_steps": 20 if "turbo" not in model_id else 1,
                        "guidance_scale": 7.5 if "turbo" not in model_id else 1.0,
                        "width": 1024,
                        "height": 1024,
                        "negative_prompt": "text, watermark, blurry, low quality, distorted"
                    }
                }
                
                print(f"Trying model: {model_id}")
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                
                # Save the image
                with open(out, "wb") as f:
                    f.write(response.content)
                
                # Validate the image
                try:
                    img = Image.open(out)
                    img.verify()
                    print(f"✅ Successfully generated image with {model_id}")
                    return out
                except Exception as e:
                    print(f"Generated image is invalid: {e}")
                    os.remove(out)
                    continue
                    
            except requests.exceptions.RequestException as e:
                print(f"API request failed for {model_id}: {e}")
                continue
            except Exception as e:
                print(f"Error with {model_id}: {e}")
                continue
        
        print("All models failed, using fallback")
        return create_fallback_background(out)
            
    except Exception as e:
        print(f"Error generating background image: {e}")
        return create_fallback_background(out)

def generate_safe_background(prompt, out="bg.png"):
    """Generate a safe background with content filtering."""
    try:
        # Simple content filtering for MVP
        unsafe_words = ['nude', 'naked', 'violence', 'blood', 'gore', 'explicit']
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in unsafe_words):
            print("Prompt contains potentially unsafe content, using fallback")
            return create_fallback_background(out)
        
        return generate_bg(prompt, out)
        
    except Exception as e:
        print(f"Error in safe background generation: {e}")
        return create_fallback_background(out)

def create_fallback_background(out="bg.png"):
    """Create a simple gradient background as fallback."""
    try:
        from PIL import Image, ImageDraw
        
        # Create a simple gradient
        width, height = 1024, 1024
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)
        
        # Create a subtle gradient
        for y in range(height):
            ratio = y / height
            r = int(200 + 55 * ratio)
            g = int(200 + 55 * ratio)
            b = int(255)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        img.save(out, "PNG")
        return out
        
    except Exception as e:
        print(f"Error creating fallback background: {e}")
        return None

def validate_image_safety(image_path):
    """Basic image safety validation."""
    try:
        if not os.path.exists(image_path):
            return False
        
        img = Image.open(image_path)
        
        # Check image dimensions
        if img.width < 100 or img.height < 100:
            return False
        
        # Check if image is mostly transparent or black
        if img.mode == 'RGBA':
            # Convert to RGB for analysis
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[-1])
            img = rgb_img
        
        # Simple brightness check
        img_array = np.array(img)
        avg_brightness = np.mean(img_array)
        
        if avg_brightness < 10:  # Too dark
            return False
        
        return True
        
    except Exception as e:
        print(f"Error validating image safety: {e}")
        return False
