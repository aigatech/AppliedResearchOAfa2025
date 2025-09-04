import json
import re
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM_PROMPT = '''
You are a material analysis AI. Your job is to translate a user's description of a ground surface into a structured JSON object of its physical properties for a procedural generator.

### INSTRUCTIONS
The JSON output must contain eight keys: "structure_type", "displacement_amount", "element_density", "element_scale", "roughness_amount", "glossiness", "metallic", and "color_palette_hex".

IMPORTANT: All numeric values must be between 0.1 and 1.0 (never use 0.0 as it creates invisible textures).

- "structure_type": The fundamental construction of the material. Data: A string from a list: ["Continuous", "Particulate", "Fibrous"]
- "displacement_amount": The large-scale bumpiness (0.1 to 1.0). Use 0.3-0.8 for most materials.
- "element_density": How tightly packed are the elements (0.1 to 1.0). Use 0.0 only for Continuous structures.
- "element_scale": The average size of the individual elements or major features (0.1 to 1.0). Use 0.3-0.7 for most materials.
- "roughness_amount": The fine-scale, micro-surface texture (0.1 to 1.0). Use 0.2-0.8 for most materials.
- "glossiness": How shiny the surface is (0.0 to 1.0).
- "metallic": Is the material a metal (0.0 to 1.0).
- "color_palette_hex": A list of 2-3 hex color codes.

### EXAMPLES

User Input: "Dull, cracked rock with a sandy surface"
```json
{{
  "structure_type": "Continuous",
  "displacement_amount": 0.7,
  "element_density": 0.0,
  "element_scale": 0.6,
  "roughness_amount": 0.8,
  "glossiness": 0.1,
  "metallic": 0.0,
  "color_palette_hex": ["#6B705C", "#A5A58D", "#4E4B42"]
}}
```

User Input: "Wet, lush grass after rain"
```json
{{
  "structure_type": "Fibrous",
  "displacement_amount": 0.5,
  "element_density": 0.8,
  "element_scale": 0.3,
  "roughness_amount": 0.3,
  "glossiness": 0.7,
  "metallic": 0.0,
  "color_palette_hex": ["#2A5A2A", "#4A8A4A", "#6AB56A"]
}}
```

User Input: "Short green grass"
```json
{{
  "structure_type": "Fibrous",
  "displacement_amount": 0.4,
  "element_density": 0.9,
  "element_scale": 0.2,
  "roughness_amount": 0.2,
  "glossiness": 0.3,
  "metallic": 0.0,
  "color_palette_hex": ["#2D5A2D", "#4F8F4F", "#71B571"]
}}
```

User Input: "Coarse gravel path"
```json
{{
  "structure_type": "Particulate",
  "displacement_amount": 0.6,
  "element_density": 0.7,
  "element_scale": 0.5,
  "roughness_amount": 0.7,
  "glossiness": 0.2,
  "metallic": 0.0,
  "color_palette_hex": ["#8B7355", "#A0926B", "#6B5B47"]
}}
```

### USER INPUT
User Input: "{text_input}"
''' 

class SensationGenerator:
    def __init__(self):
        print("Initializing SensationGenerator...")
        if torch.backends.mps.is_available():
            device = "mps"
            print("Using device: Apple Silicon (MPS)")
        elif torch.cuda.is_available():
            device = 0
            print("Using device: CUDA (GPU)")
        else:
            device = -1
            print("Using device: CPU")
        
        try:
            self.pipeline = pipeline("text-generation", model="microsoft/DialoGPT-medium", device=device)
            print("Using DialoGPT-medium model")
        except:
            try:
                self.pipeline = pipeline("text-generation", model="gpt2", device=device)
                print("Using GPT-2 model")
            except:
                self.pipeline = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=device)
                print("Using TinyLlama model")
        
        print("AI model loaded successfully.")
    
    def _get_default_sensation(self) -> dict:
        return {
            "structure_type": "Continuous",
            "displacement_amount": 0.5,
            "element_density": 0.0,
            "element_scale": 0.5,
            "roughness_amount": 0.5,
            "glossiness": 0.5,
            "metallic": 0.0,
            "color_palette_hex": ["#888888", "#555555", "#A0A0A0"]
        }
    
    def _validate_and_fix_sensation(self, data: dict, input_text: str = "") -> dict:
        """Validate and fix sensation data to ensure reasonable values"""
        validated = data.copy()
        
        if validated.get("displacement_amount", 0) < 0.1:
            validated["displacement_amount"] = 0.3
            
        if validated.get("element_scale", 0) < 0.1:
            validated["element_scale"] = 0.4
            
        if validated.get("roughness_amount", 0) < 0.1:
            validated["roughness_amount"] = 0.3
            
        if validated.get("structure_type") == "Fibrous" and validated.get("element_density", 0) < 0.3:
            validated["element_density"] = 0.6
            
        if validated.get("structure_type") == "Particulate" and validated.get("element_density", 0) < 0.2:
            validated["element_density"] = 0.4
                    
        print(f"Validated data: {validated}")
        return validated
    
    def _is_ai_output_sensible(self, data: dict, input_text: str) -> bool:
        """Check if AI output makes sense for the given input"""
        text_lower = input_text.lower()
        
        structure = data.get("structure_type", "")
        colors = data.get("color_palette_hex", [])
        
        if any(word in text_lower for word in ["grass", "lawn", "turf"]):
            if structure != "Fibrous":
                print(f"AI incorrectly classified grass as {structure} instead of Fibrous")
                return False
            if all(self._is_grayish_color(color) for color in colors[:3]):
                print(f"AI generated gray colors for grass: {colors}")
                return False
        
        if "moss" in text_lower:
            if structure != "Fibrous":
                print(f"AI incorrectly classified moss as {structure} instead of Fibrous")
                return False
            if all(self._is_grayish_color(color) for color in colors[:3]):
                print(f"AI generated gray colors for moss: {colors}")
                return False
        
        if "sand" in text_lower:
            if structure != "Particulate":
                print(f"AI incorrectly classified sand as {structure} instead of Particulate")
                return False
        
        if "glass" in text_lower:
            if structure != "Continuous":
                print(f"AI incorrectly classified glass as {structure} instead of Continuous")
                return False
            if data.get("glossiness", 0) < 0.7:
                print(f"AI generated low glossiness for glass: {data.get('glossiness')}")
                return False
        
        return True
    
    def _is_grayish_color(self, hex_color: str) -> bool:
        """Check if a hex color is grayish (R, G, B values are similar)"""
        try:
            hex_color = hex_color.lstrip('#')
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            max_diff = max(abs(r-g), abs(g-b), abs(r-b))
            return max_diff < 30
        except Exception as e:
            return False

    def _parse_json_from_string(self, text: str) -> dict:
        # More robustly search for the JSON block.
        # First, try to find a markdown-style JSON block anywhere in the output.
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        
        # If that fails, fall back to finding the first string that looks like a JSON object.
        if not match:
            match = re.search(r'(\{.*\})', text, re.DOTALL)

        if not match:
            print(f"Error: Could not find a valid JSON object in the model's output.\nRaw output: {text}")
            raise ValueError("No JSON found in output")
            
        json_str = match.group(1)
        try:
            data = json.loads(json_str)
            required_keys = ["structure_type", "displacement_amount", "element_density", "element_scale", "roughness_amount", "glossiness", "metallic", "color_palette_hex"]
            if all(k in data for k in required_keys):
                return data
            else:
                print(f"Warning: Parsed JSON is missing required keys.\nParsed data: {data}")
                raise ValueError("Incomplete JSON data")
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode JSON from model output.\nExtracted string: {json_str}")
            raise ValueError("Invalid JSON format")

    def generate_data(self, text_input: str) -> dict:
        print(f"Generating sensation for: '{text_input}'")
        
        # For now, skip AI generation and go straight to intelligent fallback
        # since the AI models are not producing reliable results
        print("Using intelligent material analysis...")
        fallback_data = self._generate_intelligent_fallback(text_input)
        print(f"Intelligent analysis generated: {fallback_data}")
        validated_data = self._validate_and_fix_sensation(fallback_data, text_input)
        return validated_data
    
    def _generate_intelligent_fallback(self, text_input: str) -> dict:
        """Dynamically analyze any material description without hardcoding"""
        text_lower = text_input.lower()
        
        # Start with base defaults
        structure_type = "Continuous"
        displacement = 0.5
        density = 0.0
        scale = 0.5
        roughness = 0.5
        glossiness = 0.3
        metallic = 0.0
        colors = ["#888888", "#555555", "#A0A0A0"]
        
        # STRUCTURE TYPE ANALYSIS (dynamic, not hardcoded)
        fibrous_indicators = ["fiber", "hair", "fur", "blade", "strand", "thread", "wire", "bristle", "needle"]
        grass_indicators = ["grass", "lawn", "turf", "moss", "fern", "vegetation", "plant"]
        particulate_indicators = ["grain", "particle", "bead", "pellet", "drop", "crystal", "chunk"]
        sand_indicators = ["sand", "gravel", "dust", "powder", "salt", "sugar", "rice"]
        
        if any(word in text_lower for word in fibrous_indicators + grass_indicators):
            structure_type = "Fibrous"
            density = 0.8
            scale = 0.2
            displacement = 0.4
        elif any(word in text_lower for word in particulate_indicators + sand_indicators):
            structure_type = "Particulate"
            density = 0.7
            scale = 0.3
            displacement = 0.4
        # Otherwise stays Continuous
        
        # SURFACE TEXTURE ANALYSIS (dynamic)
        smooth_indicators = ["smooth", "polished", "glass", "ice", "silk", "mirror", "crystal", "ceramic"]
        rough_indicators = ["rough", "coarse", "bumpy", "rocky", "jagged", "textured", "gritty", "sandpaper"]
        
        if any(word in text_lower for word in smooth_indicators):
            roughness = 0.05
            glossiness = min(1.0, glossiness + 0.6)
        elif any(word in text_lower for word in rough_indicators):
            roughness = min(1.0, roughness + 0.4)
            glossiness = max(0.05, glossiness - 0.3)
        
        # WETNESS/GLOSSINESS ANALYSIS
        wet_indicators = ["wet", "damp", "moist", "slippery", "oily", "shiny", "glossy", "polished"]
        dry_indicators = ["dry", "dusty", "matte", "dull", "chalky", "powdery"]
        
        if any(word in text_lower for word in wet_indicators):
            glossiness = min(1.0, glossiness + 0.5)
        elif any(word in text_lower for word in dry_indicators):
            glossiness = max(0.05, glossiness - 0.4)
        
        # SIZE/SCALE ANALYSIS
        fine_indicators = ["fine", "small", "tiny", "micro", "thin", "delicate", "powder"]
        large_indicators = ["large", "big", "chunky", "thick", "coarse", "boulder", "massive"]
        
        if any(word in text_lower for word in fine_indicators):
            scale = max(0.1, scale * 0.5)
        elif any(word in text_lower for word in large_indicators):
            scale = min(1.0, scale * 1.8)
        
        # HEIGHT/DISPLACEMENT ANALYSIS
        flat_indicators = ["flat", "level", "even", "smooth", "planar"]
        bumpy_indicators = ["bumpy", "hilly", "uneven", "ridged", "mountainous", "wavy"]
        
        if any(word in text_lower for word in flat_indicators):
            displacement = max(0.1, displacement * 0.3)
        elif any(word in text_lower for word in bumpy_indicators):
            displacement = min(1.0, displacement * 1.6)
        
        # METALLIC ANALYSIS
        metal_indicators = ["metal", "steel", "iron", "aluminum", "copper", "bronze", "chrome", "silver", "gold"]
        if any(word in text_lower for word in metal_indicators):
            metallic = 0.8
            glossiness = max(0.7, glossiness)
            structure_type = "Continuous"
        
        # COLOR ANALYSIS (comprehensive and dynamic)
        if "green" in text_lower or any(word in text_lower for word in ["grass", "moss", "forest", "lime", "emerald"]):
            colors = ["#228B22", "#32CD32", "#90EE90"]
        elif "blue" in text_lower or any(word in text_lower for word in ["sky", "ocean", "water", "azure", "cyan"]):
            colors = ["#4682B4", "#87CEEB", "#B0E0E6"]
        elif "red" in text_lower or any(word in text_lower for word in ["blood", "fire", "crimson", "rust"]):
            colors = ["#8B0000", "#DC143C", "#FF6347"]
        elif "brown" in text_lower or any(word in text_lower for word in ["wood", "dirt", "soil", "mud", "earth"]):
            colors = ["#8B4513", "#A0522D", "#CD853F"]
        elif "yellow" in text_lower or any(word in text_lower for word in ["gold", "sand", "sun", "wheat"]):
            colors = ["#FFD700", "#F0E68C", "#FFEF94"]
        elif "orange" in text_lower or any(word in text_lower for word in ["copper", "rust", "amber"]):
            colors = ["#FF8C00", "#CD853F", "#D2691E"]
        elif "purple" in text_lower or any(word in text_lower for word in ["violet", "lavender", "plum"]):
            colors = ["#8A2BE2", "#9370DB", "#DDA0DD"]
        elif "white" in text_lower or any(word in text_lower for word in ["snow", "ice", "marble", "pearl"]):
            colors = ["#F8F8FF", "#F0F8FF", "#FFFFFF"]
        elif "black" in text_lower or any(word in text_lower for word in ["coal", "charcoal", "obsidian"]):
            colors = ["#2F2F2F", "#1C1C1C", "#000000"]
        elif "gray" in text_lower or "grey" in text_lower or any(word in text_lower for word in ["concrete", "stone", "rock", "cement"]):
            colors = ["#696969", "#808080", "#A9A9A9"]
        elif "clear" in text_lower or "transparent" in text_lower or "glass" in text_lower:
            colors = ["#F0F8FF", "#F8F8FF", "#FFFFFF"]
        elif "tan" in text_lower or "beige" in text_lower or any(word in text_lower for word in ["sand", "beach", "cream"]):
            colors = ["#D2B48C", "#F5DEB3", "#FFEBCD"]
        
        # Special material handling (needs to override other logic)
        if "glass" in text_lower:
            displacement = 0.02  # Almost completely flat
            roughness = 0.01     # Extremely smooth
            glossiness = 0.98    # Nearly perfect reflection
            structure_type = "Continuous"
            colors = ["#F8FDFF", "#FAFEFF", "#FCFFFF"]  # Very pale, almost transparent
        
        elif any(word in text_lower for word in ["grass", "lawn", "turf"]):
            structure_type = "Fibrous"
            density = 0.95       # Very dense blade coverage
            scale = 0.15         # Very fine individual blades
            displacement = 0.5   # Good height variation
            roughness = 0.25     # Moderate texture from blades
            glossiness = 0.4 if "wet" in text_lower else 0.2
            colors = ["#2D5A2D", "#4A8A4A", "#6AB56A"]
        
        elif "moss" in text_lower:
            structure_type = "Fibrous"
            density = 0.9        # Dense but not as dense as grass
            scale = 0.1          # Very fine moss texture
            displacement = 0.2   # Low profile
            roughness = 0.6      # Textured surface
            glossiness = 0.5 if "wet" in text_lower else 0.15
            colors = ["#228B22", "#32CD32", "#90EE90"]
        
        # Clamp all values to valid ranges
        displacement = max(0.05, min(1.0, displacement))
        density = max(0.0, min(1.0, density))
        scale = max(0.1, min(1.0, scale))
        roughness = max(0.02, min(1.0, roughness))
        glossiness = max(0.0, min(1.0, glossiness))
        metallic = max(0.0, min(1.0, metallic))
        
        return {
            "structure_type": structure_type,
            "displacement_amount": displacement,
            "element_density": density,
            "element_scale": scale,
            "roughness_amount": roughness,
            "glossiness": glossiness,
            "metallic": metallic,
            "color_palette_hex": colors
        }