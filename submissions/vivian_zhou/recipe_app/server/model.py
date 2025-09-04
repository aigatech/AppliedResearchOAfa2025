import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class RecipeGenerator:
    def __init__(self, model_name="google/flan-t5-base"):
        """Initialize the recipe generator with a pre-trained LLM."""
        self.model_name = model_name
        
        print(f"Loading model {model_name}...")
        
        # Create a text generation pipeline
        self.generator = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            device=-1,  # Use CPU
            torch_dtype=torch.float32
        )
        
        print(f"Model {model_name} loaded successfully!")
    
    def generate_recipe(self, pantry, diet="none", time=30):
        """
        Generate a recipe based on pantry ingredients, diet, and time constraints.
        
        Args:
            pantry (str): Comma-separated list of available ingredients
            diet (str): Dietary restrictions (e.g., "vegetarian", "vegan", "none")
            time (int): Available cooking time in minutes
            
        Returns:
            dict: Generated recipe with Name, RecipeIngredientParts, and RecipeInstructions
        """
        # Parse pantry ingredients
        ingredients = [ing.strip() for ing in pantry.split(',')]
        
        # Create a simple, direct prompt
        prompt = f"""Create a recipe using: {', '.join(ingredients)}

Recipe name: Pasta with Tomato and Herbs
Ingredients: 
- 200g pasta
- 2 tomatoes, diced
- 2 cloves garlic, minced
- 1 onion, chopped
- Fresh basil leaves
- 2 tbsp olive oil
- 50g parmesan cheese, grated
- Salt and pepper to taste

Instructions:
1. Cook pasta according to package directions
2. Heat olive oil in a large pan
3. Sauté onion and garlic until fragrant
4. Add diced tomatoes and cook for 5 minutes
5. Add cooked pasta and toss with sauce
6. Season with salt, pepper, and fresh basil
7. Serve topped with parmesan cheese

Now create a similar recipe using these ingredients: {', '.join(ingredients)}"""

        try:
            # Generate response using the pipeline with better parameters
            response = self.generator(
                prompt,
                max_new_tokens=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                pad_token_id=self.generator.tokenizer.pad_token_id
            )
            
            generated_text = response[0]['generated_text']
            print("Generated text:", generated_text)
            
            # Try to extract JSON from the response
            recipe_json = self.extract_json_from_text(generated_text)
            
            if recipe_json:
                return recipe_json
            else:
                # If JSON extraction fails, create a structured response
                return self.create_fallback_recipe(generated_text, ingredients)
                
        except Exception as e:
            print(f"Generation error: {e}")
            return self.create_fallback_recipe(f"Error generating recipe: {str(e)}", ingredients)
    
    def extract_json_from_text(self, text):
        """Extract JSON object from generated text."""
        try:
            # Look for JSON-like structure in the text
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
            
            # If no clear JSON structure, try to parse the entire text
            return json.loads(text)
            
        except json.JSONDecodeError:
            return None
    
    def create_fallback_recipe(self, generated_text, ingredients):
        """Create a structured recipe when JSON parsing fails."""
        # Create a simple recipe based on the ingredients
        recipe_name = "Simple " + " and ".join(ingredients[:3]).title() + " Recipe"
        
        # Create ingredient list with basic quantities
        ingredient_parts = []
        for ing in ingredients[:8]:
            if ing.lower() in ['pasta', 'rice', 'noodles']:
                ingredient_parts.append(f"200g {ing}")
            elif ing.lower() in ['oil', 'olive oil']:
                ingredient_parts.append(f"2 tbsp {ing}")
            elif ing.lower() in ['salt', 'pepper']:
                ingredient_parts.append(f"{ing} to taste")
            elif ing.lower() in ['garlic']:
                ingredient_parts.append(f"2 cloves {ing}, minced")
            elif ing.lower() in ['onion']:
                ingredient_parts.append(f"1 medium {ing}, chopped")
            else:
                ingredient_parts.append(f"1 cup {ing}")
        
        # Create basic cooking instructions
        instructions = [
            f"Prepare all ingredients: wash and chop vegetables as needed.",
            f"Heat oil in a large pan over medium heat.",
            f"Add onions and garlic if available, cook until fragrant.",
            f"Add main ingredients ({', '.join(ingredients[:3])}) and cook for 8-10 minutes.",
            f"Season with salt and pepper to taste.",
            f"Serve hot and enjoy your {recipe_name.lower()}!"
        ]
        
        return {
            "Name": recipe_name,
            "RecipeIngredientParts": ingredient_parts,
            "RecipeInstructions": instructions
        }
    
    def format_recipe(self, recipe_dict):
        """Format the recipe dictionary into a readable string."""
        output = f"## {recipe_dict.get('Name', 'Recipe')}\n\n"
        
        ingredients = recipe_dict.get('RecipeIngredientParts', [])
        if ingredients:
            output += "### Ingredients:\n"
            for ing in ingredients:
                output += f"- {ing}\n"
            output += "\n"
        
        instructions = recipe_dict.get('RecipeInstructions', [])
        if instructions:
            output += "### Instructions:\n"
            for i, step in enumerate(instructions, 1):
                output += f"{i}. {step}\n"
        
        return output

def main():
    """Example usage of the recipe generator."""
    # Initialize the generator with a more capable model
    generator = RecipeGenerator(model_name="google/flan-t5-base")
    
    # Example parameters
    pantry = "pasta, tomato, garlic, onion, basil, olive oil, parmesan"
    diet = "none"
    time = 25
    
    print(f"Generating recipe with:")
    print(f"Pantry: {pantry}")
    print(f"Diet: {diet}")
    print(f"Time: {time} minutes")
    print("-" * 50)
    
    # Generate recipe
    recipe = generator.generate_recipe(pantry, diet, time)
    
    # Format and print
    formatted_recipe = generator.format_recipe(recipe)
    print(formatted_recipe)
    
    # Also print raw JSON for debugging
    print("\nRaw JSON output:")
    print(json.dumps(recipe, indent=2))

if __name__ == "__main__":
    main()
