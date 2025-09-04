#!/usr/bin/env python3
"""
Recipe Generator - From Ingredients to Meals
Uses HuggingFace transformers to generate recipes from ingredients.
"""

import sys
import argparse
import random
import time
from transformers import pipeline, set_seed

class RecipeGenerator:
    def __init__(self):
        """Initialize the recipe generator with a text generation model."""
        print("Loading model... (this may take a moment)")
        # Using GPT-2 for fast CPU inference
        self.generator = pipeline('text-generation', 
                                model='gpt2', 
                                tokenizer='gpt2',
                                device=-1)  # CPU only
        # Don't set a fixed seed - we want variety
    
    def generate_recipe(self, ingredients, max_length=200, temperature=0.7):
        """Generate a recipe from a list of ingredients."""
        # Set a random seed based on ingredients and time for variety
        seed = abs(hash(''.join(ingredients)) + int(time.time())) % (2**31)
        set_seed(seed)
        return self.create_ingredient_focused_recipe(ingredients, temperature)
    
    def create_ingredient_focused_recipe(self, ingredients, temperature):
        """Create a recipe that actually uses the provided ingredients."""
        # Generate a creative title using AI
        title_prompt = f"Recipe name for a dish made with {', '.join(ingredients)}:"
        
        try:
            title_result = self.generator(title_prompt, 
                                        max_length=len(title_prompt.split()) + 8,
                                        temperature=temperature,
                                        num_return_sequences=1,
                                        pad_token_id=50256,
                                        do_sample=True,
                                        truncation=True)
            
            generated_title = title_result[0]['generated_text'].replace(title_prompt, "").strip()
            # Clean up the title
            title_lines = generated_title.split('\n')
            title = title_lines[0] if title_lines[0] else f"{ingredients[0].title()} Delight"
            title = title.replace('"', '').replace(':', '').strip()
            
            if len(title) > 50 or len(title) < 3:
                title = f"{ingredients[0].title()} {ingredients[-1].title() if len(ingredients) > 1 else 'Dish'}"
                
        except:
            title = f"{ingredients[0].title()} {ingredients[-1].title() if len(ingredients) > 1 else 'Dish'}"
        
        # Create recipe structure
        formatted = f"🍳 {title}\n\n"
        formatted += "Ingredients:\n"
        
        for ingredient in ingredients:
            formatted += f"- {ingredient.title()}\n"
        formatted += "- Salt and pepper to taste\n"
        formatted += "- Oil or butter for cooking\n\n"
        
        formatted += "Instructions:\n"
        
        # Generate cooking steps that use the actual ingredients
        steps = self.generate_cooking_steps(ingredients, temperature)
        for i, step in enumerate(steps, 1):
            formatted += f"{i}. {step}\n"
        
        return formatted
    
    def generate_cooking_steps(self, ingredients, temperature):
        """Generate cooking steps that actually use the ingredients."""
        steps = []
        
        # Basic prep step
        steps.append(f"Prepare and wash all ingredients: {', '.join(ingredients)}.")
        
        # Generate a cooking method using AI
        cooking_prompt = f"How to cook {ingredients[0]} with {', '.join(ingredients[1:]) if len(ingredients) > 1 else 'seasonings'}. Step by step:"
        
        try:
            result = self.generator(cooking_prompt,
                                  max_length=len(cooking_prompt.split()) + 40,
                                  temperature=temperature,
                                  num_return_sequences=1,
                                  pad_token_id=50256,
                                  do_sample=True,
                                  truncation=True)
            
            generated_text = result[0]['generated_text'].replace(cooking_prompt, "").strip()
            
            # Extract useful cooking steps
            lines = generated_text.split('\n')
            for line in lines[:3]:
                line = line.strip()
                if line and len(line) > 15 and any(ing.lower() in line.lower() for ing in ingredients):
                    # Clean up the line
                    line = line.replace('1.', '').replace('2.', '').replace('3.', '').strip()
                    if line:
                        steps.append(line)
        except:
            pass
        
        # Add fallback steps if AI didn't generate good ones
        if len(steps) < 3:
            steps.append("Heat oil or butter in a pan over medium heat.")
            steps.append(f"Add {ingredients[0]} and cook until tender.")
            
            if len(ingredients) > 1:
                steps.append(f"Add {', '.join(ingredients[1:])} and cook together for 5-7 minutes.")
            
            steps.append("Season with salt and pepper to taste.")
            steps.append("Serve hot and enjoy! 🍽️")
        
        return steps[:6]  # Limit to 6 steps
    
    def format_recipe(self, raw_text, ingredients):
        """Fallback template recipe format."""
        title = f"{ingredients[0].title()} {ingredients[-1].title() if len(ingredients) > 1 else 'Delight'}"
        
        formatted = f"🍳 {title}\n\n"
        formatted += "Ingredients:\n"
        
        for ingredient in ingredients:
            formatted += f"- {ingredient.title()}\n"
        
        formatted += f"- Salt and pepper to taste\n\n"
        formatted += "Instructions:\n"
        formatted += f"1. Prepare all ingredients: {', '.join(ingredients)}.\n"
        formatted += "2. Heat a pan over medium heat.\n"
        formatted += f"3. Cook the {ingredients[0]} until tender.\n"
        
        if len(ingredients) > 1:
            formatted += f"4. Add {', '.join(ingredients[1:])} and mix well.\n"
            formatted += "5. Season with salt and pepper.\n"
            formatted += "6. Cook for 5-10 minutes until everything is well combined.\n"
        else:
            formatted += "4. Season with salt and pepper to taste.\n"
            formatted += "5. Cook until done to your preference.\n"
        
        formatted += "7. Serve hot and enjoy! 🍽️\n"
        
        return formatted

def main():
    parser = argparse.ArgumentParser(description='Generate recipes from ingredients')
    parser.add_argument('ingredients', 
                       help='Comma-separated list of ingredients (e.g., "eggs,cheese,tomatoes")')
    parser.add_argument('--output', '-o', 
                       help='Output file to save the recipe')
    parser.add_argument('--temperature', '-t', type=float, default=0.8,
                       help='Creativity level (0.1-1.0, default: 0.8)')
    
    args = parser.parse_args()
    
    # Parse ingredients
    ingredients = [ing.strip() for ing in args.ingredients.split(',')]
    
    if not ingredients or ingredients == ['']:
        print("Error: Please provide at least one ingredient!")
        sys.exit(1)
    
    # Generate recipe
    generator = RecipeGenerator()
    recipe = generator.generate_recipe(ingredients, temperature=args.temperature)
    
    # Output recipe
    print(recipe)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(recipe)
        print(f"\nRecipe saved to {args.output}")

if __name__ == "__main__":
    main()