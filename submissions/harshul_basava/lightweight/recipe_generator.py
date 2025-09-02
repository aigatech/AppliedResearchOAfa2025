"""
Simple recipe generator that tailors itself to user preferences.
- generates recipes using Ashikan/dut-recipe-generator
- updates neural network to recognize recipes that the user prefers
- selects recipes to show users based on scores from the neural network
"""
import random
import torch
import json
import sys
import os

from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preference_nn import score_recipes, update_model

"""
Assigning important global variables.
"""

MODEL_NAME = "Ashikan/dut-recipe-generator"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ITER = 5

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/common_ingredients.csv"), "r", encoding="utf-8") as f:
    COMMON_INGREDIENTS = [line.strip() for line in f if line.strip()]

def generate_recipe(ingredients: list[str], max_new_tokens: int = 1024) -> str:
    """
    Generates a recipe using a language model given a list of ingredients.

    Args:
        ingredients (list[str]): List of ingredient names.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 1024.

    Returns:
        str: Generated recipe text.
    """

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, 
                                                 device_map=DEVICE)

    prompt = '{"prompt": ' + json.dumps(ingredients)
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.2,
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return full_text.strip()

def print_recipe(recipe: dict):
    """
    Prints the recipe details including title, ingredients, and method.

    Args:
        recipe (dict): Recipe dictionary containing title, ingredients, and method.
    """

    # Print title
    print(f"\n🍽️  {recipe.get('title', 'Untitled Recipe')}\n")
    
    # Print ingredients
    ingredients = recipe.get('ingredients', [])
    if ingredients:
        print("Ingredients:")
        for i, item in enumerate(ingredients, start=1):
            print(f"- {item}")
        print()
    
    # Print method / steps
    method = recipe.get('method', [])
    if method:
        print("Method:")
        for step in method:
            # Split long method string into multiple steps if needed
            steps = [s.strip() for s in step.split('-') if s.strip()]
            for s in steps:
                print(f"- {s}")
        print()

def append_ingredients(lst):
    """
    Samples two unique ingredients from INGREDIENTS not in lst, appends them, and shuffles the list.

    Args:
        lst (list): List of current ingredients (length must be 5).

    Returns:
        list: New list with two additional ingredients, shuffled.

    Raises:
        ValueError: If lst is not length 5 or not enough unique ingredients to sample.
    """
    if len(lst) != 5:
        raise ValueError("List is wrong size.")
    
    # Sample two unique ingredients not in lst
    available = [ing for ing in COMMON_INGREDIENTS if ing not in lst]
    if len(available) < 2:
        raise ValueError("Not enough unique ingredients to sample.")
    new_ings = random.sample(available, 2)

    # Append and shuffle
    new_lst = lst + new_ings
    random.shuffle(new_lst)
    return new_lst

if __name__ == "__main__":

    while True:

        _ = input("Welcome! Type E to exit, or anything else to continue: ")

        if _.strip().upper() == "E":
            break

        ingredients: list[str] = [""] * 5
        
        for i in range(5):
            curr = input(f"Enter ingredient {i + 1}: ")
             
            if curr == "" or curr is None:
                available = [ing for ing in COMMON_INGREDIENTS if ing not in ingredients]
                ingredients[i] = random.sample(available, 1)[0]
            else: 
                ingredients[i] = curr

        curr_recipes = []
        curr_recipes_dict = {}

        while len(curr_recipes) < 4:
            new_ingredients = append_ingredients(lst=ingredients)
            recipe = generate_recipe(new_ingredients)

            try:
                recipe_dict = json.loads(recipe)
                curr_recipes.append(recipe_dict)
            except:
                continue

        scores = score_recipes(curr_recipes)
        print("Initial Scores:")
        for r, s in zip(curr_recipes, scores):
            print(f"{r['title']} → {s:.3f}\n")

        top_indices = scores.argsort()[-2:][::-1]  # descending order
        top_recipes = [curr_recipes[i] for i in top_indices]
        print("\nTop 2 to show user:")
        for r in top_recipes:
            print_recipe(r)

        while True:
            choice = input("Pick A or B: ").strip().upper()
            if choice in ["A", "B"]:
                print(f"You picked {choice}")
                break
            else:
                print("Invalid input. Please enter only A or B.")

        if choice == "A":
            user_choice = top_recipes[0]
            user_reject = top_recipes[1]
        else:
            user_choice = top_recipes[1]
            user_reject = top_recipes[0]

        loss = update_model(user_choice, user_reject)
        print(f"\nUpdated model with loss: {loss:.4f}")