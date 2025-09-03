import sys
from transformers import pipeline

def generate_recipe(ingredients: str):
    generator = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct", truncation=True)

    prompt = f"""
        Write a short, clear recipe using ONLY these ingredients: {ingredients}.
        Respond in this exact format:

        Title
        Ingredients:
        - item
        Instructions:
        1. step
        2. step

        Do not include hashtags, comments, emojis, or anything else.
        Recipe:
        """

    output = generator(
        prompt,
        max_length=300,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9
    )

    recipe = output[0]["generated_text"]

    recipe = output[0]["generated_text"][len(prompt):].strip()
    recipe = clean_recipe(recipe)
    return recipe

def clean_recipe(text: str) -> str:
    if "Instructions:" in text:
        text = text.split("Instructions:")[-1].strip()
        text = "Instructions:\n" + text

    text = text.split("Note:")[0].strip()

    text = text.replace('"""', "").strip()

    return text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py \"ingredient1, ingredient2, ...\"")
    else:
        ingredients = sys.argv[1]
        recipe = generate_recipe(ingredients)
        print("Recipe suggestion:\n")
        print(recipe)