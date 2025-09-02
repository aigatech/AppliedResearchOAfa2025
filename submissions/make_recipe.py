import os
from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer

# --- 1. Setup Paths ---
output_dir = "recipe_outputs"
output_file_path = os.path.join(output_dir, "generated_recipes.txt")

# --- 2. Create Directory if it Doesn't Exist ---
os.makedirs(output_dir, exist_ok=True)
print(f"✅ Outputs will be saved to '{output_file_path}'")

model_path = './fine_tuned_distillgpt2-concepts'
model = ''
tokenizer = ''
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    print("✅ Model and Tokenizer loaded successfully from local path!")
except Exception as e:
    print(f"❌ Failed to load model. Check that the path is correct. Error: {e}")

# Load new structured model (device = -1 means use CPU)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=-1)

# For reproducible results
set_seed(42)

#invent new dish
prompt = (
    "[START]\n"
    "name: Jalebi Chicken\n"
    "diet: non-vegetarian\n"
    "flavor_profile: sweet\n"
    "course: dessert\n"
    "state: Punjab\n"
    "region: North"
)
# The model will start generating from here, filling in times, ingredients, and instructions.

print("--- Generating a new recipe based on prompt ---")
print(f"PROMPT:\n{prompt}\n...")

# Generate the recipe
generated_recipes = generator(
    prompt,
    max_length=350,          # Increased length for full recipes
    num_return_sequences=1,
    temperature=0.9,         # Keep it creative
    top_k=50,
    top_p=0.95,
    eos_token_id=generator.tokenizer.eos_token_id, # Helps the model know when to stop
    pad_token_id=generator.tokenizer.eos_token_id
)

# --- 4. Write the Output to the File ---
try:
    with open(output_file_path, 'w', encoding='utf-8') as f:
        print("Saving recipes to file...")
        for i, recipe in enumerate(generated_recipes):
            # Write a header for each recipe
            f.write(f"--- Generated Recipe {i+1} ---\n\n")
            # Write the recipe text
            f.write(recipe['generated_text'])
            # Add separators for readability
            f.write("\n\n" + "="*50 + "\n\n")
    print(f"✅ Successfully saved recipes to '{output_file_path}'")
except Exception as e:
    print(f"❌ Error saving file: {e}")