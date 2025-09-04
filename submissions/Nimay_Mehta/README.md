# Recipe Generator – "From Ingredients to Meals"

A lightweight HuggingFace-powered application that generates cooking recipes based on user-provided ingredients using GPT-2 text generation.

## What it does

This tool takes a list of ingredients as input and generates a complete recipe including:
- Recipe title with emoji
- Formatted ingredient list
- Step-by-step cooking instructions
- Proper recipe structure

The generator uses HuggingFace's GPT-2 model for natural language generation, optimized to run efficiently on CPU.

## How to run it

### Installation

1. Install dependencies:
```bash
pip install transformers torch streamlit
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Usage Options

#### Option 1: Web Interface (Recommended)
```bash
streamlit run app_simple.py
```
Then open your browser to the displayed URL (usually http://localhost:8501)

*Note: If you encounter import issues, use `app_simple.py` which works around Streamlit import conflicts.*

#### Option 2: Command Line
**Basic usage:**
```bash
python recipe_gen.py "eggs,cheese,tomatoes"
```

**With creativity control:**
```bash
python recipe_gen.py "chicken,rice,vegetables" --temperature 0.9
```

**Save to file:**
```bash
python recipe_gen.py "pasta,garlic,olive oil" --output my_recipe.txt
```

### Example Output

Input: `python recipe_gen.py "eggs,cheese,tomatoes"`

```
🍳 Delicious Eggs Dish

Ingredients:
- Eggs
- Cheese
- Tomatoes
- Salt and pepper to taste

Instructions:
1. Prepare all ingredients: eggs, cheese, tomatoes.
2. Heat a pan over medium heat.
3. Cook the eggs until tender.
4. Add cheese, tomatoes and mix well.
5. Season with salt and pepper.
6. Cook for 5-10 minutes until everything is well combined.
7. Serve hot and enjoy! 🍽️
```

## Features

### Web Interface (app_simple.py)
- 🌐 Interactive Streamlit web app
- 🎛️ Creativity slider for recipe variation
- 📄 Download generated recipes
- 💡 Quick example ingredient buttons
- 🎨 Clean, user-friendly interface
- 🔧 Works around import conflicts

### Command Line (recipe_gen.py)
- ⚡ Fast CLI execution
- 🔧 Advanced parameter control
- 💾 File output options

## Technical Details

- **Model:** GPT-2 (CPU-optimized)
- **Libraries:** HuggingFace Transformers, PyTorch, Streamlit
- **Runtime:** ~10-30 seconds on CPU
- **Input:** Comma-separated ingredient list
- **Output:** Formatted recipe with emojis

## Parameters

- `ingredients`: Comma-separated list of ingredients (required)
- `--temperature` / `-t`: Creativity level (0.1-1.0, default: 0.8)
- `--output` / `-o`: Save recipe to file (optional)

## Notes

- First run may take longer as the model downloads (~500MB)
- Works entirely on CPU - no GPU required
- Optimized for 3-5 ingredients for best results
- Model files are cached locally after first download