# Chef Buzz - Recipe Generator

## What it does
Hi! This app is a fun twist on a recipe-generating model that adapts to user preferences by presenting them with multiple options to pick from. Using the user's preference data, we train a neural network to identify which types of recipes interest the user and which don't fit their style. Additionally, I started playing around with using direct policy optimization to directly train the language model on the user's preferences based on newly generated recipes.

- `preference_nn.py`: Implements a neural network to model user ingredient preferences.
- `recipe_generator.py`: Uses the trained model and available ingredients to generate recipe recommendations.

## How to run it
1. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```
2. Run the recipe generator:
   ```bash
   python lightweight/recipe_generator.py
   ```