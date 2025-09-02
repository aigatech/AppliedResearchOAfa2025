import pandas as pd

# Load the dataset
df = pd.read_csv('../indian_food.csv')

# --- Data Cleaning ---
# Replace -1 and missing values with sensible defaults or 'unknown'.
df['prep_time'] = df['prep_time'].replace(-1, 15)
df['cook_time'] = df['cook_time'].replace(-1, 30)
df['flavor_profile'] = df['flavor_profile'].replace('-1', 'unknown').fillna('unknown')
df['state'] = df['state'].replace('-1', 'unknown').fillna('unknown')
df['region'] = df['region'].replace('-1', 'unknown').fillna('unknown')

# Drop rows where the most essential columns are missing
df.dropna(subset=['name', 'ingredients'], inplace=True)


def create_recipe_concept(row):
    """
    Creates a structured string for a recipe concept.
    """
    name = row.get('name', '')
    diet = row.get('diet', '')
    flavor = row.get('flavor_profile', 'unknown')
    course = row.get('course', '')
    state = row.get('state', 'unknown')
    region = row.get('region', 'unknown')
    prep = row.get('prep_time', 0)
    cook = row.get('cook_time', 0)
    # Clean up the ingredients string
    ingredients = ', '.join(row.get('ingredients', '').split(','))

    # Build the structured string (ending with ingredients)
    concept_text = (
        f"[START]\n"
        f"name: {name}\n"
        f"diet: {diet}\n"
        f"flavor_profile: {flavor}\n"
        f"course: {course}\n"
        f"state: {state}\n"
        f"region: {region}\n"
        f"prep_time: {prep}\n"
        f"cook_time: {cook}\n"
        f"ingredients: {ingredients}\n"
        f"[END]\n"
    )
    return concept_text

# Apply the function to create the new text format
df['structured_concept'] = df.apply(create_recipe_concept, axis=1)

# Save the structured concepts to a single text file
with open('indian_recipe_concepts.txt', 'w', encoding='utf-8') as f:
    for text in df['structured_concept']:
        f.write(text)

print("Data preprocessing complete. 'indian_recipe_concepts.txt' created.")