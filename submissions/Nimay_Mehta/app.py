import streamlit as st
from transformers import pipeline, set_seed
import time

@st.cache_resource
def load_model():
    """Load the model once and cache it."""
    return pipeline('text-generation', 
                   model='gpt2', 
                   tokenizer='gpt2',
                   device=-1)

def format_ai_recipe(ai_instructions, ingredients):
    """Format recipe using AI-generated instructions."""
    # Clean up the AI text
    clean_instructions = ai_instructions.replace('_', '').strip()
    
    # Generate title
    title = f"{ingredients[0].title()} {ingredients[-1].title() if len(ingredients) > 1 else 'Dish'}"
    
    # Format the recipe
    formatted = f"🍳 **{title}**\n\n"
    formatted += "**Ingredients:**\n"
    
    for ingredient in ingredients:
        formatted += f"- {ingredient.title()}\n"
    formatted += "- Salt and pepper to taste\n\n"
    
    formatted += "**Instructions:**\n"
    
    # Process AI-generated instructions
    lines = clean_instructions.split('\n')
    step_num = 1
    
    for line in lines[:6]:  # Limit to 6 steps
        line = line.strip()
        if line and not line.startswith('Recipe') and len(line) > 10:
            # Clean up the line
            if line.startswith(str(step_num)):
                formatted += f"{line}\n"
            else:
                formatted += f"{step_num}. {line}\n"
            step_num += 1
    
    # Add a final step if we don't have enough
    if step_num <= 3:
        formatted += f"{step_num}. Cook until done and serve hot! 🍽️"
    
    return formatted

def format_recipe(raw_text, ingredients):
    """Fallback template recipe format."""
    title = f"{ingredients[0].title()} {ingredients[-1].title() if len(ingredients) > 1 else 'Delight'}"
    
    formatted = f"🍳 **{title}**\n\n"
    formatted += "**Ingredients:**\n"
    
    for ingredient in ingredients:
        formatted += f"- {ingredient.title()}\n"
    
    formatted += f"- Salt and pepper to taste\n\n"
    formatted += "**Instructions:**\n"
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
    
    formatted += "7. Serve hot and enjoy! 🍽️"
    
    return formatted

def generate_recipe(ingredients, temperature=0.8):
    """Generate a recipe from ingredients."""
    generator = load_model()
    set_seed(42)
    
    ingredients_str = ", ".join(ingredients)
    prompt = f"Here's a delicious recipe using {ingredients_str}:\n\n{ingredients_str.title()} Recipe\n\nIngredients:\n- {ingredients_str}\n\nInstructions:\n1."
    
    result = generator(prompt, 
                      max_length=300,
                      temperature=temperature,
                      num_return_sequences=1,
                      pad_token_id=50256,
                      do_sample=True,
                      truncation=True)
    
    generated_text = result[0]['generated_text']
    
    # Extract AI-generated instructions
    if "Instructions:" in generated_text:
        parts = generated_text.split("Instructions:")
        if len(parts) > 1:
            instructions = parts[1].strip()
            return format_ai_recipe(instructions, ingredients)
    
    # Fallback to template
    return format_recipe("", ingredients)

# Streamlit App
st.set_page_config(
    page_title="Recipe Generator",
    page_icon="🍳",
    layout="centered"
)

st.title("🍳 Recipe Generator")
st.subheader("From Ingredients to Meals")

st.write("Enter your available ingredients and get a delicious recipe!")

# Input section
with st.container():
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ingredients_input = st.text_input(
            "Enter ingredients (comma-separated):",
            placeholder="eggs, cheese, tomatoes",
            help="Enter 2-5 ingredients for best results"
        )
    
    with col2:
        temperature = st.slider(
            "Creativity",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            step=0.1,
            help="Higher = more creative"
        )

# Generate button
if st.button("🔥 Generate Recipe", type="primary"):
    if ingredients_input:
        # Parse ingredients
        ingredients = [ing.strip() for ing in ingredients_input.split(',')]
        ingredients = [ing for ing in ingredients if ing]  # Remove empty strings
        
        if len(ingredients) >= 1:
            with st.spinner("Cooking up your recipe... 👨‍🍳"):
                try:
                    recipe = generate_recipe(ingredients, temperature)
                    
                    # Display recipe
                    st.success("Recipe ready!")
                    st.markdown(recipe)
                    
                    # Download button
                    st.download_button(
                        label="📄 Download Recipe",
                        data=recipe,
                        file_name=f"recipe_{'-'.join(ingredients[:2])}.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating recipe: {str(e)}")
        else:
            st.warning("Please enter at least one ingredient!")
    else:
        st.warning("Please enter some ingredients!")

# Example section
with st.expander("💡 Example Ingredients"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🥚 Eggs, Cheese"):
            st.session_state.example = "eggs, cheese"
    
    with col2:
        if st.button("🍝 Pasta, Garlic"):
            st.session_state.example = "pasta, garlic, olive oil"
    
    with col3:
        if st.button("🐔 Chicken, Rice"):
            st.session_state.example = "chicken, rice, vegetables"

# Handle example clicks
if 'example' in st.session_state:
    st.rerun()

# Footer
st.markdown("---")
st.markdown("*Built with HuggingFace Transformers & Streamlit*")