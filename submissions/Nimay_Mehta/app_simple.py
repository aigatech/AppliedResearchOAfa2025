import streamlit as st
import subprocess
import os

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
            with st.spinner("Cooking up your recipe... 👨🍳"):
                try:
                    # Run the CLI version and capture output
                    cmd = ["python", "recipe_gen.py", ",".join(ingredients), "--temperature", str(temperature)]
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
                    
                    if result.returncode == 0:
                        recipe = result.stdout
                        # Clean up the output
                        if "Loading model..." in recipe:
                            recipe = recipe.split("Loading model... (this may take a moment)\n")[-1]
                        
                        st.success("Recipe ready!")
                        st.text(recipe)
                        
                        # Download button
                        st.download_button(
                            label="📄 Download Recipe",
                            data=recipe,
                            file_name=f"recipe_{'-'.join(ingredients[:2])}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error(f"Error: {result.stderr}")
                        
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
            st.session_state.example_ingredients = "eggs, cheese"
    
    with col2:
        if st.button("🍝 Pasta, Garlic"):
            st.session_state.example_ingredients = "pasta, garlic, olive oil"
    
    with col3:
        if st.button("🐔 Chicken, Rice"):
            st.session_state.example_ingredients = "chicken, rice, vegetables"

# Handle example clicks
if 'example_ingredients' in st.session_state:
    ingredients_input = st.session_state.example_ingredients
    del st.session_state.example_ingredients
    st.rerun()

# Footer
st.markdown("---")
st.markdown("*Built with HuggingFace Transformers & Streamlit*")

# Instructions
st.sidebar.title("How to Use")
st.sidebar.write("1. Enter ingredients separated by commas")
st.sidebar.write("2. Adjust creativity level")
st.sidebar.write("3. Click 'Generate Recipe'")
st.sidebar.write("4. Download your recipe!")

st.sidebar.markdown("---")
st.sidebar.write("**Examples:**")
st.sidebar.write("• eggs, cheese, tomatoes")
st.sidebar.write("• pasta, garlic, olive oil")
st.sidebar.write("• chicken, rice, vegetables")