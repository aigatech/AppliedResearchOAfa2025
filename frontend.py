import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from expand import Expand
from captioner import ImageCaptioner

# Load environment variables
load_dotenv()

@st.cache_resource
def get_caption():
    return ImageCaptioner()

def expansion():
    return Expand()

st.title("Make a Story!")

captioner = get_caption()
expander = expansion()

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")
    
    if st.button("Generate Caption"):
        caption = captioner.generate_caption_from_image(image)
        st.write("**Caption:**", caption)
        story = expander.expand(caption)
        st.write("**Story**", story)
