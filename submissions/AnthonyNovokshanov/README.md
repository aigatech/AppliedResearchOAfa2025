# Explaining Paintings to the Blind

This project uses AI models to **analyze paintings and generate an accessible, multi-sensory experience**.  
When a user uploads an image, the program:

1. **Classifies the painting’s artistic style** (e.g., Baroque, Impressionism).  
2. **Generates a caption** that describes what is happening in the image.  
3. **Synthesizes speech** to narrate the style and description.  
4. **Creates background music** in the painting’s style to accompany the narration.  
5. Lets the user **“listen” to the painting** by playing the voiceover and music together.  

The application is built with **Tkinter** for a simple GUI interface.

---

## Features
- Upload an image of a painting.  
- Get the **art style** (via WikiArt-trained classifier).  
- Receive a **textual description** of the painting.  
- Hear a **spoken narration** of the painting.  
- Experience **AI-generated background music** inspired by the painting’s style.  

---

## Installation

- pip install torch torchvision torchaudio
- pip install transformers datasets soundfile pygame pydub scipy pillow
- Note* you may need to download ffmpeg and add it to your path for pydub to work

## Usage

1. Run the program with "python gui.py"
2. A Tkinter window will open with a placeholder image
3. Click **“Upload Image”** to select a painting from your computer.
4. Wait as the program:
- Classifies the style
- Generates a caption
- Creates narration + music
5. Once finished, click “Listen to the Painting!” to hear the combined audio.

## Models Used
- 🎨 Art Style Classification: prithivMLmods/WikiArt-Style

- 🖼 Image Captioning: Salesforce/blip-image-captioning-base

- 🗣 Text-to-Speech: microsoft/speecht5_tts

- 🎵 Text-to-Music: facebook/musicgen-large

## Additional Notes
- The image captioning model that I am using is significantly worse than others I have found on Hugging Face. However, due to the time limitation of 1 to 1.5 hours, I was unable to get the other models to work as they ran on different versions of the transformer installs. Given more time, the generated audio descriptor of the images could be significantly improved and this submitted version is thus only a demo.
- It may take a long time to fully generate a new audio for every uploaded painting, so for the sake of time I am including the generated output that I recieved for a "High Reneissance" style painting.


