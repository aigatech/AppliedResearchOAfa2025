from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
import torch
import gradio as gr

# Load models with error handling
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    sentiment_analyzer = pipeline("sentiment-analysis", 
                                model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")

def analyze_image(image):
    """
    Analyzes an uploaded image with proper input handling for BLIP generate
    """
    try:
        if image is None:
            return "❌ Please upload an image first!"
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process the image correctly
        inputs = processor(images=image, return_tensors="pt")
        
        # Pass the pixel_values tensor specifically to generate()
        with torch.no_grad():
            output = model.generate(pixel_values=inputs["pixel_values"], max_length=50, num_beams=4)
        
        caption = processor.decode(output[0], skip_special_tokens=True)
        
        # Sentiment analysis of the caption
        sentiment_result = sentiment_analyzer(caption)[0]
        sentiment_label = sentiment_result['label']
        confidence = sentiment_result['score']
        
        # Map sentiment labels to readable mood
        if sentiment_label == "LABEL_2":
            mood = "Positive 😊"
            interpretation = "This image has a positive/happy vibe!"
        elif sentiment_label == "LABEL_0":
            mood = "Negative 😔"
            interpretation = "This image seems to have a negative/sad mood."
        else:
            mood = "Neutral 😐"
            interpretation = "This image has a neutral mood."
        
        # Format results nicely with Markdown
        result = f"## 📸 Image Analysis Results\n\n"
        result += f"**📝 Caption:** {caption}\n\n"
        result += f"**🎭 Sentiment:** {mood} ({confidence:.1%} confidence)\n\n"
        result += f"**💭 Interpretation:** {interpretation}"
        
        return result
        
    except Exception as e:
        return f"❌ Error processing image: {str(e)}\n\nPlease try uploading a different image."

# Create the Gradio interface
demo = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(type="pil", label="📤 Upload Your Image"),
    outputs=gr.Markdown(label="🔍 Analysis Results"),
    title="🖼️ BLIP Image Captioner & Sentiment Analyzer",
    description="""
    ### How it works:
    1. 📤 Upload any image (JPG, PNG, etc.)
    2. 🤖 AI generates a caption describing what it sees
    3. 🎭 Sentiment analysis determines the mood of the caption
    4. ✨ Get instant results with emojis and explanations!
    
    *Powered by Salesforce BLIP and HuggingFace Transformers*
    """,
    theme=gr.themes.Soft(),
    flagging_mode=None
)

if __name__ == "__main__":
    print("🚀 Starting BLIP Image Captioner...")
    demo.launch(share=True, server_name="0.0.0.0")
