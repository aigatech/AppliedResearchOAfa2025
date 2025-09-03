import gradio as gr
import os
from utils.translator import Translator
from utils.speak import Speaker
from utils.drawer import ImageGenerator


translator = Translator()
speaker = Speaker()

# PUT YOUR HF_TOKEN HERE FOR IMAGE GENERATION
os.environ["HF_TOKEN"] = "HF_TOKEN"

if os.environ["HF_TOKEN"]:
    image_generator = ImageGenerator(os.environ["HF_TOKEN"])
else:
    image_generator = None
    print("Warning: HF_TOKEN not found. Image generation will be disabled.")

def translate_text(text, target_language):
    if not text.strip():
        return "Please enter some text to translate."
    
    # lang codes for the translator pipeline
    language_mapping = {
        "French": "fr",
        "Hindi": "hi", 
        "German": "de",
        "Spanish": "es"
    }
    
    target_lang_code = language_mapping.get(target_language)
    if not target_lang_code:
        return f"Language {target_language} not supported."
    
    try:
        translated_text = translator.translate("en", target_lang_code, text)
        return translated_text
    except Exception as e:
        return f"Translation error: {str(e)}"

def audio_visual_learning(translated_text, target_language, original_text):
    if not translated_text.strip():
        return None, None
    
    # lang codes for the speaker pipeline
    speaker_language_mapping = {
        "French": "fra",
        "Hindi": "hin", 
        "German": "deu",
        "Spanish": "spa"
    }
    
    speaker_lang_code = speaker_language_mapping.get(target_language)
    if not speaker_lang_code:
        return None, None
    
    audio_file = None
    image_file = None
    
    try:
        # Generate audio and save to file
        output_file = f"translation_audio_{speaker_lang_code}.wav"
        speaker.speak(speaker_lang_code, translated_text, output_file)
        audio_file = output_file
    except Exception as e:
        print(f"Speech generation error: {str(e)}")
    
    try:
        # Generate image if available
        if image_generator:
            # Get the old eng text
            image = image_generator.draw(original_text)
            image_file = f"translation_image_{speaker_lang_code}.png"
            image.save(image_file)
        else:
            print("Image generator not available - HF_TOKEN not set")
    except Exception as e:
        print(f"Image generation error: {str(e)}")
    
    return audio_file, image_file

# Make interface
with gr.Blocks(title="Translation App", theme=gr.themes.Soft()) as app:
    gr.Markdown("#Translation App")
    gr.Markdown("Translate English text to French, Hindi, German, or Spanish")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="English Text",
                placeholder="Enter text in English to translate...",
                lines=2,
                max_lines=10
            )
            
            target_language = gr.Dropdown(
                choices=["French", "Hindi", "German", "Spanish"],
                label="Target Language",
                value="French"
            )
            
            translate_btn = gr.Button("Translate", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Translated Text",
                lines=4,
                max_lines=10,
                interactive=False
            )
            
            speak_btn = gr.Button("Audio + Visual Learning", variant="secondary")
            audio_output = gr.Audio(label="Audio Output", type="filepath")
            image_output = gr.Image(label="Visual Learning", type="filepath")
    
    translate_btn.click(
        fn=translate_text,
        inputs=[input_text, target_language],
        outputs=output_text
    )
    
    speak_btn.click(
        fn=audio_visual_learning,
        inputs=[output_text, target_language, input_text],
        outputs=[audio_output, image_output]
    )
    


if __name__ == "__main__":
    app.launch(share=True)
