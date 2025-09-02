import gradio as gr
import logging
import time
from functools import lru_cache
from src.lexicon import load_lexicon
from src.translate import hybrid_translate, calculate_translation_confidence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOCAL_MODEL_DIR = "./models/flan-t5-small"

@lru_cache(maxsize=1)
def load_cached_lexicon():
    """Load lexicon with caching"""
    return load_lexicon()

@lru_cache(maxsize=1) 
def load_cached_model():
    """Load model and tokenizer with caching and optimization"""
    try:
        logger.info("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            LOCAL_MODEL_DIR, 
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully on {device}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None

lex = None
model = None
tokenizer = None

def initialize_components():
    """Initialize components on first use"""
    global lex, model, tokenizer
    if lex is None:
        lex = load_cached_lexicon()
    if model is None or tokenizer is None:
        model, tokenizer = load_cached_model()

def translate_with_metrics(text, max_emojis=12, creativity=0.7):
    """Enhanced translation function with performance metrics and error handling"""
    if not text.strip():
        return "", 0.0, 0.0, "Please enter some text to translate."
    
    try:
        initialize_components()
        
        if model is None or tokenizer is None:
            return "", 0.0, 0.0, "Model failed to load. Using lexicon-only translation."
        
        start_time = time.time()
        
        with torch.no_grad():
            emojis = hybrid_translate(
                text, 
                lex, 
                model=model, 
                tokenizer=tokenizer, 
                max_emojis=max_emojis,
                temperature=creativity
            )
        
        inference_time = time.time() - start_time
        
        # Calculate confidence score
        confidence = calculate_translation_confidence(text, emojis, lex)
        
        # Format output
        emoji_string = " ".join(emojis) if emojis else "No emojis generated"
        
        return emoji_string, confidence, inference_time, "Translation successful!"
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return "Translation failed", 0.0, 0.0, f"Error: {str(e)}"

def create_enhanced_demo():
    """Create enhanced Gradio interface with better UX"""
    
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .emoji-output {
        font-size: 24px !important;
        line-height: 1.5 !important;
    }
    .metric-display {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    """
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=css,
        title="🔤➡️😀 Emoji Story Translator"
    ) as demo:
        
        gr.Markdown("""
        # 🔤➡️😀 Emoji Story Translator
        Transform your stories into expressive emoji sequences using AI!
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="📖 Enter your story",
                    placeholder="Once upon a time, there was a brave knight who loved pizza...",
                    lines=5,
                    max_lines=10
                )
                
                with gr.Row():
                    max_emojis = gr.Slider(
                        minimum=3,
                        maximum=25,
                        value=12,
                        step=1,
                        label="🎯 Maximum emojis"
                    )
                    creativity = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="🎨 Creativity level"
                    )
                
                with gr.Row():
                    translate_btn = gr.Button("✨ Translate", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
            
            with gr.Column(scale=2):
                output_emojis = gr.Textbox(
                    label="😀 Emoji Translation",
                    lines=4,
                    interactive=False,
                    elem_classes=["emoji-output"]
                )
                
                with gr.Row():
                    confidence_score = gr.Number(
                        label="🎯 Confidence Score",
                        precision=2,
                        interactive=False
                    )
                    inference_time = gr.Number(
                        label="⚡ Inference Time (s)",
                        precision=3,
                        interactive=False
                    )
                
                status_message = gr.Textbox(
                    label="📊 Status",
                    interactive=False,
                    max_lines=2
                )
        
        # Example stories
        gr.Markdown("### 💡 Try these example stories:")
        example_stories = [
            "The cat sat on the mat and smiled happily.",
            "I love eating pizza on sunny summer days with friends.",
            "The brave knight fought the fierce dragon to save the princess.",
            "She danced under the moonlight while stars twinkled above.",
            "The little boy cried when his ice cream fell on the ground."
        ]
        
        examples = gr.Examples(
            examples=[[story] for story in example_stories],
            inputs=[input_text],
            label="Click an example to try it:"
        )
        
        char_count = gr.Markdown("**Characters:** 0")
        
        def update_char_count(text):
            return f"**Characters:** {len(text)}"
        
        input_text.change(update_char_count, inputs=[input_text], outputs=[char_count])
        
        translate_btn.click(
            translate_with_metrics,
            inputs=[input_text, max_emojis, creativity],
            outputs=[output_emojis, confidence_score, inference_time, status_message]
        )
        
        def clear_all():
            return "", "", 0.0, 0.0, "Ready for translation!", "**Characters:** 0"
        
        clear_btn.click(
            clear_all,
            outputs=[input_text, output_emojis, confidence_score, inference_time, status_message, char_count]
        )
        
        gr.Markdown("""
        ---
        ###Tips for better translations:
        - **Longer stories** tend to produce more creative emoji sequences
        - **Higher creativity** values make the AI more experimental
        - **Lower creativity** values stick closer to literal meanings
        - The system combines a curated emoji dictionary with AI generation
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_enhanced_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False  
    )