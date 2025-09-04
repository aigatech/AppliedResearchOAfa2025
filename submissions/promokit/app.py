import gradio as gr
import os
import sys
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core modules
from core import branding, copywriter, ranker, flyer, video, images_remote, safety

# Load environment variables
load_dotenv()

def generate_promokit(prompt, url, tone, use_remote_image, use_tts):
    """Main generation function for PromoKit."""
    try:
        # 1) Get brief and palette
        brief = ""
        palette = None
        
        if url and url.strip():
            # Validate URL
            if not safety.validate_url(url):
                return None, None, None, "Invalid URL provided"
            
            # Scrape website
            scraped_data = branding.scrape(url)
            if scraped_data["text"]:
                summary = branding.summarize(scraped_data["text"])
                brief = summary["summary"]
                palette = branding.extract_palette(scraped_data["text"], scraped_data["candidate_hex_colors"])
                # Override tone if detected from website
                if summary["tone"]:
                    tone = summary["tone"]
        
        if not brief:
            # Use prompt if no URL or scraping failed
            if not prompt or not prompt.strip():
                return None, None, None, "Please provide either a prompt or URL"
            brief = prompt
        
        # Sanitize brief
        brief = safety.sanitize_text(brief)
        if not safety.is_safe(brief):
            return None, None, None, "Content safety check failed"
        
        # 2) Generate copy
        copy_result = copywriter.gen_copy(brief, tone, palette_hint=palette)
        
        # Safety check for generated content
        if not safety.check_content_safety(copy_result):
            copy_result = safety.get_safe_fallback_content()
        
        # 3) Rank and select best elements
        copy_result = ranker.rank_copy_elements(copy_result, goal="increase engagement")
        
        # Extract elements
        headline = copy_result["headline"]
        tagline = copy_result["tagline"]
        bullets = copy_result["bullets"]
        cta = copy_result["ctas"][0] if copy_result["ctas"] else "Get Started"
        final_palette = copy_result.get("palette", palette) or ["#2563EB", "#64748B", "#F59E0B"]
        
        # 4) Generate background image (optional)
        bg_path = None
        if use_remote_image and os.getenv("HF_TOKEN"):
            bg_prompt = f"Minimal, abstract, {tone} background related to: {brief[:50]}"
            bg_path = images_remote.generate_safe_background(bg_prompt)
        
        # 5) Generate flyer
        flyer_png = flyer.render_png(headline, tagline, bullets, cta, final_palette, bg_path)
        flyer_pdf = flyer.render_pdf(flyer_png)
        
        # 6) Generate video
        captions = [headline] + bullets + [cta]
        video_path = video.compose_slideshow(
            flyer_png, 
            captions, 
            duration_per=4, 
            tts_text=brief if use_tts else None
        )
        
        return flyer_png, flyer_pdf, video_path, "Generation completed successfully!"
        
    except Exception as e:
        print(f"Error in generate_promokit: {e}")
        return None, None, None, f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="PromoKit Lite - AI Marketing Content Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎨 PromoKit Lite
    
    Generate professional marketing flyers and promo videos from a prompt or website URL.
    
    **Features:**
    - 📄 Create flyers (PNG + PDF)
    - 🎥 Generate promo videos with captions
    - 🎨 Automatic color palette extraction
    - 🌐 Website content scraping
    - 🤖 AI-powered copywriting
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            
            prompt = gr.Textbox(
                label="Business Description",
                placeholder="Describe your business, service, or product...",
                lines=4,
                max_lines=6
            )
            
            url = gr.Textbox(
                label="Website URL (Optional)",
                placeholder="https://your-website.com",
                lines=1
            )
            
            tone = gr.Dropdown(
                choices=["friendly", "professional", "playful", "bold"],
                value="friendly",
                label="Brand Tone"
            )
            
            with gr.Row():
                use_remote_image = gr.Checkbox(
                    label="Use AI Background Image",
                    value=False,
                    info="Requires HF_TOKEN environment variable"
                )
                use_tts = gr.Checkbox(
                    label="Add Voiceover (TTS)",
                    value=False,
                    info="Coming soon"
                )
            
            generate_btn = gr.Button(
                "🚀 Generate Content",
                variant="primary",
                size="lg"
            )
            
            status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### Output")
            
            with gr.Tab("Flyer"):
                flyer_image = gr.Image(
                    label="Flyer Preview",
                    type="filepath",
                    height=400
                )
                flyer_download = gr.File(
                    label="Download Flyer (PDF)",
                    file_count="single"
                )
            
            with gr.Tab("Video"):
                video_output = gr.Video(
                    label="Promo Video",
                    height=400
                )
    
    # Event handlers
    generate_btn.click(
        fn=generate_promokit,
        inputs=[prompt, url, tone, use_remote_image, use_tts],
        outputs=[flyer_image, flyer_download, video_output, status]
    )
    
    # Add some helpful examples
    gr.Examples(
        examples=[
            [
                "Modern yoga studio in downtown Atlanta offering classes for all levels",
                "",
                "friendly",
                False,
                False
            ],
            [
                "Professional consulting firm specializing in digital transformation",
                "",
                "professional",
                True,
                False
            ],
            [
                "Creative design agency creating stunning websites and branding",
                "",
                "playful",
                False,
                False
            ]
        ],
        inputs=[prompt, url, tone, use_remote_image, use_tts],
        label="Example Prompts"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
