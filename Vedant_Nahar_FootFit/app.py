import gradio as gr
from cv_utils import process_foot_image
from recommend import get_recommendations


def analyze_foot(image, manual_length_inches, use_case, sizing_reference):
    """
    Main function to analyze foot image and generate recommendations.
    """
    if image is None:
        return None, "Please upload an image.", ""
    
    try:
        # Convert inches to cm for processing
        manual_length_cm = manual_length_inches * 2.54 if manual_length_inches else None
        
        # Process the foot image
        result = process_foot_image(
            image, 
            manual_length_cm=manual_length_cm
        )
        
        if result is None:
            return None, "Could not detect foot in the image. Please ensure the foot is clearly visible against a contrasting background.", ""
        
        overlay_image = result['overlay_image']
        arch_type = result['arch_type']
        width_category = result['width_category']
        length_mm = result['length_mm']
        men_size = result['men_size']
        women_size = result['women_size']
        
        # Create summary text
        summary = "## Foot Analysis Results\n\n"
        
        if length_mm:
            length_inches = length_mm / 25.4
            summary += f"**Foot length:** {length_inches:.1f} inches ({length_mm:.0f} mm)\n\n"
        else:
            summary += f"**Length (pixels):** {result['measurements']['length']:.0f} pixels\n\n"
            summary += "*Note: Please enter your foot length in inches for accurate sizing.*\n\n"
        
        summary += f"**Arch estimate:** {arch_type}\n\n"
        summary += f"**Width category:** {width_category}\n\n"
        
        if men_size and women_size:
            summary += f"**Estimated US Men's size:** {men_size}\n\n"
            summary += f"**Estimated US Women's size:** {women_size}\n\n"
        
        # Generate recommendations
        recommendations_text = ""
        try:
            # Clean up emoji prefixes from dropdown values
            clean_use_case = use_case.split(' ', 1)[1] if ' ' in use_case else use_case
            clean_sizing = sizing_reference.split(' ', 1)[1] if ' ' in sizing_reference else sizing_reference
            if clean_sizing == "Either":
                clean_sizing = "Unspecified"
            
            recommendations = get_recommendations(
                length_mm or 250,  # Use estimated length or default
                men_size, women_size, width_category, 
                arch_type, clean_use_case, clean_sizing
            )
            
            if length_mm:
                recommendations_text = f"## Shoe Recommendations\n\n{recommendations}"
            else:
                recommendations_text = f"## Shoe Recommendations\n\n*Based on detected foot characteristics (please enter foot length for sizing):*\n\n{recommendations}"
                
        except Exception as e:
            recommendations_text = f"## Shoe Recommendations\n\nUnable to generate recommendations: {str(e)}"
        
        return overlay_image, summary, recommendations_text
        
    except Exception as e:
        return None, f"Error processing image: {str(e)}", ""


# Custom CSS for dark modern theme
custom_css = """
/* Dark theme and modern styling */
.gradio-container {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #ffffff !important;
}

.dark {
    background: #000000 !important;
}

/* Header styling */
.main-header {
    text-align: center;
    padding: 2rem 0;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    margin-bottom: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.main-header h1 {
    color: #ffffff;
    font-size: 3rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(45deg, #00d4ff 0%, #ff6b9d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
}

.main-header p {
    color: #e0e0e0;
    font-size: 1.1rem;
    margin: 0.5rem 0 0 0;
}

.main-header .disclaimer {
    color: #ff6b9d;
    font-size: 0.9rem;
    font-style: italic;
    margin-top: 0.5rem;
}

/* Input section styling */
.input-section {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.3);
    backdrop-filter: blur(10px);
}

.input-section h3 {
    color: #ffffff;
    margin-top: 0;
    margin-bottom: 1rem;
    font-size: 1.3rem;
    font-weight: 600;
}

/* Results section styling */
.results-section {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.3);
    backdrop-filter: blur(10px);
}

/* Button styling */
.analyze-btn {
    background: linear-gradient(45deg, #00d4ff 0%, #ff6b9d 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: #ffffff !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4) !important;
    width: 100% !important;
    margin-top: 1rem !important;
}

.analyze-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 212, 255, 0.6) !important;
}

/* Input field styling */
.gr-textbox, .gr-number, .gr-dropdown {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}

/* Image upload styling */
.gr-image {
    border-radius: 12px !important;
    border: 2px dashed rgba(102, 126, 234, 0.5) !important;
    background: rgba(255, 255, 255, 0.02) !important;
}

/* Results content */
.results-content {
    background: rgba(255, 255, 255, 0.02);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid #667eea;
}

/* Step indicators */
.step-indicator {
    display: flex;
    align-items: center;
    margin: 0.5rem 0;
    color: #e0e0e0;
}

.step-number {
    background: linear-gradient(45deg, #00d4ff 0%, #ff6b9d 100%);
    color: #ffffff;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.5rem;
}
"""

# Create Gradio interface with modern dark theme
with gr.Blocks(
    title="FootFinder - AI Foot Analysis", 
    css=custom_css,
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="purple",
        neutral_hue="slate",
    ).set(
        body_background_fill="*neutral_950",
        block_background_fill="*neutral_900",
        border_color_primary="*neutral_700",
        button_primary_background_fill="linear-gradient(45deg, *primary_500, *secondary_500)",
    )
) as app:
    
    # Header section
    with gr.Row():
        with gr.Column():
            gr.HTML("""
                <div class="main-header">
                    <h1>👟 FootFinder</h1>
                    <p>AI-powered foot analysis and personalized shoe recommendations. We make money by SELLING your foot</p>
                </div>
            """)
    
    # Main content
    with gr.Row(equal_height=True):
        # Input section
        with gr.Column(scale=1):
            gr.HTML('<div class="input-section"><h3>📸 Upload & Measure</h3></div>')
            
            with gr.Group():
                # Step 1
                gr.HTML('''
                    <div class="step-indicator">
                        <div class="step-number">1</div>
                        <span>Take a top-down photo of your bare foot</span>
                    </div>
                ''')
                
                image_input = gr.Image(
                    label="",
                    type="numpy",
                    height=280,
                    container=False
                )
                
                # Step 2  
                gr.HTML('''
                    <div class="step-indicator">
                        <div class="step-number">2</div>
                        <span>Measure your foot length in inches</span>
                    </div>
                ''')
                
                manual_length = gr.Number(
                    label="",
                    placeholder="Enter foot length (4-18 inches)",
                    value=None,
                    minimum=4,
                    maximum=18,
                    step=0.1,
                    container=False
                )
                
                # Step 3
                gr.HTML('''
                    <div class="step-indicator">
                        <div class="step-number">3</div>
                        <span>Select your preferences</span>
                    </div>
                ''')
                
                with gr.Row():
                    use_case = gr.Dropdown(
                        label="Activity",
                        choices=["👟 Walking", "🏃 Running", "🥾 Hiking", "🎾 Court Sports", "⚽ Cleats/Boots"],
                        value="👟 Walking",
                        container=False
                    )
                    
                    sizing_reference = gr.Dropdown(
                        label="Sizing",
                        choices=["👨 Men's", "👩 Women's", "🚻 Either"],
                        value="🚻 Either",
                        container=False
                    )
            
            analyze_button = gr.Button("🔍 Analyze My Foot", elem_classes="analyze-btn")
        
        # Results section
        with gr.Column(scale=1.5):
            gr.HTML('<div class="results-section"><h3>📊 Analysis Results</h3></div>')
            
            with gr.Group():
                output_image = gr.Image(
                    label="📏 Foot Measurements",
                    type="numpy",
                    height=280,
                    container=True
                )
                
                with gr.Row():
                    with gr.Column():
                        summary_output = gr.Markdown(
                            "Upload a photo and enter your measurements to see detailed foot analysis.",
                            elem_classes="results-content"
                        )
                    
                with gr.Row():
                    with gr.Column():
                        recommendations_output = gr.Markdown(
                            "Get personalized shoe recommendations based on your unique foot characteristics.",
                            elem_classes="results-content"
                        )
    
    # Connect the analyze function to the button
    analyze_button.click(
        fn=analyze_foot,
        inputs=[image_input, manual_length, use_case, sizing_reference],
        outputs=[output_image, summary_output, recommendations_output]
    )


if __name__ == "__main__":
    app.launch()