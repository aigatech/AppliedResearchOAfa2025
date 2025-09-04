import gradio as gr
from transformers import pipeline
import json

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define mood categories and hardcode songs
mood_categories = ["happy and energetic","sad and melancholic", "nostalgic and reflective","motivated and determined","relaxed and peaceful","anxious and stressed","romantic and dreamy",
    "angry and frustrated","confident and powerful","contemplative and introspective"]
mood_playlists = {
    "happy and energetic": {
        "title": "Happy and Energetic", "songs": ["Can't Stop the Feeling - Justin Timberlake","Uptown Funk - Mark Ronson ft. Bruno Mars", "Happy - Pharrell Williams","Good as Hell - Lizzo","Shake It Off - Taylor Swift","Walking on Sunshine - Katrina and the Waves"]
    },
    "sad and melancholic": {"title": "Melancholic Mood","songs": ["Mad World - Gary Jules","The Night We Met - Lord Huron","Skinny Love - Bon Iver","Hurt - Johnny Cash","Black - Pearl Jam","Breathe Me - Sia"]
    },
    "nostalgic and reflective": {"title": "Walk down Memory Lane","songs": ["The Way You Look Tonight - Frank Sinatra","Yesterday - The Beatles","Photograph - Ed Sheeran","Vienna - Billy Joel","Time After Time - Cyndi Lauper","Castle on the Hill - Ed Sheeran"]
    },
    "motivated and determined": {"title": "Motivation Time","songs": ["Eye of the Tiger - Survivor","Stronger - Kelly Clarkson","Can't Hold Us - Macklemore & Ryan Lewis","Thunder - Imagine Dragons","Roar - Katy Perry","Don't Stop Believin' - Journey"]
    },
    "relaxed and peaceful": {"title": "Zen Zone","songs": ["Weightless - Marconi Union","Clair de Lune - Claude Debussy","River - Joni Mitchell","Mad About You - Sting","Holocene - Bon Iver","The Scientist - Coldplay"]
    },
    "anxious and stressed": {"title": "Opposite of Calm","songs": ["Breathe (2 AM) - Anna Nalick","Unwell - Matchbox Twenty","Heavy - Linkin Park ft. Kiiara","Anxiety - Julia Michaels ft. Selena Gomez","Peace of Mind - Boston","Don't Worry Be Happy - Bobby McFerrin"]
    },
    "romantic and dreamy": {"title": "Love & Dreams","songs": ["Perfect - Ed Sheeran","All of Me - John Legend","Thinking Out Loud - Ed Sheeran","At Last - Etta James","La Vie En Rose - Édith Piaf","Make You Feel My Love - Adele"]
    },
    "angry and frustrated": {"title": "⚡ Release the Rage","songs": ["Break Stuff - Limp Bizkit","In the End - Linkin Park","Bodies - Drowning Pool","Killing in the Name - Rage Against the Machine","Break My Stride - Matthew Wilder","Stressed Out - Twenty One Pilots"]
    },
    "confident and powerful": {"title": "Queens and Kings","songs": ["Confident - Demi Lovato","Stronger (What Doesn't Kill You) - Kelly Clarkson","Fight Song - Rachel Platten","Titanium - David Guetta ft. Sia","Unstoppable - Sia","Champion - Carrie Underwood"]
    },
    "contemplative and introspective": {"title": "Deep Thoughts","songs": ["The Sound of Silence - Simon & Garfunkel","Hurt - Nine Inch Nails","Mad World - Tears for Fears","Everybody Hurts - R.E.M.","Losing Religion - R.E.M.","Comfortably Numb - Pink Floyd"]
    }
}

def analyze(mood_text):    
    try:
        result = classifier(mood_text, mood_categories)
        predicted_mood = result['labels'][0]
        confidence = result['scores'][0]
        
        # Format classification results
        classification_output = f"**Detected Mood:** {predicted_mood.title()}\n"
        classification_output += f"**Confidence:** {confidence:.2%}\n\n"
        
        # Top 3 mood predictions
        classification_output += "**Top Mood Predictions:**\n"
        for i in range(min(3, len(result['labels']))):
            mood = result['labels'][i]
            score = result['scores'][i]
            classification_output += f"• {mood.title()}: {score:.1%}\n"
        
        # Get corresponding playlist
        playlist = mood_playlists.get(predicted_mood, mood_playlists["happy and energetic"])
        
        # Format playlist output
        playlist_output = f"# {playlist['title']}\n\n"
        playlist_output += "**Recommended Songs:**\n"
        
        for i, song in enumerate(playlist['songs'], 1):
            playlist_output += f"{i}. {song}\n"
        
        # Create additional insights
        insights = f"**Mood Insights:**\n"
        insights += f"Based on your input '{mood_text}', our AI detected that you're feeling **{predicted_mood}**.\n\n"
        
        if confidence > 0.7:
            insights += "High confidence detection - this playlist should be a great match!"
        elif confidence > 0.5:
            insights += "Moderate confidence - you might also enjoy songs from similar moods."
        else:
            insights += "Lower confidence - consider trying playlists from multiple mood categories!"
        
        return classification_output, playlist_output, insights
        
    except Exception as e:
        return f"Error analyzing mood: {str(e)}", "", ""

# Create the Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Mood-to-Music Recommender",
    css="""
    .gradio-container {
        max-width: 900px !important;
        margin: auto !important;
    }
    .mood-input {
        font-size: 16px !important;
    }
    """
) as demo:
    
    gr.Markdown("""
    # Mood-to-Music Recommender    
    Describe your current mood in natural language, and let AI recommend the perfect playlist for you! 
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            mood_input = gr.Textbox(
                label="How are you feeling right now?",
                lines=2,
                elem_classes=["mood-input"]
            )
            
            with gr.Row():
                analyze_btn = gr.Button("Analyze My Mood & Get Playlist", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column():
            classification_output = gr.Markdown(label="Mood Analysis")
        
        with gr.Column():
            insights_output = gr.Markdown(label="Insights")
    
    playlist_output = gr.Markdown(label="Your Personalized Playlist")
    
    # Event handlers
    analyze_btn.click(
        fn=analyze,
        inputs=[mood_input],
        outputs=[classification_output, playlist_output, insights_output]
    )

if __name__ == "__main__":
    demo.launch(
        share=True,
        debug=True,
        server_name="0.0.0.0",
        server_port=7860
    )
