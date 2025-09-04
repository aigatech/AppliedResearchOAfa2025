import random
import gradio as gr
from transformers import pipeline

# Hugging Face Pipelines
nlp = pipeline("text2text-generation", model="google/flan-t5-small")

# Emojis
EMOJI_MAP = {
    "sunny": "☀️",
    "cloud": "☁️",
    "rain": "🌧️",
    "fog": "🌫️",
    "wind": "💨",
    "snow": "❄️",
    "storm": "🌩️",
}

# Cleaning Helper
def clean_output(text: str) -> str:
    """Ensure clean single-sentence output ending with a period."""
    text = text.strip()
    if "." in text:
        text = text.split(".")[0].strip()
    return text if text.endswith(".") else text + "."

# NNLP Helper Functions
def generate_summary(city, desc, temp, wind):
    prompt = (
        f"Write ONE short professional news-style sentence about the weather in {city}: "
        f"{desc}, {temp}, {wind}. Keep it under 20 words."
    )
    out = nlp(prompt, max_length=25, num_return_sequences=1, do_sample=False)[0]["generated_text"]
    return clean_output(out)

def generate_insight(desc):
    prompt = (
        f"Give ONE short, useful weather tip for people when it is {desc}. "
        f"Keep it practical (clothing, safety, travel). Max 8 words."
    )
    out = nlp(prompt, max_length=20, num_return_sequences=1, do_sample=False)[0]["generated_text"]
    return clean_output(out)

def generate_headline(desc):
    prompt = f"Write ONE concise 5-word weather headline for: {desc}."
    out = nlp(prompt, max_length=12, num_return_sequences=1, do_sample=False)[0]["generated_text"]
    return clean_output(out)

def pick_emoji(desc):
    for key, symbol in EMOJI_MAP.items():
        if key in desc.lower():
            return symbol
    return "🌍"

# Weather Data (curated & fallback)
def get_weather(city: str):
    curated = {
        "atlanta": {"desc": "Cloudy with light rain", "temp": "22°C", "wind": "12 km/h"},
        "new york": {"desc": "Sunny with clear skies", "temp": "28°C", "wind": "8 km/h"},
        "chicago": {"desc": "Windy with scattered clouds", "temp": "18°C", "wind": "20 km/h"},
        "san francisco": {"desc": "Foggy in the morning, clearing by noon", "temp": "20°C", "wind": "10 km/h"},
    }

    city_lower = city.lower()
    if city_lower in curated:
        return curated[city_lower]

    # fallback simulation
    conditions = [
        "Sunny with clear skies",
        "Cloudy with light rain",
        "Foggy in the morning, clearing by noon",
        "Windy with scattered clouds",
        "Snow showers throughout the afternoon",
        "Thunderstorms expected in the evening",
    ]
    return {
        "desc": random.choice(conditions),
        "temp": f"{random.randint(10, 30)}°C",
        "wind": f"{random.randint(5, 20)} km/h",
    }

# Main Weather Report Function
def ai_weather_report(city: str) -> str:
    if not city.strip():
        return "⚠️ Please enter a valid city name."

    data = get_weather(city)
    desc, temp, wind = data["desc"], data["temp"], data["wind"]

    # NLP outputs
    headline = generate_headline(desc)
    summary = generate_summary(city, desc, temp, wind)
    insight = generate_insight(desc)

    # Emoji
    emoji = pick_emoji(desc)

    # Markdown Output
    return f"""
# {emoji} {headline}
**Weather Report for {city.title()}**  
{summary}  
💡 **Insight**  
{insight}  
🌡️ **Details**  
- Condition: {desc}  
- Temperature: {temp}  
- Wind: {wind}  
"""

# Gradio App
demo = gr.Interface(
    fn=ai_weather_report,
    inputs=gr.Textbox(label="Enter City:"),
    outputs="markdown",
    title="🌐 AI Weather Analyst",
    description="Enter any city to receive a professional AI-generated weather report with headline, summary, insight, and structured details."
)

if __name__ == "__main__":
    demo.launch()
