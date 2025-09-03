import gradio as gr
from transformers import pipeline
import numpy as np 
from pydub import AudioSegment
import tempfile
import os

def load_models():
    # return whisper.load_whisper_model("tiny")
    models = {}

    models["fin_sent"] = pipeline("sentiment-analysis", model="ProsusAI/finbert", return_all_scores = True)
    models["audio_emotion"] = pipeline("audio-classification", model = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    models["transcriber"] = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")


    return models

def extract_sample(audio_file, sample_length):
    audio = AudioSegment.from_file(audio_file)
    total_length = len(audio) / 1000

    start_time = (total_length/2) - sample_length
    if start_time < 0:
        start_time = 0
    
    clip = audio[start_time * 1000:(start_time + sample_length) *1000]
    return clip, start_time

def transcribe_audio(audio_clip, model):
    # temporarily converts to wav file for model analyss
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp:
        audio_clip.export(temp.name, format="wav")
        result = model(temp.name)

        os.unlink(temp.name) #deletes temp file

        return result['text'].strip()
    
def analyze_emotion(audio_clip, model):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp:
        audio_clip.export(temp.name, format="wav")

        emotions = model(temp.name) #voice emotion data
        os.unlink(temp.name)

        '''
        Categorize pos/neg emotions
        higher numbers mean greater deception likelihood
        '''

        negative_emotions = {'angry': 0.7, 'fearful': 0.9, 'surprised': 0.2, 'sad': 0.7, 'disgusted': 0.5}
        positive_emotions = {'happy': -0.5, 'calm': -0.1, 'neutral': 0.0}

        emotional_score = 0.0
        max_emotion = emotions[0]['label'].lower()

        #calculate emotional score based on emotions found in audio
        for emotion in emotions:
            e = emotion['label'].lower()
            score = emotion['score']

            if e in negative_emotions:
                emotional_score += score * negative_emotions[e] * 100
            elif e in positive_emotions:
                emotional_score += score * positive_emotions[e] * 100
            else: 
                emotional_score += score * 25 #default average level stress

            if emotional_score > 100:
                emotional_score = 100

        return emotional_score, max_emotion, emotions
    
def analyze_sentiment(text, model):

    fin_sentiment = model(text)

    sentiment_score = 0
    sentiment_analysis = {}

    for result in fin_sentiment[0]:
        r = result['label']
        score = result['score']
        sentiment_analysis[r] = score

        if r =='negative':
            sentiment_score += score * 70
        elif r == 'neutral':
            sentiment_score += score * 25
        else:
            sentiment_score += score * 5

    return sentiment_score, fin_sentiment, sentiment_analysis

# #analyze voice pitch
# def analyze_stress(audio_clip):
#     with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp:
#         audio_clip.export(temp.name, format="wav")

def analyze_whole(audio_file):

    models = load_models()
    audio_clip, start_time = extract_sample(audio_file, 20)
    text = transcribe_audio(audio_clip, models["transcriber"])
    emotional_score, max_emotion, emotion_details = analyze_emotion(audio_clip, models["audio_emotion"])
    sentiment_score, fin_sentiment, sentiment_analysis = analyze_sentiment(text, models["fin_sent"])

    total_score = (
        sentiment_score * 0.5 + 
        emotional_score * 0.5
    )

    if total_score > 50:
        risk = "HIGH RISK"
    elif total_score > 27:
        risk = "Moderate Risk"
    elif total_score > 10:
        risk = "Low Risk"
    else:
        risk = "no/minimal risk"

    results = f"""
# RESULTS
Uncertainty/Deception Risk Score: {total_score:.0f}/100** - {risk}

Strongest voice emotion: {max_emotion}

Analysis:
{emotion_details}

Text sentiment: {fin_sentiment}

Analysis:
{sentiment_analysis}


"""
    
    return results
    
def create_interface():
    with gr.Blocks(title="Earnings Call Uncertainty Detector", theme=gr.themes.Soft()) as demo:
        gr.Markdown("Earnings Call Uncertainty Detector")
        gr.Markdown("Upload an earnings call audio file to analyze for deception markers using AI models.")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath",
                    sources=["upload"]
                )
                analyze_btn = gr.Button("🔍 Analyze", variant="primary")
            
            with gr.Column():
                results_output = gr.Markdown(
                    label="Analysis Results",
                    value="Upload file and click Analyze for results"
                )
        
        analyze_btn.click(
            fn=analyze_whole,
            inputs=[audio_input],
            outputs=[results_output]
        )
        
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
    




    





        


 

    