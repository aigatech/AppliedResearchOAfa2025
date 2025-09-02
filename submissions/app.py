import streamlit as st
import whisper
import librosa
import numpy as np 
from pydub import AudioSegment
import tempfile
import os

@st.cache_resource #loads model once and saves it in memory
def load_whisper_model():
    return whisper.load_whisper_model("tiny")

def extract_samples(audio_file, sample_length = 20):
    audio = AudioSegment.from_file(audio_file)
    total_length = len(audio) / 1000


    