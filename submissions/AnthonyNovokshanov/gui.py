import math
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import torch
from transformers import pipeline, infer_device
import scipy
from datasets import load_dataset
import soundfile as sf
import pygame
from pydub import AudioSegment



pygame.mixer.init()

dev=infer_device()

def playSound():
    sound1 = pygame.mixer.Sound("output_quieter.wav")
    sound2 = pygame.mixer.Sound("textspeech.wav")
    sound1.play()
    sound2.play()

def openFile():
    path = filedialog.askopenfilename()
    image = Image.open(path)
    if image.height > 400:
        ratio = math.ceil(image.height/400)
        image = image.resize((int(image.width/ratio), int(image.height/ratio)), Image.LANCZOS)
    tk_image = ImageTk.PhotoImage(image)
    panel.configure(image=tk_image)
    panel.image =tk_image
    label.configure(text="Getting style...")

    #find image style
    image = Image.open(path)
    pipe = pipeline("image-classification", model="prithivMLmods/WikiArt-Style", device=dev)
    style = pipe(image)[0]["label"]
    label.configure(text="Examining the contents of the image...")

    #describe what is happening in the image
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    caption = captioner(image)[0]["generated_text"]
    label.configure(text="Generating the voiceover...")

    #voice over
    voice_over = "This is drawn in a " + style + " style. It is a " + caption
    voice_synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    speech = voice_synthesiser(voice_over, forward_params={"speaker_embeddings": speaker_embedding})
    sf.write("textspeech.wav", speech["audio"], samplerate=speech["sampling_rate"])
    label.configure(text="Creating the music...(May take a while)")

    #text to music
    synthesiser = pipeline("text-to-audio", model="facebook/musicgen-large", device=dev)
    music = synthesiser("Generate a melody in a "+style+" style, such that it would make sense when envisioning a painting of " + caption, forward_params={"do_sample": True})
    scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
    audio = AudioSegment.from_wav("musicgen_out.wav")
    quiet = audio - 10
    quiet.export("output_quieter.wav", "wav")

    label.configure(text='Done! Click the button to "listen" to the painting!')


window = Tk()
window.geometry("800x800")
tk_image = ImageTk.PhotoImage(Image.open('placeholder.webp'))
panel = Label(window, image=tk_image)
panel.pack(side="top", fill="both", expand="yes")

display_text = "Waiting for Uploaded Image"
label = Label(window, text=display_text)
label.pack()

button = Button(text="Upload Image", command=openFile)
button.pack(pady=20)

music_button = Button(text="Listen to the Painting!", command=playSound)
music_button.pack(pady=20)
music_button.pack()

window.mainloop()