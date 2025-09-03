from transformers import VitsModel, VitsTokenizer
import torch
import soundfile as sf

class Speaker:
    def __init__(self, hf_token=None):
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
        
        # get all language models
        self.models = {
            "spa": { 
                "model": VitsModel.from_pretrained("facebook/mms-tts-spa"),
                "tokenizer": VitsTokenizer.from_pretrained("facebook/mms-tts-spa")
            },
            "hin": { 
                "model": VitsModel.from_pretrained("facebook/mms-tts-hin"),
                "tokenizer": VitsTokenizer.from_pretrained("facebook/mms-tts-hin")
            },
            "fra": {  
                "model": VitsModel.from_pretrained("facebook/mms-tts-fra"),
                "tokenizer": VitsTokenizer.from_pretrained("facebook/mms-tts-fra")
            },
            "deu": {  
                "model": VitsModel.from_pretrained("facebook/mms-tts-deu"),
                "tokenizer": VitsTokenizer.from_pretrained("facebook/mms-tts-deu")
            }
        }
        
    def speak(self, language, text, output_file="output.wav"):
        if language not in self.models:
            available_languages = list(self.models.keys())
            raise ValueError(f"Language '{language}' not available. Available languages: {available_languages}")
        
        #get the right model 
        model = self.models[language]["model"]
        tokenizer = self.models[language]["tokenizer"]
        

        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        # random error fix
        input_ids = input_ids.long()
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        speech = outputs["waveform"]
        
        # save audio
        sf.write(output_file, speech.squeeze().numpy(), 16000)
        
        return speech


if __name__ == "__main__":

    speaker = Speaker()
    

    text = "Hello, how are you?"

    speaker.speak("deu", text, "output.wav")