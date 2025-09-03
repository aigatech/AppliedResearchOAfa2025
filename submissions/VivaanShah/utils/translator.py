from transformers import AutoTokenizer, MarianMTModel
import torch


class Translator:
    def __init__(self, hf_token=None):
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
        
        # get all models
        self.models = {
            "en-hi": {
                "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi"),
                "tokenizer": AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
            },
            "en-fr": {
                "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr"),
                "tokenizer": AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
            },
            "en-de": {
                "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de"),
                "tokenizer": AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
            },
            "en-es": {
                "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es"),
                "tokenizer": AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
            }   
        }
        
    def translate(self, sourceLang, targetLang, text):
        model_key = f"{sourceLang}-{targetLang}"
        
        if model_key not in self.models:
            available_models = list(self.models.keys())
            raise ValueError(f"Model {model_key} not available. Available models: {available_models}")
        
        # get the right model
        model = self.models[model_key]["model"]
        tokenizer = self.models[model_key]["tokenizer"]
        
        batch = tokenizer([text], return_tensors="pt")
        generated_ids = model.generate(**batch)
        
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


if __name__ == "__main__":
    translator = Translator()

    print(translator.translate("en", "fr", "The fox jumped over the river."))
    print(translator.translate("en", "hi", "Hello, how are you?"))
    print(translator.translate("en", "de", "Good morning!"))
    print(translator.translate("en", "es", "The fox jumped over the river."))