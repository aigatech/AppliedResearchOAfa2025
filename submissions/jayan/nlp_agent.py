# nlp_agent.py
from transformers import pipeline, Pipeline
import warnings

class CancerInfoAgent:
    def __init__(self, model_name: str = "google/flan-t5-large"):
        """
        Initialize the text generation pipeline for cancer information.
        model_name: HF model to use (default: flan-t5-small)
        """
        try:
            # Suppress pipeline warnings
            warnings.filterwarnings("ignore")
            self.generator: Pipeline = pipeline(
                "text2text-generation",
                model=model_name,
                device=-1  # CPU by default; set 0 for GPU
            )
        except Exception as e:
            print(f"Error initializing NLP pipeline: {e}")
            self.generator = None

    def get_cancer_info(self, cancer_type: str) -> str:
        """
        Return a concise explanation of a cancer type.
        Includes typical risk factors, common treatments, and disclaimer.
        """
        if self.generator is None:
            return "(NLP pipeline not available)"

        prompt = (
            f"Explain in medical terms what {cancer_type} cancer is. "
            "List typical risk factors and common treatment approaches. "
            "Include a short disclaimer 'not medical advice'."
        )

        try:
            # Use max_new_tokens to avoid warning about max_length
            out = self.generator(
                prompt,
                max_new_tokens=250,
                num_return_sequences=1
            )
            text = out[0]["generated_text"].strip()

            # Ensure disclaimer is included
            if "not medical advice" not in text.lower():
                text += " (Disclaimer: Not medical advice.)"

            return text

        except Exception as e:
            return f"(NLP generation failed: {e})"
