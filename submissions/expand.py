import os
from huggingface_hub import InferenceClient

class Expand:
    def __init__(self):
        token = os.getenv('HUGGINGFACE_TOKEN')
        if not token:
            raise ValueError("HUGGINGFACE_TOKEN environment variable not set")
        self.client = InferenceClient(token=token)

    def expand(self, caption: str) -> str:
        prompt = (
            "Based on the following caption, expand on it with more details "
            "and context in a children's book style, keep it under 100 words:\n\n" + caption
        )

        response = self.client.chat_completion(
            model="google/gemma-2-2b-it",
            messages=[{"role": "user", "content": prompt}],
        )

        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content
        return str(response)
