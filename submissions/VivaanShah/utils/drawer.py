import os
from huggingface_hub import InferenceClient

class ImageGenerator:
    def __init__(self, token):
        self.client = InferenceClient(provider="together", api_key=token)
        self.model_name = "black-forest-labs/FLUX.1-dev"  

    def draw(self, description):
        return self.client.text_to_image(description, model=self.model_name)


# if __name__ == "__main__":
#     os.environ["HF_TOKEN"] = "HF_TOKEN"
#     generator = ImageGenerator(os.environ["HF_TOKEN"])
#     image = generator.draw("The fox jumped over the river.")
#     image.show()
