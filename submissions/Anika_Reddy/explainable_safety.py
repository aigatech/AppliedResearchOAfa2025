import argparse
from PIL import Image
from transformers import pipeline

#Load all the models
# Object detection with YOLOS
detector = pipeline("object-detection", model="hustvl/yolos-small")

# Flan-T5 explainer
explainer = pipeline("text2text-generation", model="google/flan-t5-base")


# function to detect objects and check for person
def analyze_image(img_path):
    image = Image.open(img_path).convert("RGB")
    detections = detector(image)

    labels = [d["label"].lower() for d in detections]
    print("Detected objects:", labels) 

    if "person" in labels:
        safety_label = "unsafe"
    else:
        safety_label = "safe"

    return safety_label, labels


#Function to generate explanations for the predictions
def generate_explanation(safety_label, labels):
    base_prompt = f"The objects detected are: {', '.join(labels)}. "

    if safety_label == "unsafe":
        prompt = base_prompt + "Since a person was detected, classify this environment as unsafe and explain why in one sentence."
    else:
        prompt = base_prompt + "No person was detected, so classify this environment as safe and explain why in one sentence."

    response = explainer(
        prompt,
        max_length=60,
        do_sample=True,
        temperature=0.7,
        num_return_sequences=1,
        early_stopping=True
    )

    return response[0]["generated_text"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    args = parser.parse_args()

    # Analyze with YOLOS
    safety_label, labels = analyze_image(args.image)

    # Generate explanation with Flan-T5
    explanation = generate_explanation(safety_label, labels)

    # Print results
    print(f"\nPrediction: {safety_label.upper()}")
    print(f"Explanation: {explanation}")
