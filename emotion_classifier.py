from transformers import pipeline

def main():
    # Load Hugging Face emotion classifier
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

    print("Simple Emotion Classifier (type 'quit' to exit)")
    while True:
        text = input("Enter a sentence: ")
        if text.lower() == "quit":
            break

        # Run prediction
        result = classifier(text)[0]  # take the top result
        label = result["label"]
        score = round(result["score"], 3)

        print(f"Predicted emotion: {label} (confidence {score})\n")

if __name__ == "__main__":
    main()
