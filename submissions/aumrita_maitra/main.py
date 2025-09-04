from transformers import pipeline, AutoTokenizer
import math

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
NEUTRAL_THRESHOLD = 0.998

def print_result(text, label, score):
    emoticons = {
        "POSITIVE": ":)",
        "NEGATIVE": ":(",
        "NEUTRAL": ":|"
    }

    score_display = round(score, 3)
    print(f"Entry: {text}\nSentiment: {label} {emoticons[label]} (Confidence: {score_display})\n")

def labels(hf_label, hf_score, threshold=NEUTRAL_THRESHOLD):
    if hf_score < threshold: #add neutral threshold
        return "NEUTRAL", hf_score
    return hf_label, hf_score

def main():
    classifier = pipeline("sentiment-analysis", model=MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    entries = [
        "I love programming!",
        "This is a terrible mistake.",
        "I'm not sure how I feel about this."
    ]

    print("Sentiment Analysis Results:")
    print("="*60)

    for text in entries:
        result = classifier(text)[0]
        mapped_label, mapped_score = labels(result['label'], result['score'])
        print_result(text, mapped_label, mapped_score)

    #allow user to type
    print("Type a journal entry(sentence) and press 'enter' for sentiment analysis, or '0' to quit:")
    print("="*60)

    while True:
        user_input = input("> ").strip()
        if user_input == "0":
            print("hope you have a nice day, bye bye!")
            break
        if user_input == "":
            continue

        #running sentiment analysis
        result = classifier(user_input)[0]
        mapped_label, mapped_score = labels(result['label'], result['score'])
        print_result(user_input, mapped_label, mapped_score)

if __name__ == "__main__":
    main()