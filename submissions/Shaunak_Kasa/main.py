# main.py
# Multi-Task NLP Demo using Hugging Face
# Author: Shaunak Kasa

from transformers import pipeline

def sentiment_analysis():
    classifier = pipeline("sentiment-analysis")
    user_text = input("Enter text to analyze sentiment: ")
    result = classifier(user_text)[0]
    print(f"Sentiment: {result['label']} (confidence: {round(result['score'], 3)})\n")

def summarization():
    summarizer = pipeline("summarization")
    user_text = input("Enter a passage to summarize: ")
    result = summarizer(user_text, max_length=24, min_length=10, do_sample=False)[0]
    print("Summary:", result['summary_text'], "\n")

def translation():
    print("1 - French")
    print("2 - Spanish")
    user_choice = input("Choose what language to translate to!\n").strip()

    if user_choice == "1":
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
        user_text = input("Enter text in English to translate to French: ")
        result = translator(user_text, max_length=50)[0]
        print("Translation (FR):", result['translation_text'], "\n")

    # Have to install sentencepiece for this to work
    elif user_choice == "2":
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
        user_text = input("Enter text in English to translate to Spanish: ")
        result = translator(user_text, max_length=50)[0]
        print("Translation (ES):", result['translation_text'], "\n")

    else:
        print("Invalid choice. Returning to main menu.\n")

def main():
    print("\n")
    print("Hugging Face Project")
    print("\n")

    while True:
        print("Choose a task:")
        print("1 = Sentiment Analysis")
        print("2 = Summarization")
        print("3 = Translation (EN → FR) or (EN → ES)")
        print("q = Quit")

        choice = input("Enter choice: ").strip()

        if choice == "1":
            sentiment_analysis()
        elif choice == "2":
            summarization()
        elif choice == "3":
            translation()
        elif choice.lower() == "q":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.\n")

if __name__ == "__main__":
    main()
