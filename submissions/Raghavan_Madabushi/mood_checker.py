#!/usr/bin/env python3
"""
Mood Checker - A simple AI tool to analyze emotional tone in text
Uses HuggingFace transformers for sentiment analysis
"""

from transformers import pipeline
import time

def main():
    print("🎭 Welcome to Mood Checker! 🎭")
    print("I can tell you if your message sounds happy, sad, or neutral!")
    print("=" * 50)
    
    # Load the sentiment analysis model (this might take a moment on first run)
    print("Loading AI model... Please wait...")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    print("✅ Model loaded successfully!")
    print()
    
    while True:
        # Get user input
        user_message = input("💬 Type your message (or 'quit' to exit): ").strip()
        
        if user_message.lower() in ['quit', 'exit', 'q']:
            print("👋 Thanks for using Mood Checker! Goodbye!")
            break
        
        if not user_message:
            print("❌ Please type something!")
            continue
        
        # Analyze the sentiment
        print("🤔 Analyzing your message...")
        result = sentiment_analyzer(user_message)[0]
        
        # Convert sentiment to mood
        sentiment = result['label']
        confidence = result['score']
        
        if sentiment == 'POSITIVE':
            mood = "HAPPY"
            emoji = "😊"
        elif sentiment == 'NEGATIVE':
            mood = "SAD"
            emoji = "😔"
        else:
            mood = "NEUTRAL"
            emoji = "😐"
        
        # Display results
        print(f"{emoji} This message sounds {mood}!")
        print(f"   Confidence: {confidence:.1%}")
        print("-" * 50)
        print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Oops! Something went wrong: {e}")
        print("Make sure you have the required packages installed:")
        print("pip install transformers torch")
