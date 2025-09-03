#!/usr/bin/env python3
"""
Quick Test Script for Advanced Sports AI System
Simple test to verify all 5 AI models are working correctly
"""

from ai_sports_analyzer import AdvancedSportsAI

def quick_test():
    print("Quick Test - Advanced Sports AI System")
    print("=" * 50)
    
    try:
        # Initialize the AI system
        print("Initializing AI system...")
        ai_system = AdvancedSportsAI()
        
        result = ai_system.analyze_sentiment("LeBron James is absolutely incredible!")
    
        result = ai_system.extract_sports_entities("Patrick Mahomes plays for the Kansas City Chiefs")
        
        result = ai_system.answer_sports_questions("Who won the first Super Bowl?")
        
        commentary = ai_system.generate_sports_commentary("Super Bowl")
    
        article = "The Los Angeles Lakers defeated the Golden State Warriors 120-115 in a thrilling overtime victory. LeBron James scored 35 points and grabbed 12 rebounds, leading his team to victory."
        summary = ai_system.summarize_sports_news(article)
       
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("Please check your dependencies and try again.")
        print("Make sure you have installed: pip install -r requirements.txt")

if __name__ == "__main__":
    quick_test()
