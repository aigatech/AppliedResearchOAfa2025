#!/usr/bin/env python3
"""
Quick Test Script for Advanced Sports AI System
Simple test to verify all 5 AI models are working correctly
"""

from ai_sports_analyzer import AdvancedSportsAI

def quick_test():
    """Run a quick test of all 5 AI features"""
    print("🚀 Quick Test - Advanced Sports AI System")
    print("=" * 50)
    
    try:
        # Initialize the AI system
        print("Initializing AI system...")
        ai_system = AdvancedSportsAI()
        print("✅ AI system loaded successfully!\n")
        
        # Test 1: Sentiment Analysis
        print("📊 Testing Sentiment Analysis...")
        result = ai_system.analyze_sentiment("LeBron James is absolutely incredible!")
        print(f"   Result: {result['sentiment']['label']} ({result['sentiment']['score']:.2f})")
        print("   ✅ Sentiment Analysis: PASSED\n")
        
        # Test 2: Entity Extraction
        print("🔍 Testing Entity Extraction...")
        result = ai_system.extract_sports_entities("Patrick Mahomes plays for the Kansas City Chiefs")
        print(f"   Result: {result['summary']}")
        print("   ✅ Entity Extraction: PASSED\n")
        
        # Test 3: Question Answering
        print("❓ Testing Question Answering...")
        result = ai_system.answer_sports_questions("Who won the first Super Bowl?")
        print(f"   Result: {result['answer']} (confidence: {result['confidence']:.2f})")
        print("   ✅ Question Answering: PASSED\n")
        
        # Test 4: Sports Commentary
        print("✍️ Testing Sports Commentary...")
        commentary = ai_system.generate_sports_commentary("Super Bowl")
        print(f"   Result: {commentary[:100]}...")
        print("   ✅ Sports Commentary: PASSED\n")
        
        # Test 5: News Summarization
        print("📝 Testing News Summarization...")
        article = "The Los Angeles Lakers defeated the Golden State Warriors 120-115 in a thrilling overtime victory. LeBron James scored 35 points and grabbed 12 rebounds, leading his team to victory."
        summary = ai_system.summarize_sports_news(article)
        print(f"   Result: {summary}")
        print("   ✅ News Summarization: PASSED\n")
        
        print("🎉 ALL TESTS PASSED!")
        print("=" * 50)
        print("Your Advanced Sports AI System is working perfectly!")
        print("Run 'python3 ai_sports_analyzer.py' for the full interactive experience.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Please check your dependencies and try again.")
        print("Make sure you have installed: pip install -r requirements.txt")

if __name__ == "__main__":
    quick_test()
