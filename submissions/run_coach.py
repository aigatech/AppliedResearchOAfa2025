#!/usr/bin/env python3
"""
Real-Time Dating Coach - AI@GT Applied Research Team Assessment
Simple launcher script for the RAG-enhanced dating coach

Usage:
    python run_coach.py

This will start the RAG-enhanced dating coach with GUI interface.
"""

import sys
import os

def main():
    """Launch the RAG-enhanced dating coach"""
    print("🚀 Starting Real-Time Dating Coach...")
    print("💡 RAG-Enhanced Version with HuggingFace Integration")
    print("="*60)
    
    try:
        # Import and run the RAG dating coach
        from rag_dating_coach import RAGDatingCoach
        
        print("🤖 Initializing AI models...")
        coach = RAGDatingCoach()
        
        print("✅ Dating Coach ready!")
        print("📱 GUI interface will open shortly...")
        
        # Start the GUI
        coach.run()
        
    except ImportError as e:
        print(f"❌ Error importing modules: {e}")
        print("💡 Please make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting dating coach: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
