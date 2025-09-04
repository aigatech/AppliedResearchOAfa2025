"""
Real-Time Dating Coach - AI@GT Applied Research Team Submission
A real-time conversation coach that analyzes chat sentiment and provides respectful reply suggestions.
"""

import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
from datetime import datetime
from typing import List, Dict, Tuple
import json
import random
from datasets import load_dataset

class DatingCoach:
    def __init__(self):
        # Initialize Hugging Face models
        print("Loading AI models...")
        self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                         model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.emotion_analyzer = pipeline("text-classification", 
                                       model="j-hartmann/emotion-english-distilroberta-base")
        
        # Load the rizz corpus for better conversation training
        print("Loading rizz corpus dataset...")
        try:
            self.rizz_dataset = load_dataset('the-rizz/the-rizz-corpus')
            self.rizz_examples = self.rizz_dataset['train']['text']
            print(f"Loaded {len(self.rizz_examples)} conversation examples")
        except Exception as e:
            print(f"Could not load rizz corpus: {e}")
            self.rizz_examples = []
        
        # Initialize conversation state
        self.conversation_history = []
        self.analysis_results = []
        self.suggestions_history = []
        
        # Updated respectful responses with lowercase, conversational style
        self.respectful_responses = {
            'positive': [
                "that's really cool! tell me more about {topic}",
                "i love that! what got you into {topic}?",
                "that sounds amazing! how did you discover {topic}?",
                "no way, that's awesome! i'd love to hear more about {topic}",
                "that's so interesting! what's your favorite thing about {topic}?"
            ],
            'neutral': [
                "that's cool! what do you think about {topic}?",
                "interesting! how do you feel about {topic}?",
                "i'd love to hear more about your thoughts on {topic}",
                "tell me more about that",
                "what's that like for you?"
            ],
            'negative': [
                "i understand that might be tough to talk about",
                "thanks for sharing that with me",
                "i appreciate you being open about that",
                "that sounds challenging",
                "i hear you on that"
            ],
            'flirty_positive': [
                "you're making me curious about {topic} now 😊",
                "i love how passionate you are about {topic}",
                "you have great taste! what else are you into?",
                "that's really attractive, someone who knows what they like",
                "you're full of surprises! what other hidden talents do you have?"
            ],
            'playful': [
                "okay now you're just showing off 😏",
                "are you trying to impress me? because it's working",
                "you're making this conversation way too interesting",
                "i'm starting to think you're cooler than you let on",
                "well now i definitely need to know more about you"
            ]
        }
        
        self.boundary_keywords = ['stop', 'uncomfortable', 'not interested', 'leave me alone', 
                                'busy', 'can\'t talk', 'not now', 'please stop']
        
        # GUI setup
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI interface"""
        self.root = tk.Tk()
        self.root.title("Real-Time Dating Coach - Respectful Conversation Assistant")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Real-Time Dating Coach", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Conversation input area
        ttk.Label(main_frame, text="Conversation (paste messages here):").grid(row=1, column=0, sticky=tk.W)
        self.conversation_text = scrolledtext.ScrolledText(main_frame, height=10, width=60)
        self.conversation_text.grid(row=2, column=0, padx=(0, 10), pady=(0, 10))
        
        # Analysis and suggestions area
        ttk.Label(main_frame, text="AI Analysis & Suggestions:").grid(row=1, column=1, sticky=tk.W)
        self.suggestions_text = scrolledtext.ScrolledText(main_frame, height=10, width=60, 
                                                         bg='#e8f4f8')
        self.suggestions_text.grid(row=2, column=1, pady=(0, 10))
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.analyze_btn = ttk.Button(button_frame, text="Analyze Conversation", 
                                     command=self.analyze_conversation)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(button_frame, text="Clear All", 
                                   command=self.clear_all)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(button_frame, text="Save Analysis", 
                                  command=self.save_analysis)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Status area
        self.status_label = ttk.Label(main_frame, text="Ready to analyze conversations", 
                                     foreground='green')
        self.status_label.grid(row=4, column=0, columnspan=2, pady=5)
        
        # Auto-analysis toggle
        self.auto_analyze_var = tk.BooleanVar()
        auto_check = ttk.Checkbutton(main_frame, text="Auto-analyze on text change", 
                                    variable=self.auto_analyze_var)
        auto_check.grid(row=5, column=0, columnspan=2, pady=5)
        
        # Bind text change event for real-time analysis
        self.conversation_text.bind('<KeyRelease>', self.on_text_change)
        
    def on_text_change(self, event=None):
        """Handle text changes for real-time analysis"""
        if self.auto_analyze_var.get():
            # Debounce - only analyze after 2 seconds of no typing
            if hasattr(self, 'analyze_timer'):
                self.analyze_timer.cancel()
            self.analyze_timer = threading.Timer(2.0, self.analyze_conversation)
            self.analyze_timer.start()
    
    def extract_messages(self, text: str) -> List[str]:
        """Extract individual messages from conversation text"""
        # Split by common message separators
        messages = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 2:  # Filter out very short lines
                # Remove timestamps and usernames if present
                cleaned = re.sub(r'^\d{1,2}:\d{2}.*?:', '', line)
                cleaned = re.sub(r'^[A-Za-z\s]+:', '', cleaned)
                cleaned = cleaned.strip()
                if cleaned:
                    messages.append(cleaned)
        
        return messages[-10:]  # Last 10 messages for context
    
    def detect_boundaries(self, messages: List[str]) -> bool:
        """Detect if conversation partner is setting boundaries"""
        recent_text = ' '.join(messages[-3:]).lower()  # Last 3 messages
        
        for keyword in self.boundary_keywords:
            if keyword in recent_text:
                return True
        
        return False
    
    def extract_rizz_wisdom(self, context: str, sentiment: str) -> List[str]:
        """Extract relevant conversation examples from the rizz corpus"""
        if not self.rizz_examples:
            return []
        
        # Look for examples that match the current context/sentiment
        relevant_examples = []
        context_lower = context.lower()
        
        # Sample some examples and filter for relevance
        sample_size = min(100, len(self.rizz_examples))
        sampled_examples = random.sample(self.rizz_examples, sample_size)
        
        for example in sampled_examples:
            example_lower = example.lower()
            # Look for examples that contain similar themes or are appropriate for the sentiment
            if any(word in example_lower for word in ['witty', 'funny', 'light', 'kind', 'conversation']):
                # Extract conversational parts (remove system prompts)
                clean_example = re.sub(r'<<SYS>>.*?<</SYS>>', '', example)
                clean_example = clean_example.strip()
                
                if clean_example and len(clean_example) > 20 and len(clean_example) < 200:
                    # Make it lowercase and conversational
                    conversational = clean_example.lower()
                    # Remove any remaining formatting
                    conversational = re.sub(r'[<>]', '', conversational)
                    if conversational not in relevant_examples:
                        relevant_examples.append(conversational)
                    
                    if len(relevant_examples) >= 3:
                        break
        
        return relevant_examples[:3]
    
    def analyze_sentiment_and_emotion(self, messages: List[str]) -> Dict:
        """Analyze sentiment and emotion of recent messages"""
        if not messages:
            return {'sentiment': 'neutral', 'emotion': 'neutral', 'engagement': 'low'}
        
        recent_text = ' '.join(messages[-3:])  # Analyze last 3 messages
        
        # Sentiment analysis
        sentiment_result = self.sentiment_analyzer(recent_text[:512])  # Truncate if too long
        sentiment = sentiment_result[0]['label']
        sentiment_score = sentiment_result[0]['score']
        
        # Emotion analysis
        emotion_result = self.emotion_analyzer(recent_text[:512])
        emotion = emotion_result[0]['label']
        emotion_score = emotion_result[0]['score']
        
        # Engagement analysis based on message length and frequency
        avg_length = sum(len(msg) for msg in messages) / len(messages)
        engagement = 'high' if avg_length > 20 else 'medium' if avg_length > 10 else 'low'
        
        return {
            'sentiment': sentiment.lower(),
            'sentiment_score': sentiment_score,
            'emotion': emotion.lower(),
            'emotion_score': emotion_score,
            'engagement': engagement,
            'avg_message_length': avg_length
        }
    
    def generate_suggestions(self, analysis: Dict, messages: List[str]) -> List[str]:
        """Generate respectful, conversational suggestions with rizz corpus integration"""
        suggestions = []
        
        # Check for boundaries first
        if self.detect_boundaries(messages):
            suggestions.append("⚠️ boundary detected: consider giving them space or asking if they'd prefer to chat later")
            suggestions.append("respectful response: 'no worries at all! feel free to reach out whenever you'd like to chat'")
            return suggestions
        
        # Extract potential topics from recent messages
        recent_text = ' '.join(messages[-2:]) if len(messages) >= 2 else messages[-1] if messages else ""
        
        # Get rizz corpus wisdom for context
        rizz_examples = self.extract_rizz_wisdom(recent_text, analysis['sentiment'])
        
        # Generate contextual suggestions based on sentiment
        sentiment = analysis['sentiment']
        emotion = analysis['emotion']
        
        # Determine conversation style based on emotion and engagement
        if analysis['engagement'] == 'high' and sentiment in ['positive', 'joy']:
            style = 'flirty_positive'
        elif emotion in ['joy', 'surprise'] and analysis['engagement'] in ['medium', 'high']:
            style = 'playful' 
        elif sentiment in ['positive', 'joy', 'optimism']:
            style = 'positive'
        elif sentiment in ['negative', 'sadness', 'pessimism']:
            style = 'negative'
        else:
            style = 'neutral'
        
        # Get appropriate response templates
        if style in self.respectful_responses:
            response_templates = self.respectful_responses[style]
            
            # Add 2-3 template-based suggestions
            for i, template in enumerate(response_templates[:3]):
                suggestion = template.format(topic='that')
                suggestions.append(f"💬 {suggestion}")
        
        # Add rizz corpus inspired suggestions
        if rizz_examples:
            suggestions.append("📚 rizz corpus inspiration:")
            for example in rizz_examples[:2]:
                # Clean up and make it a suggestion
                clean_example = example.strip()
                if clean_example and not clean_example.startswith('you are'):
                    suggestions.append(f"   • {clean_example}")
        
        # Add engagement-based suggestions with lowercase style
        if analysis['engagement'] == 'low':
            suggestions.append("💡 low engagement detected - maybe ask about their interests or share something fun")
        elif analysis['engagement'] == 'high':
            suggestions.append("💡 high engagement! great flow - keep the energy up but stay natural")
        
        # Add conversation flow suggestions
        if len(messages) > 5:
            suggestions.append("🔄 conversation flowing well - maybe suggest moving to a different topic or activity")
        
        # Add playful/flirty elements if appropriate
        if style in ['flirty_positive', 'playful'] and analysis['engagement'] == 'high':
            playful_suggestions = [
                "😏 they seem really engaged - maybe add some light teasing or humor",
                "✨ good energy here - you could be a bit more playful",
                "🎯 they're responding well - don't be afraid to show your personality"
            ]
            suggestions.append(random.choice(playful_suggestions))
        
        return suggestions[:6]  # Limit to 6 suggestions
    
    def analyze_conversation(self):
        """Main analysis function"""
        self.status_label.config(text="Analyzing conversation...", foreground='orange')
        self.root.update()
        
        try:
            # Get conversation text
            conv_text = self.conversation_text.get('1.0', tk.END)
            messages = self.extract_messages(conv_text)
            
            if not messages:
                self.suggestions_text.delete('1.0', tk.END)
                self.suggestions_text.insert(tk.END, "No messages to analyze. Please paste some conversation text above.")
                self.status_label.config(text="No messages found", foreground='red')
                return
            
            # Perform analysis
            analysis = self.analyze_sentiment_and_emotion(messages)
            suggestions = self.generate_suggestions(analysis, messages)
            
            # Display results
            self.display_analysis(analysis, suggestions, messages)
            
            # Store results
            self.analysis_results.append({
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis,
                'suggestions': suggestions,
                'message_count': len(messages)
            })
            
            self.status_label.config(text="Analysis complete!", foreground='green')
            
        except Exception as e:
            self.suggestions_text.delete('1.0', tk.END)
            self.suggestions_text.insert(tk.END, f"Error during analysis: {str(e)}")
            self.status_label.config(text="Analysis failed", foreground='red')
    
    def display_analysis(self, analysis: Dict, suggestions: List[str], messages: List[str]):
        """Display analysis results in the GUI with conversational lowercase style"""
        self.suggestions_text.delete('1.0', tk.END)
        
        # Analysis summary with conversational tone
        self.suggestions_text.insert(tk.END, "🔍 conversation analysis\n")
        self.suggestions_text.insert(tk.END, "="*40 + "\n\n")
        
        self.suggestions_text.insert(tk.END, f"sentiment: {analysis['sentiment'].title()} ({analysis['sentiment_score']:.2f})\n")
        self.suggestions_text.insert(tk.END, f"emotion: {analysis['emotion'].title()} ({analysis['emotion_score']:.2f})\n")
        self.suggestions_text.insert(tk.END, f"engagement: {analysis['engagement'].title()}\n")
        self.suggestions_text.insert(tk.END, f"analyzed {len(messages)} recent messages\n\n")
        
        # Suggestions with conversational style
        self.suggestions_text.insert(tk.END, "💡 conversational suggestions\n")
        self.suggestions_text.insert(tk.END, "="*40 + "\n\n")
        
        for i, suggestion in enumerate(suggestions, 1):
            self.suggestions_text.insert(tk.END, f"{suggestion}\n\n")
        
        # General tips with lowercase conversational style
        self.suggestions_text.insert(tk.END, "📋 general vibes\n")
        self.suggestions_text.insert(tk.END, "="*40 + "\n")
        self.suggestions_text.insert(tk.END, "• always respect boundaries and consent\n")
        self.suggestions_text.insert(tk.END, "• listen actively and show genuine interest\n")
        self.suggestions_text.insert(tk.END, "• be authentic - use these as inspiration, not scripts\n")
        self.suggestions_text.insert(tk.END, "• when in doubt, ask open-ended questions\n")
        self.suggestions_text.insert(tk.END, "• keep it playful but respectful\n")
        self.suggestions_text.insert(tk.END, "• let your personality shine through\n")
    
    def clear_all(self):
        """Clear all text areas"""
        self.conversation_text.delete('1.0', tk.END)
        self.suggestions_text.delete('1.0', tk.END)
        self.status_label.config(text="Ready to analyze conversations", foreground='green')
    
    def save_analysis(self):
        """Save analysis results to file"""
        if not self.analysis_results:
            self.status_label.config(text="No analysis to save", foreground='red')
            return
        
        filename = f"dating_coach_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.analysis_results, f, indent=2)
            self.status_label.config(text=f"Analysis saved to {filename}", foreground='green')
        except Exception as e:
            self.status_label.config(text=f"Save failed: {str(e)}", foreground='red')
    
    def run(self):
        """Start the application"""
        print("Starting Real-Time Dating Coach...")
        print("Remember: This tool is designed to help with respectful, genuine conversation!")
        self.root.mainloop()

if __name__ == "__main__":
    # Create and run the dating coach app
    coach = DatingCoach()
    coach.run()
