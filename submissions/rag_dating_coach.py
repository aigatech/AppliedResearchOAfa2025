"""
Enhanced Real-Time Dating Coach with RAG and Embedding-based Generation
Uses retrieval-augmented generation with the rizz corpus for dynamic suggestions
"""

import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json
import random
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
from dataclasses import dataclass

@dataclass
class ConversationContext:
    """Structured context for conversation analysis"""
    messages: List[str]
    sentiment: str
    emotion: str
    engagement_level: str
    topics: List[str]
    conversation_stage: str
    boundary_signals: bool

class RAGDatingCoach:
    def __init__(self):
        # Initialize Hugging Face models
        print("Loading AI models...")
        self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                         model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.emotion_analyzer = pipeline("text-classification", 
                                       model="j-hartmann/emotion-english-distilroberta-base")
        
        # Initialize embedding model for RAG
        print("Loading embedding model for RAG...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load and process the rizz corpus for RAG
        print("Loading and processing rizz corpus for RAG...")
        self.setup_rag_system()
        
        # Initialize conversation state
        self.conversation_history = []
        self.analysis_results = []
        self.suggestions_history = []
        
        # Safety filters and respectful response templates (fallback)
        self.boundary_keywords = ['stop', 'uncomfortable', 'not interested', 'leave me alone', 
                                'busy', 'can\'t talk', 'not now', 'please stop', 'don\'t want to',
                                'feeling pressured', 'need space', 'not in the mood']
        
        # GUI setup
        self.setup_gui()
    
    def setup_rag_system(self):
        """Setup the RAG system with rizz corpus embeddings"""
        try:
            # Load the rizz corpus
            self.rizz_dataset = load_dataset('the-rizz/the-rizz-corpus')
            raw_examples = self.rizz_dataset['train']['text']
            
            print("Processing conversation examples for RAG...")
            
            # Clean and filter the corpus
            self.processed_examples = []
            self.example_metadata = []
            
            for i, example in enumerate(raw_examples):
                # Clean the example
                cleaned = self.clean_conversation_example(example)
                if cleaned and self.is_good_example(cleaned):
                    self.processed_examples.append(cleaned)
                    
                    # Extract metadata
                    metadata = self.extract_example_metadata(cleaned, example)
                    self.example_metadata.append(metadata)
                    
                    if len(self.processed_examples) >= 5000:  # Limit for performance
                        break
            
            print(f"Processed {len(self.processed_examples)} examples for RAG")
            
            # Create embeddings for the corpus
            print("Creating embeddings for conversation corpus...")
            self.corpus_embeddings = self.embedding_model.encode(self.processed_examples)
            
            # Setup FAISS index for fast similarity search
            self.setup_faiss_index()
            
            print("RAG system ready!")
            
        except Exception as e:
            print(f"Error setting up RAG system: {e}")
            self.processed_examples = []
            self.corpus_embeddings = None
            self.faiss_index = None
    
    def clean_conversation_example(self, example: str) -> Optional[str]:
        """Clean and extract useful conversation parts from corpus examples"""
        # Remove system prompts
        cleaned = re.sub(r'<<SYS>>.*?<</SYS>>', '', example, flags=re.DOTALL)
        
        # Remove special tokens
        cleaned = re.sub(r'</?s>|\[/?INST\]', '', cleaned)
        
        # Clean up whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove very short or very long examples
        if len(cleaned) < 10 or len(cleaned) > 300:
            return None
            
        return cleaned.lower()
    
    def is_good_example(self, example: str) -> bool:
        """Filter for high-quality conversation examples"""
        # Check for positive conversation markers
        positive_markers = ['funny', 'witty', 'light', 'kind', 'interesting', 'amazing', 
                          'love', 'awesome', 'great', 'wonderful', 'exciting']
        
        # Check for negative markers to avoid
        negative_markers = ['rude', 'mean', 'aggressive', 'inappropriate', 'offensive']
        
        example_lower = example.lower()
        
        # Must have some positive markers and no negative ones
        has_positive = any(marker in example_lower for marker in positive_markers)
        has_negative = any(marker in example_lower for marker in negative_markers)
        
        return has_positive and not has_negative and len(example.split()) >= 3
    
    def extract_example_metadata(self, cleaned_example: str, original: str) -> Dict:
        """Extract metadata from conversation examples"""
        return {
            'length': len(cleaned_example),
            'word_count': len(cleaned_example.split()),
            'has_question': '?' in cleaned_example,
            'has_emoji': any(char in cleaned_example for char in '😊😏😍🔥💕'),
            'energy_level': self.estimate_energy_level(cleaned_example),
            'conversation_type': self.classify_conversation_type(cleaned_example)
        }
    
    def estimate_energy_level(self, text: str) -> str:
        """Estimate the energy level of a conversation example"""
        high_energy_words = ['amazing', 'awesome', 'incredible', 'fantastic', 'wow', '!']
        medium_energy_words = ['cool', 'nice', 'good', 'interesting']
        
        text_lower = text.lower()
        high_count = sum(1 for word in high_energy_words if word in text_lower)
        medium_count = sum(1 for word in medium_energy_words if word in text_lower)
        
        if high_count >= 2 or '!' in text:
            return 'high'
        elif medium_count >= 1 or high_count >= 1:
            return 'medium'
        else:
            return 'low'
    
    def classify_conversation_type(self, text: str) -> str:
        """Classify the type of conversation"""
        if any(word in text.lower() for word in ['flirt', 'cute', 'beautiful', 'attractive']):
            return 'flirty'
        elif any(word in text.lower() for word in ['funny', 'joke', 'laugh', 'hilarious']):
            return 'humorous'
        elif any(word in text.lower() for word in ['interest', 'hobby', 'passion', 'love doing']):
            return 'interest_based'
        else:
            return 'general'
    
    def setup_faiss_index(self):
        """Setup FAISS index for fast similarity search"""
        if self.corpus_embeddings is not None and len(self.corpus_embeddings) > 0:
            # Create FAISS index
            dimension = self.corpus_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.corpus_embeddings)
            self.faiss_index.add(self.corpus_embeddings.astype('float32'))
            
            print(f"FAISS index created with {self.faiss_index.ntotal} examples")
        else:
            self.faiss_index = None
    
    def retrieve_relevant_examples(self, query: str, context: ConversationContext, k: int = 5) -> List[Dict]:
        """Retrieve relevant examples using RAG"""
        if self.faiss_index is None or not self.processed_examples:
            return []
        
        try:
            # Create enhanced query with context
            enhanced_query = self.create_enhanced_query(query, context)
            
            # Get query embedding
            query_embedding = self.embedding_model.encode([enhanced_query])
            faiss.normalize_L2(query_embedding)
            
            # Search for similar examples
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
            
            # Retrieve examples with metadata
            retrieved_examples = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.processed_examples):
                    example = {
                        'text': self.processed_examples[idx],
                        'score': float(score),
                        'metadata': self.example_metadata[idx] if idx < len(self.example_metadata) else {}
                    }
                    retrieved_examples.append(example)
            
            # Filter examples based on context
            filtered_examples = self.filter_examples_by_context(retrieved_examples, context)
            
            return filtered_examples
            
        except Exception as e:
            print(f"Error in RAG retrieval: {e}")
            return []
    
    def create_enhanced_query(self, original_query: str, context: ConversationContext) -> str:
        """Create an enhanced query with conversation context"""
        # Add context information to the query
        context_parts = [original_query]
        
        if context.sentiment:
            context_parts.append(f"sentiment: {context.sentiment}")
        
        if context.emotion:
            context_parts.append(f"emotion: {context.emotion}")
        
        if context.topics:
            context_parts.append(f"topics: {', '.join(context.topics)}")
        
        if context.engagement_level:
            context_parts.append(f"engagement: {context.engagement_level}")
        
        return " ".join(context_parts)
    
    def filter_examples_by_context(self, examples: List[Dict], context: ConversationContext) -> List[Dict]:
        """Filter retrieved examples based on conversation context"""
        filtered = []
        
        for example in examples:
            metadata = example['metadata']
            
            # Filter based on energy level matching
            if context.engagement_level == 'high' and metadata.get('energy_level', 'low') == 'low':
                continue
            
            # Filter based on conversation stage
            if context.boundary_signals:
                continue  # Skip examples when boundaries are detected
            
            # Prefer examples with similar characteristics
            if context.engagement_level == 'high' and metadata.get('energy_level') == 'high':
                example['score'] += 0.1  # Boost score for matching energy
            
            filtered.append(example)
        
        # Sort by score and return top examples
        filtered.sort(key=lambda x: x['score'], reverse=True)
        return filtered[:3]  # Return top 3
    
    def generate_rag_suggestions(self, context: ConversationContext) -> List[str]:
        """Generate suggestions using RAG with the conversation context"""
        if not context.messages:
            return ["start with a friendly greeting!"]
        
        # Create query from recent messages
        recent_text = ' '.join(context.messages[-2:])
        
        # Retrieve relevant examples
        relevant_examples = self.retrieve_relevant_examples(recent_text, context)
        
        suggestions = []
        
        # Handle boundary detection first
        if context.boundary_signals:
            suggestions.append("🚨 boundary detected: give them space and respect their comfort level")
            suggestions.append("💬 try: 'no worries at all! feel free to reach out whenever you'd like to chat'")
            return suggestions
        
        # Generate suggestions based on retrieved examples
        if relevant_examples:
            suggestions.append("🤖 rag-generated suggestions based on similar conversations:")
            
            for i, example in enumerate(relevant_examples[:3], 1):
                # Adapt the example to current context
                adapted_suggestion = self.adapt_example_to_context(example['text'], context)
                suggestions.append(f"💬 option {i}: {adapted_suggestion}")
        
        # Add context-aware suggestions
        contextual_suggestions = self.generate_contextual_suggestions(context)
        suggestions.extend(contextual_suggestions)
        
        # Add engagement tips
        engagement_tip = self.generate_engagement_tip(context)
        if engagement_tip:
            suggestions.append(engagement_tip)
        
        return suggestions[:6]  # Limit to 6 suggestions
    
    def adapt_example_to_context(self, example_text: str, context: ConversationContext) -> str:
        """Adapt retrieved example to current conversation context"""
        # Basic adaptation - make it more contextual
        adapted = example_text.strip()
        
        # Make it conversational and contextual
        if context.topics:
            # Try to incorporate detected topics
            main_topic = context.topics[0] if context.topics else "that"
            adapted = adapted.replace("it", main_topic).replace("that", main_topic)
        
        # Adjust energy level based on engagement
        if context.engagement_level == 'high' and not any(char in adapted for char in '!😊'):
            if not adapted.endswith(('!', '?', '😊')):
                adapted += " 😊"
        
        # Ensure it's conversational lowercase style
        if adapted and adapted[0].isupper():
            adapted = adapted[0].lower() + adapted[1:]
        
        return adapted
    
    def generate_contextual_suggestions(self, context: ConversationContext) -> List[str]:
        """Generate additional contextual suggestions"""
        suggestions = []
        
        # Based on engagement level
        if context.engagement_level == 'high':
            if context.sentiment == 'positive':
                suggestions.append("🔥 high engagement + positive vibe: perfect time to be a bit more playful!")
            else:
                suggestions.append("📈 they're engaged: keep the conversation flowing with open questions")
        elif context.engagement_level == 'low':
            suggestions.append("💡 low engagement: try asking about their interests or sharing something fun")
        
        # Based on conversation stage
        if len(context.messages) > 6:
            suggestions.append("🎯 conversation flowing: consider suggesting a specific activity or moving topics")
        elif len(context.messages) <= 3:
            suggestions.append("🌱 early stage: focus on finding common ground and shared interests")
        
        return suggestions
    
    def generate_engagement_tip(self, context: ConversationContext) -> Optional[str]:
        """Generate engagement-specific tips"""
        if context.engagement_level == 'high' and context.sentiment == 'positive':
            return "✨ great chemistry detected: they're really responding well to your energy!"
        elif context.boundary_signals:
            return "⚠️ respect mode: give them space and let them lead the pace"
        elif context.engagement_level == 'medium':
            return "📊 steady engagement: you're on the right track, keep being authentic"
        else:
            return None
    
    def extract_conversation_topics(self, messages: List[str]) -> List[str]:
        """Extract main topics from conversation using simple keyword extraction"""
        topics = []
        common_topics = {
            'hiking': ['hike', 'hiking', 'trail', 'mountain', 'outdoor'],
            'travel': ['travel', 'trip', 'vacation', 'visit', 'country'],
            'music': ['music', 'song', 'concert', 'band', 'album'],
            'food': ['food', 'restaurant', 'cooking', 'eat', 'dinner'],
            'work': ['work', 'job', 'career', 'office', 'business'],
            'hobbies': ['hobby', 'interest', 'passion', 'love doing'],
            'movies': ['movie', 'film', 'cinema', 'watch', 'show'],
            'sports': ['sport', 'game', 'play', 'team', 'gym']
        }
        
        text = ' '.join(messages).lower()
        
        for topic, keywords in common_topics.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)
        
        return topics[:3]  # Return top 3 topics
    
    def determine_conversation_stage(self, messages: List[str]) -> str:
        """Determine what stage the conversation is in"""
        message_count = len(messages)
        
        if message_count <= 2:
            return 'opening'
        elif message_count <= 5:
            return 'getting_to_know'
        elif message_count <= 10:
            return 'building_connection'
        else:
            return 'established_flow'
    
    # Keep all the existing methods from the original DatingCoach class
    def extract_messages(self, text: str) -> List[str]:
        """Extract individual messages from conversation text"""
        messages = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 2:
                cleaned = re.sub(r'^\d{1,2}:\d{2}.*?:', '', line)
                cleaned = re.sub(r'^[A-Za-z\s]+:', '', cleaned)
                cleaned = cleaned.strip()
                if cleaned:
                    messages.append(cleaned)
        
        return messages[-10:]
    
    def detect_boundaries(self, messages: List[str]) -> bool:
        """Detect if conversation partner is setting boundaries"""
        recent_text = ' '.join(messages[-3:]).lower()
        
        for keyword in self.boundary_keywords:
            if keyword in recent_text:
                return True
        
        return False
    
    def analyze_sentiment_and_emotion(self, messages: List[str]) -> Dict:
        """Analyze sentiment and emotion of recent messages"""
        if not messages:
            return {'sentiment': 'neutral', 'emotion': 'neutral', 'engagement': 'low'}
        
        recent_text = ' '.join(messages[-3:])
        
        sentiment_result = self.sentiment_analyzer(recent_text[:512])
        sentiment = sentiment_result[0]['label']
        sentiment_score = sentiment_result[0]['score']
        
        emotion_result = self.emotion_analyzer(recent_text[:512])
        emotion = emotion_result[0]['label']
        emotion_score = emotion_result[0]['score']
        
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
    
    def create_conversation_context(self, messages: List[str], analysis: Dict) -> ConversationContext:
        """Create structured conversation context for RAG"""
        return ConversationContext(
            messages=messages,
            sentiment=analysis.get('sentiment', 'neutral'),
            emotion=analysis.get('emotion', 'neutral'),
            engagement_level=analysis.get('engagement', 'low'),
            topics=self.extract_conversation_topics(messages),
            conversation_stage=self.determine_conversation_stage(messages),
            boundary_signals=self.detect_boundaries(messages)
        )
    
    def generate_suggestions(self, analysis: Dict, messages: List[str]) -> List[str]:
        """Generate suggestions using RAG system"""
        # Create conversation context
        context = self.create_conversation_context(messages, analysis)
        
        # Use RAG to generate suggestions
        rag_suggestions = self.generate_rag_suggestions(context)
        
        return rag_suggestions
    
    # Keep the GUI setup and other methods from original class
    def setup_gui(self):
        """Setup the main GUI interface with RAG indicators"""
        self.root = tk.Tk()
        self.root.title("RAG-Enhanced Dating Coach - AI Conversation Assistant")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title with RAG indicator
        title_label = ttk.Label(main_frame, text="RAG-Enhanced Dating Coach", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 5))
        
        # Subtitle showing RAG capabilities
        subtitle_label = ttk.Label(main_frame, text="Powered by Retrieval-Augmented Generation with 5000+ conversation examples", 
                                  font=('Arial', 10), foreground='blue')
        subtitle_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        
        # Conversation input area
        ttk.Label(main_frame, text="Conversation (paste messages here):").grid(row=2, column=0, sticky=tk.W)
        self.conversation_text = scrolledtext.ScrolledText(main_frame, height=12, width=60)
        self.conversation_text.grid(row=3, column=0, padx=(0, 10), pady=(0, 10))
        
        # Analysis and suggestions area
        ttk.Label(main_frame, text="RAG-Generated Analysis & Suggestions:").grid(row=2, column=1, sticky=tk.W)
        self.suggestions_text = scrolledtext.ScrolledText(main_frame, height=12, width=60, 
                                                         bg='#e8f4f8')
        self.suggestions_text.grid(row=3, column=1, pady=(0, 10))
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        self.analyze_btn = ttk.Button(button_frame, text="Analyze with RAG", 
                                     command=self.analyze_conversation)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(button_frame, text="Clear All", 
                                   command=self.clear_all)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(button_frame, text="Save Analysis", 
                                  command=self.save_analysis)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Status area
        rag_status = "Ready" if self.faiss_index is not None else "RAG system not available"
        self.status_label = ttk.Label(main_frame, text=f"RAG System: {rag_status} | Ready to analyze conversations", 
                                     foreground='green' if self.faiss_index else 'orange')
        self.status_label.grid(row=5, column=0, columnspan=2, pady=5)
        
        # Auto-analysis toggle
        self.auto_analyze_var = tk.BooleanVar()
        auto_check = ttk.Checkbutton(main_frame, text="Auto-analyze with RAG on text change", 
                                    variable=self.auto_analyze_var)
        auto_check.grid(row=6, column=0, columnspan=2, pady=5)
        
        self.conversation_text.bind('<KeyRelease>', self.on_text_change)
    
    def on_text_change(self, event=None):
        """Handle text changes for real-time analysis"""
        if self.auto_analyze_var.get():
            if hasattr(self, 'analyze_timer'):
                self.analyze_timer.cancel()
            self.analyze_timer = threading.Timer(2.0, self.analyze_conversation)
            self.analyze_timer.start()
    
    def analyze_conversation(self):
        """Main analysis function using RAG"""
        self.status_label.config(text="Analyzing conversation with RAG...", foreground='orange')
        self.root.update()
        
        try:
            conv_text = self.conversation_text.get('1.0', tk.END)
            messages = self.extract_messages(conv_text)
            
            if not messages:
                self.suggestions_text.delete('1.0', tk.END)
                self.suggestions_text.insert(tk.END, "No messages to analyze. Please paste some conversation text above.")
                self.status_label.config(text="No messages found", foreground='red')
                return
            
            analysis = self.analyze_sentiment_and_emotion(messages)
            suggestions = self.generate_suggestions(analysis, messages)
            
            self.display_analysis(analysis, suggestions, messages)
            
            self.analysis_results.append({
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis,
                'suggestions': suggestions,
                'message_count': len(messages),
                'rag_enabled': self.faiss_index is not None
            })
            
            status_text = "RAG analysis complete!" if self.faiss_index else "Analysis complete (RAG unavailable)"
            self.status_label.config(text=status_text, foreground='green')
            
        except Exception as e:
            self.suggestions_text.delete('1.0', tk.END)
            self.suggestions_text.insert(tk.END, f"Error during RAG analysis: {str(e)}")
            self.status_label.config(text="Analysis failed", foreground='red')
    
    def display_analysis(self, analysis: Dict, suggestions: List[str], messages: List[str]):
        """Display analysis results with RAG indicators"""
        self.suggestions_text.delete('1.0', tk.END)
        
        # Analysis summary with RAG indicator
        self.suggestions_text.insert(tk.END, "🤖 rag-enhanced conversation analysis\n")
        self.suggestions_text.insert(tk.END, "="*50 + "\n\n")
        
        self.suggestions_text.insert(tk.END, f"sentiment: {analysis['sentiment'].title()} ({analysis['sentiment_score']:.2f})\n")
        self.suggestions_text.insert(tk.END, f"emotion: {analysis['emotion'].title()} ({analysis['emotion_score']:.2f})\n")
        self.suggestions_text.insert(tk.END, f"engagement: {analysis['engagement'].title()}\n")
        self.suggestions_text.insert(tk.END, f"analyzed {len(messages)} recent messages\n")
        
        # RAG system status
        rag_status = "✅ Active" if self.faiss_index is not None else "❌ Unavailable"
        self.suggestions_text.insert(tk.END, f"rag system: {rag_status}\n\n")
        
        # Suggestions
        self.suggestions_text.insert(tk.END, "🎯 rag-generated suggestions\n")
        self.suggestions_text.insert(tk.END, "="*50 + "\n\n")
        
        for suggestion in suggestions:
            self.suggestions_text.insert(tk.END, f"{suggestion}\n\n")
        
        # General tips
        self.suggestions_text.insert(tk.END, "📋 conversation wisdom\n")
        self.suggestions_text.insert(tk.END, "="*50 + "\n")
        self.suggestions_text.insert(tk.END, "• suggestions generated from thousands of real conversations\n")
        self.suggestions_text.insert(tk.END, "• always respect boundaries and consent\n")
        self.suggestions_text.insert(tk.END, "• use these as inspiration for your authentic voice\n")
        self.suggestions_text.insert(tk.END, "• rag retrieves contextually relevant examples\n")
        self.suggestions_text.insert(tk.END, "• let your personality shine through naturally\n")
    
    def clear_all(self):
        """Clear all text areas"""
        self.conversation_text.delete('1.0', tk.END)
        self.suggestions_text.delete('1.0', tk.END)
        rag_status = "Ready" if self.faiss_index is not None else "RAG system not available"
        self.status_label.config(text=f"RAG System: {rag_status} | Ready to analyze conversations", 
                               foreground='green' if self.faiss_index else 'orange')
    
    def save_analysis(self):
        """Save analysis results to file"""
        if not self.analysis_results:
            self.status_label.config(text="No analysis to save", foreground='red')
            return
        
        filename = f"rag_dating_coach_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.analysis_results, f, indent=2)
            self.status_label.config(text=f"RAG analysis saved to {filename}", foreground='green')
        except Exception as e:
            self.status_label.config(text=f"Save failed: {str(e)}", foreground='red')
    
    def run(self):
        """Start the RAG-enhanced application"""
        print("Starting RAG-Enhanced Dating Coach...")
        print("Features: Retrieval-Augmented Generation with conversation corpus!")
        self.root.mainloop()

if __name__ == "__main__":
    coach = RAGDatingCoach()
    coach.run()
