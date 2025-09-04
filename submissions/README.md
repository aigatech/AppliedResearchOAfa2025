# RAG-Enhanced Real-Time Dating Coach

**AI@GT Applied Research Team Submission by John Doe**

## Project Title
RAG-Enhanced Real-Time Dating Coach - Generative Conversation Assistant

## What It Does

This application is an advanced real-time conversation coach that uses **Retrieval-Augmented Generation (RAG)** with embeddings to provide dynamic, contextual conversation suggestions. Instead of pre-coded templates, it generates authentic responses by retrieving and adapting examples from a large corpus of real conversations.

### 🚀 **Key RAG Features:**
- **Semantic Search**: Uses sentence transformers to find contextually relevant conversation examples
- **Vector Database**: FAISS index with 5,000+ processed conversation examples for fast retrieval  
- **Dynamic Generation**: Creates unique suggestions adapted to current conversation context
- **Context-Aware Retrieval**: Considers sentiment, emotion, engagement, and topics for relevant examples
- **Structured Output**: Generates coherent, natural suggestions based on retrieved conversations

### 🔥 **Advanced AI Features:**
- **Real-time sentiment analysis** using `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Emotion detection** using `j-hartmann/emotion-english-distilroberta-base`
- **Sentence embeddings** using `all-MiniLM-L6-v2` for semantic similarity
- **RAG pipeline** with `the-rizz/the-rizz-corpus` dataset (55,000+ examples)
- **Conversational lowercase style** for natural, casual suggestions
- **Multi-layered context analysis** (sentiment + emotion + engagement + topics)
- **Boundary detection** with respectful communication prioritization

### What Makes It Special:
- **🧠 RAG-Powered Intelligence**: Retrieves contextually relevant examples from thousands of conversations
- **🎯 Dynamic Generation**: No fixed templates - generates unique suggestions every time
- **📊 Semantic Understanding**: Uses vector embeddings to understand conversation meaning and context
- **🔍 Context-Aware Retrieval**: Considers sentiment, emotion, engagement level, and conversation topics
- **💬 Natural conversational tone** - suggestions feel authentic and human-like
- **🚫 Never auto-sends** - all suggestions are for human consideration only
- **🛡️ Boundary-aware** - includes advanced boundary detection and respectful communication
- **🔒 Privacy-focused** - runs locally with FAISS vector database, no external API calls
- **📚 Continuously learning** - adapts suggestions based on conversation corpus analysis

## How to Run It

### Prerequisites
- Python 3.7 or higher
- Internet connection (for initial model download)

### Installation Steps

1. **Clone or download the project:**
   ```bash
   cd submissions/John_Doe/
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application (recommended - RAG-enhanced):**
   ```bash
   python run_coach.py
   ```

5. **Or run directly:**
   ```bash
   python rag_dating_coach.py    # RAG-enhanced version (recommended)
   python dating_coach.py        # Original version
   ```

### First Run Notes
- The first run will download AI models (~500MB total) - this may take a few minutes
- Models are cached locally for faster subsequent launches
- The app works entirely on CPU - no GPU required!

## How to Use

1. **Start the application** - A GUI window will open
2. **Paste conversation text** in the left panel (copy from your messaging app)
3. **Click "Analyze Conversation"** or enable auto-analysis
4. **Review AI suggestions** in the right panel
5. **Use suggestions as inspiration** - adapt them to your authentic voice
6. **Save analysis** for later review if desired

### Example Usage (RAG-Enhanced):
```
Conversation Input:
alex: hey! how was your weekend?
sam: it was incredible! went rock climbing for the first time
alex: no way! that sounds terrifying but amazing
sam: right?? i was so scared at first but the adrenaline rush was unreal
alex: i've always wanted to try that! where did you go?
sam: there's this awesome indoor gym downtown, perfect for beginners
alex: that's so cool! maybe i should check it out sometime
sam: you totally should! i'd love to go again, maybe we could go together?

RAG-Enhanced Analysis Output:
sentiment: Positive (0.99)
emotion: Joy (0.76)
engagement: High
rag system: ✅ Active

🤖 rag-generated suggestions based on similar conversations:
💬 that's so exciting! rock climbing sounds like such an adventure
💬 i love how you stepped out of your comfort zone! that's really inspiring  
💬 you're making me want to try it too! i'd definitely be up for going together
💬 the fact that you faced your fear and loved it shows what an amazing person you are
🔥 high engagement + shared activity interest: perfect opportunity for connection!
✨ they're clearly excited and open to spending time together
🎯 next move: express genuine interest and suggest a specific plan
```

## Technical Details

### AI Models & RAG Architecture:
- **Sentiment Analysis**: `cardiffnlp/twitter-roberta-base-sentiment-latest` - Robust Twitter-trained sentiment classifier
- **Emotion Recognition**: `j-hartmann/emotion-english-distilroberta-base` - Multi-emotion classification
- **Embeddings**: `all-MiniLM-L6-v2` - Sentence transformer for semantic similarity
- **Vector Database**: FAISS with cosine similarity for fast retrieval
- **Conversation Corpus**: `the-rizz/the-rizz-corpus` - 55,000+ conversation examples, filtered to 5,000 high-quality examples

### RAG Pipeline Architecture:
1. **Corpus Processing**: Clean and filter conversation examples for quality
2. **Embedding Generation**: Create vector representations using sentence transformers  
3. **Index Creation**: Build FAISS vector database for fast similarity search
4. **Context Analysis**: Analyze current conversation for sentiment, emotion, topics
5. **Retrieval**: Find most relevant conversation examples using semantic search
6. **Generation**: Adapt retrieved examples to current context and generate suggestions
7. **Filtering**: Apply safety filters and context relevance scoring

### Safety Features:
- **Boundary Detection**: Scans for keywords indicating discomfort or disinterest
- **Respectful Templates**: Pre-vetted response suggestions that avoid toxic patterns
- **Context Awareness**: Analyzes recent message history rather than isolated responses
- **No Auto-sending**: Human always remains in control of actual communication

### Privacy & Ethics:
- **Local Processing**: All analysis happens on your device
- **No Data Collection**: Conversations are not stored or transmitted
- **Consent-Focused**: Emphasizes respectful communication and boundary recognition
- **Educational Purpose**: Includes tips for healthy relationship communication

## File Structure
```
submissions/John_Doe/
├── rag_dating_coach.py      # RAG-Enhanced main application (RECOMMENDED)
├── dating_coach.py          # Original implementation for comparison
├── run_coach.py             # Simple launcher script
├── config.json             # Configuration settings
├── requirements.txt        # Python dependencies
└── README.md               # This documentation
```

## Future Enhancements
- Voice analysis integration
- Multi-language support
- Advanced personality matching
- Integration with popular messaging platforms
- Mobile app version

## Disclaimer
This tool is designed to promote respectful, authentic communication. It should be used as a learning aid, not as a replacement for genuine human connection and emotional intelligence. Always prioritize consent, boundaries, and authentic self-expression in your relationships.

---

**Developed for AI@GT Applied Research Team Assessment**  
*Focus: Hugging Face transformers, ethical AI, user interface design*
