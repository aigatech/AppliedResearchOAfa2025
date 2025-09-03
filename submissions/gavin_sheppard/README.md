## Project Title
**Demonstrating Different AI Models In HuggingFace Using Sports**

## What It Does
This project demonstrates advanced AI/ML capabilities using **5 different HuggingFace models** for comprehensive sports analysis. 

### Key AI Features:

#### **1. Advanced Sentiment Analysis**
- **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Capability**: Analyzes sentiment of sports news, social media posts, and fan reactions
- **Output**: Sentiment labels (POSITIVE/NEGATIVE/NEUTRAL) with confidence scores
- **Innovation**: Specialized for sports content with detailed sentiment interpretation

#### **2. Sports Commentary Generation**
- **Model**: `gpt2` (improved from DialoGPT)
- **Capability**: Generates realistic sports commentary and predictions
- **Output**: Coherent, contextually relevant sports commentary
- **Innovation**: Smart prompts, quality control, and high-quality fallback commentary
- **Features**:
  - Topic-specific prompts for different sports
  - Multiple generation attempts with quality scoring
  - Rich fallback commentary for reliable results

#### **3. Named Entity Recognition**
- **Model**: `dbmdz/bert-large-cased-finetuned-conll03-english`
- **Capability**: Extracts players, teams, locations, and organizations from text
- **Output**: Categorized entities with confidence scores
- **Innovation**: Specialized entity categorization for sports content

#### **4. Enhanced Question Answering System**
- **Model**: `distilbert-base-cased-distilled-squad`
- **Capability**: Answers sports-related questions with real-time data and web search
- **Output**: Direct answers with confidence scores and source tracking
- **Innovation**: Multi-source context building (Web search + Sports APIs + Known facts)
- **Features**: 
  - Real-time NFL/NBA news from ESPN APIs
  - Web search integration via DuckDuckGo
  - Known sports facts database for historical questions
  - Smart fallback system for low-confidence answers

#### **5. News Summarization**
- **Model**: `facebook/bart-large-cnn`
- **Capability**: Summarizes long sports articles into concise summaries
- **Output**: Key points and highlights from articles
- **Innovation**: Sports-specific summarization with relevant detail retention

## How to Run It

### Prerequisites
```bash
# Install all required dependencies
pip install -r requirements.txt

### Running the System
```bash
# Run the complete AI system
python3 ai_sports_analyzer.py
```


## Dependencies
- **Core**: transformers, torch, tokenizers
- **ML**: numpy, pandas, scikit-learn
- **NLP**: nltk, spacy
- **Utils**: requests, tqdm, datasets
- **Performance**: accelerate, sentencepiece
