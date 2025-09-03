# Portfolio Risk Coach Agent

An AI-powered portfolio risk analysis system that combines gradient learning, HuggingFace NLP models, and conversational AI to provide intelligent portfolio risk assessment and learning capabilities.

## 🎯 What This Project Does

This agent analyzes investment portfolios using multiple AI approaches:
- **Neural Network Risk Assessment**: PyTorch-based gradient learning that adapts to user feedback
- **NLP Analysis**: HuggingFace models for sentiment analysis and text generation
- **Conversational Interface**: LangChain agent with natural language understanding

The system learns from user corrections, improving its risk predictions over time through real gradient updates.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up API Key (Optional)
Create a `.env` file for Gemini integration:
```
GOOGLE_API_KEY=your-gemini-api-key-here
```

### 3. Run the Agent
```bash
python main.py
```

## 💡 Key Features

### **Portfolio Analysis**
- Risk level assessment (LOW/MEDIUM/HIGH)
- Sector exposure calculations
- Concentration and volatility metrics
- Comprehensive risk insights

### **AI Learning**
- Gradient-based learning from user feedback
- Neural network weight updates
- Learning history tracking
- Adaptive risk assessment

### **NLP Integration**
- DistilBERT for sentiment analysis
- GPT-2 for text generation
- Natural language understanding

## 🎯 Example Commands

```bash
# Analyze portfolios
"Analyze tech_heavy portfolio"
"Compare tech_heavy and diversified portfolios"

# Learn from feedback
"Learn from feedback: tech_heavy, too risky, LOW"

# Add custom portfolios
"Add custom portfolio: my_portfolio, AAPL:0.4,MSFT:0.3,TSLA:0.3"

# Educational questions
"What is portfolio risk?"
"Explain diversification"
```

## 📊 Built-in Portfolios

- **Tech Heavy**: AAPL (40%), MSFT (30%), TSLA (30%)
- **Diversified**: AAPL, JPM, XOM, JNJ, V (20% each)
- **Energy Focused**: XOM (40%), CVX (30%), COP (30%)

## 🛠️ Technical Stack

- **PyTorch**: Neural networks and gradient learning
- **HuggingFace**: Sentiment analysis and text generation
- **LangChain**: Conversational agent framework
- **Gemini**: LLM for reasoning (optional)

## 📝 Sample Output

```
RISK ANALYSIS: tech_heavy
Risk Level: HIGH (65.2%)
Gradient Score: 62.1%
Sentiment Score: 68.3%

COMPOSITION: 3 stocks, 100.0% total weight
SECTOR EXPOSURE: Technology: 100.0%

RISK METRICS:
Concentration: 0.340 (High)
Volatility: 0.250 (High)
Diversification: 0.000 (Poor)

INSIGHTS: High tech exposure, High concentration risk, Above-average volatility, Low diversification

RECOMMENDATIONS: Reduce high-risk exposures, add defensive positions
```

This system demonstrates real gradient learning with HuggingFace integration for intelligent portfolio risk analysis! 🎉
