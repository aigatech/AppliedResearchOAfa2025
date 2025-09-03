#!/usr/bin/env python3
"""
Simple Portfolio Risk Coach Agent with Gradient Updates and HuggingFace Integration
A unified, simplified implementation combining LangChain, PyTorch gradients, and HuggingFace.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from dotenv import load_dotenv

# LangChain imports
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage

# HuggingFace imports
from transformers import pipeline

class PortfolioRiskModel(nn.Module):
    """Simple neural network for portfolio risk assessment with gradient learning."""
    
    def __init__(self):
        super(PortfolioRiskModel, self).__init__()
        
        # Risk weights that can be learned
        self.risk_weights = nn.Parameter(torch.ones(5))  # 5 risk factors
        
        # Simple neural network
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass to predict portfolio risk."""
        # Apply learned risk weights to features
        weighted_features = features * self.risk_weights
        
        # Neural network processing
        x = self.relu(self.fc1(weighted_features))
        risk_score = self.sigmoid(self.fc2(x))
        
        return risk_score
    
    def get_risk_weights(self) -> Dict[str, float]:
        """Get current risk weights as dictionary."""
        weights = self.risk_weights.detach().numpy()
        return {
            'tech_risk': float(weights[0]),
            'energy_risk': float(weights[1]),
            'concentration_risk': float(weights[2]),
            'volatility_risk': float(weights[3]),
            'sector_diversity': float(weights[4])
        }

class SimplePortfolioAgent:
    """
    Simplified Portfolio Risk Coach Agent with gradient updates and HuggingFace integration.
    """
    
    def __init__(self, gemini_api_key: str = None):
        # Load environment variables
        load_dotenv()
        
        # Set up Gemini API key
        if gemini_api_key:
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
        
        # Initialize components
        self._setup_huggingface_models()
        self._setup_gradient_model()
        self._setup_langchain_components()
        
        # Portfolio data
        self.sample_portfolios = {
            'tech_heavy': {'AAPL': 0.4, 'MSFT': 0.3, 'TSLA': 0.3},
            'diversified': {'AAPL': 0.2, 'JPM': 0.2, 'XOM': 0.2, 'JNJ': 0.2, 'V': 0.2},
            'energy_focused': {'XOM': 0.4, 'CVX': 0.3, 'COP': 0.3}
        }
        
        self.sectors = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'TSLA': 'Technology',
            'JPM': 'Financial', 'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
            'JNJ': 'Healthcare', 'V': 'Financial'
        }
        
        self.learning_history = []
    
    def _setup_huggingface_models(self):
        """Initialize HuggingFace models."""
        print("🤖 Loading HuggingFace models...")
        
        # Risk sentiment analysis
        self.risk_classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            return_all_scores=True
        )
        
        # Text generation for explanations
        self.text_generator = pipeline(
            "text-generation",
            model="gpt2",
            max_length=100,
            do_sample=True,
            temperature=0.7
        )
    
    def _setup_gradient_model(self):
        """Set up PyTorch model for gradient learning."""
        print("🧠 Setting up gradient learning model...")
        self.model = PortfolioRiskModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.BCELoss()
    
    def _setup_gemini_llm(self):
        """Set up Gemini LLM."""
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.1,
                max_output_tokens=500,
                convert_system_message_to_human=True
            )
            print("✅ Connected to Gemini 1.5 Flash")
            return llm
        except Exception as e:
            print(f"⚠️ Gemini failed: {str(e)[:50]}...")
            return self._create_mock_llm()
    
    def _create_mock_llm(self):
        """Create a mock LLM for demo purposes."""
        class MockLLM:
            def invoke(self, messages, **kwargs):
                last_message = messages[-1].content if messages else ""
                if "analyze" in last_message.lower():
                    return AIMessage(content="I'll analyze the portfolio using my risk assessment tools.")
                elif "learn" in last_message.lower():
                    return AIMessage(content="I'm learning from your feedback using gradient updates.")
                else:
                    return AIMessage(content="I'm here to help with portfolio risk analysis. What would you like me to do?")
        return MockLLM()
    
    def _setup_langchain_components(self):
        """Set up LangChain agent with tools and memory."""
        
        # Initialize LLM
        self.llm = self._setup_gemini_llm()
        
        # Create tools
        self.tools = [
            self._create_analyze_portfolio_tool(),
            self._create_learn_from_feedback_tool(),
            self._create_add_portfolio_tool(),
            self._create_compare_portfolios_tool(),
            self._create_general_question_tool(),
            self._create_educational_tool()
        ]
        
        # Set up memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a Portfolio Risk Coach Agent with gradient learning capabilities. Your role is to:

1. Analyze portfolio risk using HuggingFace sentiment analysis
2. Learn from user feedback using PyTorch gradient updates
3. Provide clear, actionable risk assessments
4. Maintain conversation context

You have access to tools for portfolio analysis, gradient learning, and portfolio management.

Always be helpful and willing to learn from user corrections."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=3
        )
    
    def extract_features(self, portfolio: Dict[str, float]) -> torch.Tensor:
        """Extract numerical features from portfolio for gradient learning."""
        sector_exposure = self._calculate_sector_exposure(portfolio)
        concentration = self._calculate_concentration(portfolio)
        volatility = self._calculate_volatility(portfolio)
        diversification = self._calculate_diversification(portfolio)
        
        features = torch.tensor([
            sector_exposure.get('Technology', 0.0),
            sector_exposure.get('Energy', 0.0),
            concentration,
            volatility,
            diversification
        ], dtype=torch.float32)
        
        return features
    
    def _create_analyze_portfolio_tool(self):
        """Tool for analyzing portfolio risk using HuggingFace and gradient model."""
        
        @tool
        def analyze_portfolio_risk(portfolio_name: str) -> str:
            """Analyze portfolio risk using HuggingFace sentiment analysis and gradient model."""
            # Find portfolio with flexible matching
            clean_name = portfolio_name.lower().replace("portfolio", "").replace("_", "").replace(" ", "").strip()
            matching_portfolio = None
            
            # Try exact match first
            if portfolio_name in self.sample_portfolios:
                matching_portfolio = portfolio_name
            else:
                # Try flexible matching
                for key in self.sample_portfolios.keys():
                    key_clean = key.lower().replace("_", "").replace(" ", "")
                    if (clean_name in key_clean or key_clean in clean_name or 
                        any(word in key_clean for word in clean_name.split()) or
                        any(word in clean_name for word in key_clean.split())):
                        matching_portfolio = key
                        break
            
            if not matching_portfolio:
                available = ", ".join([f"'{p}'" for p in self.sample_portfolios.keys()])
                return f"Portfolio '{portfolio_name}' not found. Available portfolios: {available}. You can also add a custom portfolio using the add_custom_portfolio tool."
            
            portfolio = self.sample_portfolios[matching_portfolio]
            
            # Calculate comprehensive metrics
            sector_exposure = self._calculate_sector_exposure(portfolio)
            concentration = self._calculate_concentration(portfolio)
            volatility = self._calculate_volatility(portfolio)
            diversification = self._calculate_diversification(portfolio)
            
            # Generate detailed risk analysis text
            risk_text = f"Portfolio {portfolio_name}: Tech exposure {sector_exposure.get('Technology', 0):.1%}, "
            risk_text += f"Energy exposure {sector_exposure.get('Energy', 0):.1%}, "
            risk_text += f"Concentration {concentration:.2f}, Volatility {volatility:.2f}"
            
            # Use HuggingFace for sentiment analysis
            try:
                risk_scores = self.risk_classifier(risk_text)
                sentiment_risk = next((score['score'] for score in risk_scores if score['label'] == 'NEGATIVE'), 0)
            except:
                sentiment_risk = 0.5  # Default if HuggingFace fails
            
            # Use gradient model for prediction
            features = self.extract_features(portfolio)
            self.model.eval()
            with torch.no_grad():
                gradient_risk = self.model(features).item()
            
            # Combine both approaches
            combined_risk = (sentiment_risk + gradient_risk) / 2
            
            # Determine risk level with detailed thresholds
            if combined_risk > 0.7:
                risk_level = "VERY HIGH"
                risk_color = "🔴"
            elif combined_risk > 0.6:
                risk_level = "HIGH"
                risk_color = "🟠"
            elif combined_risk > 0.4:
                risk_level = "MEDIUM-HIGH"
                risk_color = "🟡"
            elif combined_risk > 0.3:
                risk_level = "MEDIUM"
                risk_color = "🟢"
            else:
                risk_level = "LOW"
                risk_color = "🟢"
            
            # Generate concise analysis
            analysis = f"""RISK ANALYSIS: {portfolio_name}
Risk Level: {risk_level} ({combined_risk:.1%})
Gradient Score: {gradient_risk:.1%}
Sentiment Score: {sentiment_risk:.1%}

COMPOSITION: {len(portfolio)} stocks, {sum(portfolio.values()):.1%} total weight

SECTOR EXPOSURE:"""
            for sector, exposure in sector_exposure.items():
                analysis += f" {sector}: {exposure:.1%}"
            
            analysis += f"""

RISK METRICS:
Concentration: {concentration:.3f} {'(High)' if concentration > 0.3 else '(Moderate)' if concentration > 0.2 else '(Low)'}
Volatility: {volatility:.3f} {'(High)' if volatility > 0.2 else '(Moderate)' if volatility > 0.15 else '(Low)'}
Diversification: {diversification:.3f} {'(Good)' if diversification > 0.5 else '(Moderate)' if diversification > 0.3 else '(Poor)'}

INSIGHTS:"""
            
            # Add key insights
            insights = []
            if sector_exposure.get('Technology', 0) > 0.5:
                insights.append("High tech exposure")
            if concentration > 0.3:
                insights.append("High concentration risk")
            if volatility > 0.2:
                insights.append("Above-average volatility")
            if diversification < 0.3:
                insights.append("Low diversification")
            
            if insights:
                analysis += " " + ", ".join(insights)
            else:
                analysis += " Portfolio appears balanced"
            
            # Add brief recommendations
            analysis += f"""

RECOMMENDATIONS:"""
            if risk_level in ["VERY HIGH", "HIGH"]:
                analysis += " Reduce high-risk exposures, add defensive positions"
            elif risk_level == "MEDIUM":
                analysis += " Monitor regularly, consider rebalancing"
            else:
                analysis += " Conservative profile, may add growth opportunities"
            
            analysis += f"""

AI Status: {len(self.learning_history)} learning events recorded
Feedback: "Learn from feedback: {portfolio_name}, [feedback], [target risk]"
"""
            
            return analysis
        
        return analyze_portfolio_risk
    
    def _create_learn_from_feedback_tool(self):
        """Tool for learning from user feedback using gradient updates."""
        
        @tool
        def learn_from_feedback(portfolio_name: str, user_feedback: str, target_risk: str) -> str:
            """Learn from user feedback using PyTorch gradient updates."""
            # Find portfolio
            clean_name = portfolio_name.lower().replace("portfolio", "").replace("_", "").strip()
            matching_portfolio = None
            for key in self.sample_portfolios.keys():
                if clean_name in key.lower() or key.lower() in clean_name:
                    matching_portfolio = key
                    break
            
            if not matching_portfolio:
                return f"Portfolio '{portfolio_name}' not found."
            
            portfolio = self.sample_portfolios[matching_portfolio]
            
            # Gradient learning
            self.model.train()
            features = self.extract_features(portfolio)
            
            # Convert target to numerical
            target_map = {"LOW": 0.2, "MEDIUM": 0.5, "HIGH": 0.8}
            target = torch.tensor([target_map.get(target_risk.upper(), 0.5)], dtype=torch.float32)
            
            # Forward pass
            predicted_risk = self.model(features)
            
            # Calculate loss and update
            loss = self.criterion(predicted_risk, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Record learning
            learning_event = {
                'timestamp': datetime.now().isoformat(),
                'portfolio': portfolio_name,
                'feedback': user_feedback,
                'target_risk': target_risk,
                'loss': loss.item(),
                'predicted_risk': predicted_risk.item(),
                'updated_weights': self.model.get_risk_weights()
            }
            self.learning_history.append(learning_event)
            
            # Generate explanation using HuggingFace
            try:
                explanation = self.text_generator(f"Learning from feedback: {user_feedback}", max_length=50)[0]['generated_text']
            except:
                explanation = f"Adapted analysis based on your feedback about {target_risk} risk."
            
            return f"✅ Learned from feedback!\nLoss: {loss.item():.4f}\nExplanation: {explanation}\nUpdated weights: {self.model.get_risk_weights()}"
        
        return learn_from_feedback
    
    def _create_add_portfolio_tool(self):
        """Tool for adding custom portfolios."""
        
        @tool
        def add_custom_portfolio(portfolio_name: str, stocks_and_weights: str) -> str:
            """Add a custom portfolio. Format: 'AAPL:0.4,MSFT:0.3,TSLA:0.3'"""
            try:
                portfolio = {}
                total_weight = 0
                
                for item in stocks_and_weights.split(','):
                    stock, weight_str = item.strip().split(':')
                    weight = float(weight_str)
                    portfolio[stock.strip()] = weight
                    total_weight += weight
                
                # Normalize weights
                if abs(total_weight - 1.0) > 0.01:
                    for stock in portfolio:
                        portfolio[stock] /= total_weight
                
                self.sample_portfolios[portfolio_name] = portfolio
                
                return f"✅ Added portfolio '{portfolio_name}' with {len(portfolio)} stocks. Total weight: {sum(portfolio.values()):.2f}"
                
            except Exception as e:
                return f"❌ Error: {str(e)}. Use format: 'AAPL:0.4,MSFT:0.3,TSLA:0.3'"
        
        return add_custom_portfolio
    
    def _create_compare_portfolios_tool(self):
        """Tool for comparing two portfolios."""
        
        @tool
        def compare_portfolios(portfolio1_name: str, portfolio2_name: str) -> str:
            """Compare two portfolios based on risk and diversification."""
            # Find portfolios
            clean_name1 = portfolio1_name.lower().replace("portfolio", "").replace("_", "").strip()
            clean_name2 = portfolio2_name.lower().replace("portfolio", "").replace("_", "").strip()
            
            matching_portfolio1 = None
            for key in self.sample_portfolios.keys():
                if clean_name1 in key.lower() or key.lower() in clean_name1:
                    matching_portfolio1 = key
                    break
            
            matching_portfolio2 = None
            for key in self.sample_portfolios.keys():
                if clean_name2 in key.lower() or key.lower() in clean_name2:
                    matching_portfolio2 = key
                    break
            
            if not matching_portfolio1 or not matching_portfolio2:
                return f"One or both portfolios not found. Available: {list(self.sample_portfolios.keys())}"
            
            portfolio1 = self.sample_portfolios[matching_portfolio1]
            portfolio2 = self.sample_portfolios[matching_portfolio2]
            
            # Calculate metrics for both
            sector_exposure1 = self._calculate_sector_exposure(portfolio1)
            sector_exposure2 = self._calculate_sector_exposure(portfolio2)
            
            concentration1 = self._calculate_concentration(portfolio1)
            concentration2 = self._calculate_concentration(portfolio2)
            
            volatility1 = self._calculate_volatility(portfolio1)
            volatility2 = self._calculate_volatility(portfolio2)
            
            diversification1 = self._calculate_diversification(portfolio1)
            diversification2 = self._calculate_diversification(portfolio2)
            
            # Generate comparison text
            comparison_text = f"COMPARISON: {portfolio1_name} vs {portfolio2_name}\n"
            comparison_text += f"Sector Exposure:\n"
            comparison_text += f"  {portfolio1_name}: Tech {sector_exposure1.get('Technology', 0):.1%}, Energy {sector_exposure1.get('Energy', 0):.1%}\n"
            comparison_text += f"  {portfolio2_name}: Tech {sector_exposure2.get('Technology', 0):.1%}, Energy {sector_exposure2.get('Energy', 0):.1%}\n"
            comparison_text += f"Concentration:\n"
            comparison_text += f"  {portfolio1_name}: {concentration1:.3f}\n"
            comparison_text += f"  {portfolio2_name}: {concentration2:.3f}\n"
            comparison_text += f"Volatility:\n"
            comparison_text += f"  {portfolio1_name}: {volatility1:.3f}\n"
            comparison_text += f"  {portfolio2_name}: {volatility2:.3f}\n"
            comparison_text += f"Diversification:\n"
            comparison_text += f"  {portfolio1_name}: {diversification1:.3f}\n"
            comparison_text += f"  {portfolio2_name}: {diversification2:.3f}\n"
            
            return comparison_text
        
        return compare_portfolios
    
    def _create_general_question_tool(self):
        """Tool for answering general questions."""
        
        @tool
        def answer_general_question(question: str) -> str:
            """Answer general questions."""
            if "what" in question.lower() and "risk" in question.lower():
                return "Portfolio risk is a measure of the uncertainty or variability of returns on a portfolio of assets. It's typically assessed using metrics like volatility, concentration, and sector diversity. A higher risk score indicates a greater potential for losses."
            elif "how" in question.lower() and "risk" in question.lower():
                return "Risk is typically quantified using a risk score, which is a numerical representation of the likelihood of an adverse event occurring. This score is derived from a combination of factors including market volatility, sector concentration, and diversification."
            elif "what" in question.lower() and "volatility" in question.lower():
                return "Volatility is a statistical measure of the dispersion of returns for a given asset or portfolio. It's often expressed as a standard deviation of returns. A higher volatility indicates greater price fluctuations."
            elif "what" in question.lower() and "concentration" in question.lower():
                return "Concentration refers to the extent to which a portfolio's assets are concentrated in a few specific sectors or assets. A highly concentrated portfolio can be more vulnerable to sector-specific risks or market downturns."
            elif "what" in question.lower() and "diversification" in question.lower():
                return "Diversification is the strategy of investing in a variety of assets to reduce risk. By holding assets with different risk profiles and correlations, you can mitigate the impact of any single asset's performance on your overall portfolio."
            else:
                return "I'm not sure I can answer that question directly. I can help you analyze portfolio risk, learn from feedback, and manage portfolios."
        
        return answer_general_question
    
    def _create_educational_tool(self):
        """Tool for providing educational content."""
        
        @tool
        def provide_educational_content(topic: str) -> str:
            """Provide educational content on a given topic."""
            if "risk" in topic.lower():
                return "Risk is a fundamental concept in investing. It refers to the possibility of losing money or experiencing a decline in the value of an investment. Investors assess risk to make informed decisions about their portfolios. Risk can be quantified using various metrics, including volatility, concentration, and diversification."
            elif "volatility" in topic.lower():
                return "Volatility is a statistical measure of the dispersion of returns for a given asset or portfolio. It's often expressed as a standard deviation of returns. A higher volatility indicates greater price fluctuations and potential for larger gains or losses."
            elif "concentration" in topic.lower():
                return "Concentration refers to the extent to which a portfolio's assets are concentrated in a few specific sectors or assets. A highly concentrated portfolio can be more vulnerable to sector-specific risks or market downturns. Diversification is a key strategy to mitigate this."
            elif "diversification" in topic.lower():
                return "Diversification is the strategy of investing in a variety of assets to reduce risk. By holding assets with different risk profiles and correlations, you can mitigate the impact of any single asset's performance on your overall portfolio. This is a fundamental principle of modern portfolio theory."
            else:
                return "I can provide educational content on risk, volatility, concentration, and diversification. What topic would you like to learn about?"
        
        return provide_educational_content
    
    def _calculate_sector_exposure(self, portfolio: Dict[str, float]) -> Dict[str, float]:
        """Calculate sector exposure."""
        sector_exposure = {}
        for stock, weight in portfolio.items():
            sector = self.sectors.get(stock, 'Other')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
        return sector_exposure
    
    def _calculate_concentration(self, portfolio: Dict[str, float]) -> float:
        """Calculate concentration."""
        return sum(weight**2 for weight in portfolio.values())
    
    def _calculate_volatility(self, portfolio: Dict[str, float]) -> float:
        """Calculate volatility."""
        return sum(
            weight * (0.25 if self.sectors.get(stock) == 'Technology' else 0.15)
            for stock, weight in portfolio.items()
        )
    
    def _calculate_diversification(self, portfolio: Dict[str, float]) -> float:
        """Calculate diversification."""
        sector_exposure = self._calculate_sector_exposure(portfolio)
        return 1 - max(sector_exposure.values()) if sector_exposure else 0
    
    def chat(self, message: str) -> str:
        """Main chat interface."""
        try:
            response = self.agent_executor.invoke({"input": message})
            return response["output"]
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_learning_history(self) -> List[Dict]:
        """Get learning history."""
        return self.learning_history

def interactive_demo():
    """Interactive demo of the simplified agent."""
    print("🤖 Simple Portfolio Risk Coach Agent")
    print("=" * 50)
    print("Features: Gradient Learning + HuggingFace + LangChain")
    print("=" * 50)
    
    # Initialize agent
    agent = SimplePortfolioAgent()
    
    print("\nAvailable portfolios:")
    for i, (name, portfolio) in enumerate(agent.sample_portfolios.items(), 1):
        print(f"{i}. {name.replace('_', ' ').title()}")
    
    print("\nExample commands:")
    print("- 'Analyze tech_heavy portfolio'")
    print("- 'Learn from feedback: tech_heavy, too risky, LOW'")
    print("- 'Add custom portfolio: my_portfolio, AAPL:0.4,MSFT:0.3,TSLA:0.3'")
    print("- 'quit' to exit")
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("\n🤖 Agent: ", end="")
            response = agent.chat(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    interactive_demo()
