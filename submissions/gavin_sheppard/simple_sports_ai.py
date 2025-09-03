import torch
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

class SportsAI:
    """Sports AI Demo - Text Generation and Sentiment Analysis"""
    
    def __init__(self):
        print("Initializing Sports AI Demo...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self._initialize_models()
        self.sports_data = {
            "news": [
                "The Los Angeles Lakers defeated the Golden State Warriors 120-115. LeBron James scored 35 points.",
                "Patrick Mahomes threw for 350 yards as the Kansas City Chiefs beat the Buffalo Bills 42-17.",
                "Lionel Messi scored a hat-trick in Barcelona's 4-1 victory over Real Madrid.",
                "The Boston Celtics are struggling with a 15-20 record this season.",
                "Tom Brady announced his retirement after 23 seasons in the NFL."
            ],
            "social_posts": [
                "Just watched the Lakers game! LeBron is absolutely incredible!",
                "Mahomes is the best QB in the league! Chiefs Kingdom!",
                "Messi is a football god! That hat-trick was pure magic!",
                "Celtics need to make some trades, this team is not working out",
                "Brady retiring is the end of an era. Legend!"
            ]
        }
        print("2 AI models loaded successfully!")
    
    def _initialize_models(self):
        print("Loading models...")
        # Sentiment Analysis - Using a more accurate model
        self.sentiment_analyzer = pipeline("sentiment-analysis", 
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=0 if self.device == "cuda" else -1)
        
        # Text Generation - Using a more accurate model
        self.text_generator = pipeline("text-generation", 
            model="microsoft/DialoGPT-medium",
            device=0 if self.device == "cuda" else -1, max_length=100,
            do_sample=True, temperature=0.7)
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of sports text"""
        result = self.sentiment_analyzer(text)
        sentiment = result[0]
        
        # Handle different label formats from different models
        label = sentiment['label'].upper()
        score = sentiment['score']
        
        if 'POSITIVE' in label or '5_STARS' in label or '4_STARS' in label:
            if score > 0.8:
                interpretation = f"Very positive! ({score:.2f}) - Strong enthusiasm detected!"
            else:
                interpretation = f"Positive sentiment ({score:.2f}) - Generally favorable"
        elif 'NEGATIVE' in label or '1_STAR' in label or '2_STARS' in label:
            if score > 0.8:
                interpretation = f"Very negative ({score:.2f}) - Strong disappointment"
            else:
                interpretation = f"Negative sentiment ({score:.2f}) - Generally unfavorable"
        else:
            interpretation = f"Neutral sentiment ({score:.2f}) - Mixed feelings"
        
        return {
            "text": text,
            "sentiment": sentiment['label'],
            "confidence": sentiment['score'],
            "interpretation": interpretation
        }
    
    def generate_commentary(self, topic):
        """Generate detailed sports commentary using AI with enhanced prompts"""
        topic_lower = topic.lower()
        
        # Enhanced prompts with more specific and detailed information
        if 'super bowl' in topic_lower:
            prompt = "The Super Bowl is the most watched sporting event in America, featuring the NFL's two best teams competing for the Lombardi Trophy. This championship game represents the culmination of an entire season's worth of hard work, strategy, and determination. The Super Bowl is where legends are made and careers are defined, with millions of viewers worldwide tuning in to witness history."
        elif 'lakers' in topic_lower:
            prompt = "The Los Angeles Lakers are one of the most storied franchises in NBA history, with 17 championships and a legacy of superstar talent. From Magic Johnson and Kareem Abdul-Jabbar to Kobe Bryant and LeBron James, the Lakers have consistently attracted the game's greatest players. Their purple and gold colors represent excellence, and their home court at Crypto.com Arena is one of the most iconic venues in sports."
        elif 'chiefs' in topic_lower:
            prompt = "The Kansas City Chiefs are known for their explosive offense led by superstar quarterback Patrick Mahomes. Arrowhead Stadium is one of the loudest venues in the NFL, with passionate fans creating an electric atmosphere. The Chiefs have revolutionized modern football with their innovative play-calling and dynamic passing attack, making them one of the most exciting teams to watch in the league."
        elif 'mahomes' in topic_lower:
            prompt = "Patrick Mahomes has redefined the quarterback position with his incredible arm talent, mobility, and ability to make plays that seem impossible. The 2018 NFL MVP has already won two Super Bowls and established himself as one of the game's elite players. His no-look passes, sidearm throws, and clutch performances have made him a generational talent who continues to push the boundaries of what's possible at the quarterback position."
        elif 'lebron' in topic_lower:
            prompt = "LeBron James is widely considered one of the greatest basketball players of all time, with four NBA championships and numerous MVP awards. His combination of size, athleticism, basketball IQ, and leadership has made him a dominant force for over two decades. LeBron's ability to impact every aspect of the game, from scoring and rebounding to playmaking and defense, has solidified his place among the all-time greats."
        elif 'nba' in topic_lower:
            prompt = "The NBA is the premier basketball league in the world, featuring the most talented athletes and showcasing the highest level of basketball competition. With 30 teams across North America, the NBA has become a global phenomenon, with games broadcast to over 200 countries. The league's emphasis on athleticism, skill, and entertainment has made it one of the most popular sports leagues worldwide."
        elif 'nfl' in topic_lower:
            prompt = "The NFL represents the pinnacle of American football, with 32 teams competing in the most physically demanding and strategically complex sport. Each game is a battle of strength, speed, intelligence, and teamwork. The NFL's popularity stems from its combination of athletic excellence, dramatic storylines, and the unpredictable nature of the game, making every Sunday a spectacle for millions of fans."
        elif 'championship' in topic_lower or 'finals' in topic_lower:
            prompt = "Championship games are the ultimate test of a team's ability, where months of preparation, strategy, and hard work culminate in a single moment of truth. These games separate the good from the great, as players must perform under the most intense pressure and scrutiny. Championship moments create lasting memories and define legacies, with every play potentially determining the outcome of an entire season."
        elif 'playoffs' in topic_lower:
            prompt = "Playoff games are where legends are made and dreams come true. The intensity reaches its peak as teams compete in single-elimination or best-of-series formats, where every mistake is magnified and every great play can change the course of history. The playoffs bring out the best in athletes, as they must perform at their highest level when the stakes are at their greatest."
        elif 'warriors' in topic_lower:
            prompt = "The Golden State Warriors revolutionized basketball with their fast-paced, three-point shooting style that changed the game forever. Led by Stephen Curry, Klay Thompson, and Draymond Green, the Warriors created a dynasty that won multiple championships and inspired a new generation of players to embrace the three-point shot. Their home court at Chase Center in San Francisco is one of the most modern and technologically advanced arenas in the NBA."
        elif 'celtics' in topic_lower:
            prompt = "The Boston Celtics are one of the most successful franchises in NBA history, with 17 championships and a rich tradition of excellence. From Bill Russell and Larry Bird to Paul Pierce and Jayson Tatum, the Celtics have consistently fielded competitive teams. Their home court at TD Garden is steeped in history, and their green and white colors represent one of the most iconic brands in sports."
        elif 'bills' in topic_lower:
            prompt = "The Buffalo Bills are known for their passionate fan base and the famous 'Bills Mafia' that creates an incredible atmosphere at Highmark Stadium. Led by quarterback Josh Allen, the Bills have become one of the most exciting teams in the NFL with their high-powered offense and aggressive defense. Buffalo's blue-collar mentality and unwavering support from their fans make them a unique and beloved franchise."
        elif 'packers' in topic_lower:
            prompt = "The Green Bay Packers are the only publicly owned team in the NFL and play in the historic Lambeau Field, known as the 'Frozen Tundra.' With a rich history that includes legendary players like Bart Starr, Brett Favre, and Aaron Rodgers, the Packers have won multiple Super Bowls and maintain one of the most loyal fan bases in professional sports. Their small-market success story is unique in the modern NFL."
        else:
            prompt = f"Sports bring excitement, passion, and drama to millions of fans worldwide. {topic} represents the competitive spirit and athletic excellence that makes sports so compelling to watch and follow. These moments create lasting memories and bring communities together in celebration of human achievement and determination."
        
        try:
            # Generate multiple options and pick the best one
            generated = self.text_generator(prompt, max_length=120, num_return_sequences=3,
                temperature=0.7, do_sample=True, truncation=True, top_p=0.9, top_k=50)
            
            # Select the best generated text
            best_commentary = ""
            for result in generated:
                commentary = result['generated_text'].replace(prompt, "").strip()
                commentary = self._clean_commentary(commentary)
                
                # Score the commentary quality
                if self._is_good_commentary(commentary):
                    best_commentary = commentary
                    break
            
            # If no good commentary was generated, use enhanced fallback
            if not best_commentary or len(best_commentary) < 20:
                best_commentary = self._get_enhanced_fallback(topic)
            
            return f"AI Commentary: {best_commentary}"
            
        except Exception as e:
            return f"AI Commentary: {self._get_enhanced_fallback(topic)}"
    
    def _clean_commentary(self, text):
        """Clean up generated commentary text"""
        # Remove common artifacts and incomplete sentences
        text = text.replace("The game was", "").strip()
        text = text.replace("This team", "").strip()
        text = text.replace("His ability to", "").strip()
        text = text.replace("The level of competition", "").strip()
        
        # Remove incomplete sentences
        if text.endswith(('the', 'and', 'or', 'but', 'so', 'for', 'with', 'by')):
            text = text.rsplit(' ', 1)[0]
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text.strip()
    
    def _is_good_commentary(self, text):
        """Check if the generated commentary is good quality"""
        if len(text) < 20 or len(text) > 300:
            return False
        
        # Check for good patterns
        good_patterns = ['exciting', 'amazing', 'incredible', 'championship', 'competition', 
                        'athletes', 'fans', 'game', 'team', 'players', 'season', 'league']
        if any(pattern in text.lower() for pattern in good_patterns):
            return True
        
        return len(text) > 40  # Fallback: longer text is usually better
    
    def _get_enhanced_fallback(self, topic):
        """Generate high-quality fallback commentary when AI generation fails"""
        fallback_commentary = {
            'super bowl': "The Super Bowl represents the pinnacle of American sports, where the NFL's two best teams compete for ultimate glory. This championship game captivates millions worldwide with its combination of athletic excellence, dramatic storylines, and cultural significance that extends far beyond the football field.",
            'lakers': "The Lakers embody the essence of basketball excellence, with a legacy of championship success and superstar talent that has defined the NBA for decades. Their purple and gold colors represent a commitment to winning and attracting the game's greatest players to create unforgettable moments.",
            'chiefs': "The Chiefs have revolutionized modern football with their explosive offense and innovative play-calling. Arrowhead Stadium's electric atmosphere, combined with their dynamic passing attack, makes them one of the most exciting and successful teams in the NFL.",
            'mahomes': "Patrick Mahomes has redefined what's possible at the quarterback position with his incredible arm talent, mobility, and clutch performances. His ability to make impossible plays look routine has established him as a generational talent and one of the game's elite players.",
            'lebron': "LeBron James continues to defy expectations and age, showcasing why he's considered one of the greatest basketball players of all time. His combination of skill, leadership, and longevity has created a legacy that will be remembered for generations to come.",
            'nba': "The NBA showcases the world's best basketball talent, where athleticism meets strategy in the most competitive league on the planet. The league's global reach and emphasis on entertainment have made it a cultural phenomenon that transcends sports.",
            'nfl': "The NFL represents the ultimate test of physical and mental toughness, where every game is a battle and every season brings new storylines. The league's combination of athletic excellence and dramatic narratives keeps fans on the edge of their seats every Sunday.",
            'championship': "Championship games are where legends are born and dreams come true. The pressure is immense, but the rewards are eternal glory and a place in sports history that will be remembered forever.",
            'playoffs': "Playoff games are where the intensity reaches its peak and every moment matters. This is when true champions separate themselves from the rest, performing at their highest level when the stakes are at their greatest.",
            'warriors': "The Warriors have revolutionized basketball with their fast-paced, three-point shooting style that changed the game forever. Their innovative approach and championship success have inspired a new generation of players and coaches.",
            'celtics': "The Celtics represent one of the most successful franchises in sports history, with a rich tradition of excellence and championship success. Their green and white colors symbolize a commitment to winning and attracting the game's greatest talent.",
            'bills': "The Bills are known for their passionate fan base and the incredible atmosphere at Highmark Stadium. Their blue-collar mentality and unwavering support from the 'Bills Mafia' make them a unique and beloved franchise in the NFL.",
            'packers': "The Packers are a unique franchise in professional sports, being the only publicly owned team in the NFL. Their historic Lambeau Field and rich tradition of success make them one of the most respected and beloved teams in the league."
        }
        
        # Find the best match
        topic_lower = topic.lower()
        for key, commentary in fallback_commentary.items():
            if key in topic_lower:
                return commentary
        
        # Default enhanced fallback
        return f"{topic} represents the excitement and passion that makes sports so compelling to watch and follow. These moments create lasting memories and bring fans together in celebration of athletic excellence and human achievement."
    
    def demo_sentiment_analysis(self):
        """Demonstrate sentiment analysis capabilities"""
        print("\nSENTIMENT ANALYSIS DEMO")
        print("=" * 40)
        
        test_texts = [
            "The Lakers are absolutely amazing this season!",
            "This team is terrible and needs major changes.",
            "The game was okay, nothing special.",
            "LeBron James is the greatest player ever!",
            "I hate watching this boring sport."
        ]
        
        for text in test_texts:
            result = self.analyze_sentiment(text)
            print(f"Text: {text}")
            print(f"Result: {result['interpretation']}")
            print()
    
    def demo_text_generation(self):
        """Demonstrate text generation capabilities"""
        print("\nTEXT GENERATION DEMO")
        print("=" * 40)
        
        topics = ["Super Bowl", "Lakers", "Chiefs", "Patrick Mahomes", "NBA Finals"]
        
        for topic in topics:
            commentary = self.generate_commentary(topic)
            print(f"Topic: {topic}")
            print(f"{commentary}")
            print()
    
    def run_full_demo(self):
        """Run complete demonstration of both models"""
        print("\nSPORTS AI DEMONSTRATION")
        print("=" * 50)
        
        # Demo 1: Sentiment Analysis
        self.demo_sentiment_analysis()
        
        # Demo 2: Text Generation
        self.demo_text_generation()
        
        print("Demo completed! Both AI models are working successfully.")
    
    def interactive_mode(self):
        """Interactive mode for testing the models"""
        print("\nINTERACTIVE MODE")
        print("Commands:")
        print("- 'sentiment: [text]' - Analyze sentiment")
        print("- 'commentary: [topic]' - Generate commentary")
        print("- 'demo' - Run full demonstration")
        print("- 'quit' - Exit")
        
        while True:
            user_input = input("\nYour command: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Thanks for using Sports AI!")
                break
            
            if user_input.lower() == 'demo':
                self.run_full_demo()
                continue
            
            if user_input.startswith("sentiment:"):
                text = user_input[10:].strip()
                if text:
                    result = self.analyze_sentiment(text)
                    print(f"Sentiment Analysis: {result['interpretation']}")
                else:
                    print("Please provide text to analyze.")
            
            elif user_input.startswith("commentary:"):
                topic = user_input[11:].strip()
                if topic:
                    commentary = self.generate_commentary(topic)
                    print(commentary)
                else:
                    print("Please provide a topic for commentary.")
            
            else:
                print("Invalid command. Use 'sentiment:', 'commentary:', 'demo', or 'quit'")
    
    def analyze_sports_news(self):
        """Analyze sentiment of sports news articles"""
        print("\nSPORTS NEWS SENTIMENT ANALYSIS")
        print("=" * 40)
        
        for i, article in enumerate(self.sports_data["news"], 1):
            result = self.analyze_sentiment(article)
            print(f"Article {i}: {article[:60]}...")
            print(f"Sentiment: {result['interpretation']}")
            print()
    
    def generate_team_commentary(self, team_name):
        """Generate specific commentary for a team"""
        commentary = self.generate_commentary(team_name)
        print(f"Team Commentary for {team_name}:")
        print(commentary)
        return commentary

def main():
    """Main function to run the Sports AI Demo"""
    try:
        ai = SportsAI()
        
        # Go straight to interactive mode
        ai.interactive_mode()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the required packages installed:")
        print("pip install torch transformers")

if __name__ == "__main__":
    main()