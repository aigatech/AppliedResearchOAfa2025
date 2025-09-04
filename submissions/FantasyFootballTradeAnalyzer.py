from transformers import pipeline
import random

class FantasyFootballTradeAnalyzer:
    def __init__(self):
        self.generator = pipeline("text-generation", model="gpt2")
        
        self.teams = {
            "49ers": {"RB": ["Christian McCaffrey", "Isaac Gurendo"], "WR": ["Tyreek Hill", "Travis Kelce", "Mecole Hardman", "Kadarius Toney"]},
            "Patriots": {"RB": ["Rhamondre Stevenson", "Rex Burkehead", "James White"], "WR": ["Julain Edelman", "Stefon Diggs"]},
            "Seahawks": {"RB": ["Marshawn Lynch"], "WR": ["Tyler Lockett", "DK Metcalf", "JSN"]},
        }
    
    def getTrade(self):
        for team1 in self.teams:
            for team2 in self.teams:
                if team1 != team2:
                    if len(self.teams[team1]["WR"]) > 3 and len(self.teams[team2]["RB"]) > 2:
                        wr_player = random.choice(self.teams[team1]["WR"])
                        rb_player = random.choice(self.teams[team2]["RB"])
                        return {
                            "team1": team1,
                            "team2": team2, 
                            "team1_gives": wr_player,
                            "team2_gives": rb_player
                        }
        return None
    
    def expTrade(self, trade):
        prompt = "Fantasy football trade: " + trade["team1_gives"] + " for " + trade["team2_gives"] + ". This trade works because"
        
        result = self.generator(
                prompt, 
                max_new_tokens=15, 
                do_sample=True,
                pad_token_id=50256
            )
        explanation = result[0]['generated_text'][len(prompt):].strip()
        return explanation.split('.')[0] + '.'
        
    
    def run(self):
        trade = self.getTrade()
        if trade:
            print(trade['team1'] + " sends: " + trade['team1_gives'])
            print(trade['team2'] + " sends: " + trade['team2_gives'])
            print("\nThis trade is good because " + self.expTrade(trade)) 
        else:
            print("No good trades found.")

if __name__ == "__main__":
    analyzer = FantasyFootballTradeAnalyzer()
    analyzer.run()
