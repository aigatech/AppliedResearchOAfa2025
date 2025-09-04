import numpy as np
import json
import random
import time
from lib.algo.ga import EnhancedGAEngine
from lib.ai.llm import EnhancedLLMSolver
from lib.rl.rl_agent import RL_agent

class TournamentMode:
    def __init__(self):
        self.problems = self.load_problems()
        self.llm_agent = RL_agent("llm")
        self.ga_agent = RL_agent("ga")
        self.llm_solver = EnhancedLLMSolver()
        self.tournament_results = []
        
        print("=" * 60)
        print("TOURNAMENT MODE - LLM vs GA COMPETITION")
        print("=" * 60)
        print("Loading AI agents...")
        print("LLM Agent: Ready for battle!")
        print("GA Agent: Ready for battle!")
        print()
    
    def load_problems(self):
        with open('lib/problems/probs.json', 'r') as f:
            return json.load(f)
    
    def run_tournament(self, num_rounds=5):
        print(f"Starting {num_rounds} round tournament...")
        print("=" * 60)
        
        llm_wins = 0
        ga_wins = 0
        ties = 0
        
        for round_num in range(num_rounds):
            print(f"\nROUND {round_num + 1}/{num_rounds}")
            print("-" * 40)
            
            problem = random.choice(self.problems[:20])
            print(f"Problem: {problem['name']}")
            print(f"Type: {problem['type'].upper()}")
            print(f"Difficulty: {problem['difficulty'].upper()}")
            print()
            
            llm_result = self.run_llm_agent(problem, round_num)
            ga_result = self.run_ga_agent(problem, round_num)
            
            print(f"LLM Agent: {llm_result['best_fitness']:.2f}")
            print(f"GA Agent:  {ga_result['best_fitness']:.2f}")
            
            if llm_result['best_fitness'] > ga_result['best_fitness']:
                llm_wins += 1
                winner = "LLM Agent"
            elif ga_result['best_fitness'] > llm_result['best_fitness']:
                ga_wins += 1
                winner = "GA Agent"
            else:
                ties += 1
                winner = "TIE"
            
            print(f"Winner: {winner}")
            print("-" * 40)
            
            self.tournament_results.append({
                'round': round_num + 1,
                'problem': problem['name'],
                'llm_fitness': llm_result['best_fitness'],
                'ga_fitness': ga_result['best_fitness'],
                'winner': winner
            })
        
        self.show_tournament_results(llm_wins, ga_wins, ties)
        return llm_wins, ga_wins, ties
    
    def run_llm_agent(self, problem, round_num):
        engine = EnhancedGAEngine(problem)
        best_fitness = 0
        generations = 10
        
        print("LLM Agent Evolution:")
        for gen in range(generations):
            fitness = engine.get_fitness(engine.pop)
            current_best = max(fitness)
            best_fitness = max(best_fitness, current_best)
            
            state_key = self.llm_agent.get_state_key(problem, current_best, gen)
            action = self.llm_agent.get_action(state_key, ["use_llm", "use_ga"])
            
            if action == "use_llm":
                llm_candidates = self.llm_solver.get_candidates(problem, engine.pop, 3)
                engine.pop = engine.next_gen(engine.pop, llm_candidates)
                action_used = "LLM"
            else:
                engine.pop = engine.next_gen(engine.pop)
                action_used = "GA"
            
            if gen % 2 == 0:
                progress_bar = "█" * int((gen / generations) * 20) + "░" * (20 - int((gen / generations) * 20))
                print(f"Gen {gen:2d}: [{progress_bar}] {current_best:6.2f} ({action_used})")
            
            if gen > 0:
                old_fitness = max(engine.get_fitness(engine.pop)) if gen > 0 else current_best
                reward = self.llm_agent.calculate_reward(old_fitness, current_best, problem)
                next_state_key = self.llm_agent.get_state_key(problem, current_best, gen + 1)
                self.llm_agent.update_q_value(state_key, action, reward, next_state_key)
                self.llm_agent.performance_history.append(reward)
            
            time.sleep(0.3)
        
        self.llm_agent.decay_epsilon()
        return {'best_fitness': best_fitness}
    
    def run_ga_agent(self, problem, round_num):
        engine = EnhancedGAEngine(problem)
        best_fitness = 0
        generations = 10
        
        print("GA Agent Evolution:")
        for gen in range(generations):
            fitness = engine.get_fitness(engine.pop)
            current_best = max(fitness)
            best_fitness = max(best_fitness, current_best)
            
            state_key = self.ga_agent.get_state_key(problem, current_best, gen)
            action = self.ga_agent.get_action(state_key, ["mutate", "crossover", "both"])
            
            if action == "mutate":
                for i in range(len(engine.pop)):
                    if random.random() < 0.7:
                        engine.pop[i] = engine.mutate(engine.pop[i])
                action_used = "MUTATE"
            elif action == "crossover":
                for i in range(0, len(engine.pop), 2):
                    if i + 1 < len(engine.pop):
                        engine.pop[i], engine.pop[i+1] = engine.cross(engine.pop[i], engine.pop[i+1])
                action_used = "CROSS"
            else:
                for i in range(len(engine.pop)):
                    if random.random() < 0.7:
                        engine.pop[i] = engine.mutate(engine.pop[i])
                for i in range(0, len(engine.pop), 2):
                    if i + 1 < len(engine.pop):
                        engine.pop[i], engine.pop[i+1] = engine.cross(engine.pop[i], engine.pop[i+1])
                action_used = "BOTH"
            
            if gen % 2 == 0:
                progress_bar = "█" * int((gen / generations) * 20) + "░" * (20 - int((gen / generations) * 20))
                print(f"Gen {gen:2d}: [{progress_bar}] {current_best:6.2f} ({action_used})")
            
            if gen > 0:
                old_fitness = max(engine.get_fitness(engine.pop)) if gen > 0 else current_best
                reward = self.ga_agent.calculate_reward(old_fitness, current_best, problem)
                next_state_key = self.ga_agent.get_state_key(problem, current_best, gen + 1)
                self.ga_agent.update_q_value(state_key, action, reward, next_state_key)
                self.ga_agent.performance_history.append(reward)
            
            time.sleep(0.3)
        
        self.ga_agent.decay_epsilon()
        return {'best_fitness': best_fitness}
    
    def show_tournament_results(self, llm_wins, ga_wins, ties):
        print("\n" + "=" * 60)
        print("TOURNAMENT RESULTS")
        print("=" * 60)
        print(f"LLM Agent Wins: {llm_wins}")
        print(f"GA Agent Wins:  {ga_wins}")
        print(f"Ties:           {ties}")
        print()
        
        if llm_wins > ga_wins:
            print(" LLM AGENT WINS THE TOURNAMENT!")
        elif ga_wins > llm_wins:
            print(" GA AGENT WINS THE TOURNAMENT!")
        else:
            print("🤝 TOURNAMENT ENDS IN A TIE!")
        
        print("\nRound-by-Round Results:")
        for result in self.tournament_results:
            print(f"Round {result['round']}: {result['problem']}")
            print(f"  LLM: {result['llm_fitness']:.2f} | GA: {result['ga_fitness']:.2f} | Winner: {result['winner']}")
        
        print("\nAgent Performance Stats:")
        llm_stats = self.llm_agent.get_performance_stats()
        ga_stats = self.ga_agent.get_performance_stats()
        
        print(f"LLM Agent - Avg Reward: {llm_stats['avg_reward']:.2f}, Success Rate: {llm_stats['success_rate']:.2f}")
        print(f"GA Agent  - Avg Reward: {ga_stats['avg_reward']:.2f}, Success Rate: {ga_stats['success_rate']:.2f}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    tournament = TournamentMode()
    tournament.run_tournament(5)
