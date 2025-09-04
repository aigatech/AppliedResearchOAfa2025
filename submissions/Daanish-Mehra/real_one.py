import numpy as np
import json
import random
import time
import matplotlib.pyplot as plt
from lib.algo.ga import EnhancedGAEngine
from lib.ai.llm import EnhancedLLMSolver
from lib.rl.rl_loop import get_reward, update_rl_state, init_rl_state

class RealEvolutionArena:
    def __init__(self):
        self.problems = self.load_problems()
        self.score = 0
        self.level = 1
        self.unlocked = set(range(1, 11))
        self.rl_state = init_rl_state()
        self.fitness_history = []
        self.problem_history = []
        print("=" * 60)
        print("REAL EVOLUTION ARENA - FULL LLM INTEGRATION")
        print("=" * 60)
        print("Loading Hugging Face LLM...")
        self.llm_solver = EnhancedLLMSolver()
        print("LLM Ready! Starting evolution...")
        print()
    
    def load_problems(self):
        with open('lib/problems/probs.json', 'r') as f:
            return json.load(f)
    
    def run_session(self, num_problems=5):
        print(f"Starting {num_problems} problem evolution session...")
        print("=" * 60)
        
        for i in range(num_problems):
            problem_id = random.choice(list(self.unlocked))
            problem = self.problems[problem_id - 1]
            
            print(f"\nPROBLEM {i+1}/{num_problems}")
            print("-" * 40)
            print(f"Name: {problem['name']}")
            print(f"Type: {problem['type'].upper()}")
            print(f"Difficulty: {problem['difficulty'].upper()}")
            print(f"Description: {problem['description']}")
            print()
            
            result = self.solve_problem(problem)
            self.score += result['score']
            self.fitness_history.append(result['best_fitness'])
            self.problem_history.append(problem['name'])
            
            print(f"BEST FITNESS: {result['best_fitness']:.2f}")
            print(f"SCORE EARNED: {result['score']}")
            print(f"TOTAL SCORE: {self.score}")
            print(f"LLM EFFECTIVENESS: {self.rl_state['llm_effectiveness']:.3f}")
            
            if result['score'] > 50:
                self.level += 1
                new_problems = list(range(11, min(21, 11 + self.level * 3)))
                self.unlocked.update(new_problems)
                print(f"LEVEL UP! Unlocked {len(new_problems)} new problems")
            
            print("-" * 40)
        
        self.show_final_results()
        return self.score
    
    def solve_problem(self, problem):
        engine = EnhancedGAEngine(problem)
        best_fitness = 0
        generations = 10
        gen_fitness = []
        
        print("Evolution Progress:")
        for gen in range(generations):
            fitness = engine.get_fitness(engine.pop)
            current_best = max(fitness)
            best_fitness = max(best_fitness, current_best)
            gen_fitness.append(current_best)
            
            if random.random() < self.rl_state['llm_effectiveness']:
                llm_candidates = self.llm_solver.get_candidates(problem, engine.pop, 3)
                engine.pop = engine.next_gen(engine.pop, llm_candidates)
                llm_used = "LLM"
            else:
                engine.pop = engine.next_gen(engine.pop)
                llm_used = "GA"
            
            if gen % 2 == 0:
                progress_bar = "█" * int((gen / generations) * 20) + "░" * (20 - int((gen / generations) * 20))
                print(f"Gen {gen:2d}: [{progress_bar}] {current_best:6.2f} ({llm_used})")
        
        self.plot_evolution(problem['name'], gen_fitness)
        
        score = int(best_fitness * 10)
        reward = get_reward([0], [best_fitness])
        self.rl_state = update_rl_state(self.rl_state, reward, problem['id'])
        
        return {
            'best_fitness': best_fitness,
            'score': score,
            'generations': generations
        }
    
    def plot_evolution(self, problem_name, fitness_history):
        plt.figure(figsize=(8, 4))
        plt.plot(fitness_history, 'r-', linewidth=2, marker='s', markersize=4)
        plt.title(f'Evolution Progress: {problem_name}', fontsize=12, fontweight='bold')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
    
    def show_final_results(self):
        print("\n" + "=" * 60)
        print("SESSION COMPLETE - FINAL RESULTS")
        print("=" * 60)
        print(f"Total Score: {self.score}")
        print(f"Problems Solved: {len(self.problem_history)}")
        print(f"Final Level: {self.level}")
        print(f"LLM Effectiveness: {self.rl_state['llm_effectiveness']:.3f}")
        print(f"Average Reward: {self.rl_state['avg_reward']:.2f}")
        print()
        
        print("Problem Performance:")
        for i, (name, fitness) in enumerate(zip(self.problem_history, self.fitness_history)):
            print(f"  {i+1}. {name}: {fitness:.2f}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    arena = RealEvolutionArena()
    arena.run_session(5)
