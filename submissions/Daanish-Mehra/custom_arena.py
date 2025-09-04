import numpy as np
import json
import random
import time
from lib.algo.ga import EnhancedGAEngine
from lib.ai.llm import EnhancedLLMSolver
from lib.rl.rl_loop import get_reward, update_rl_state, init_rl_state

class CustomArena:
    def __init__(self, difficulty, problem_types):
        self.problems = self.load_problems()
        self.score = 0
        self.level = 1
        self.difficulty = difficulty
        self.problem_types = problem_types
        self.unlocked = self.get_unlocked_problems()
        self.rl_state = init_rl_state()
        self.fitness_history = []
        self.problem_history = []
        
        print("=" * 60)
        print("CUSTOM CHALLENGE ARENA - TAILORED EVOLUTION")
        print("=" * 60)
        print("Initializing LLM Solver...")
        self.llm_solver = EnhancedLLMSolver()
        print("Ready for custom challenge!")
        print()
    
    def load_problems(self):
        with open('lib/problems/probs.json', 'r') as f:
            return json.load(f)
    
    def get_unlocked_problems(self):
        if self.difficulty == 1:
            return set(range(1, 21))
        elif self.difficulty == 2:
            return set(range(21, 51))
        elif self.difficulty == 3:
            return set(range(51, 81))
        elif self.difficulty == 4:
            return set(range(81, 101))
        else:
            return set(range(1, 101))
    
    def filter_problems_by_type(self, problem):
        if "all" in self.problem_types:
            return True
        
        type_map = {
            1: "bitstring",
            2: "permutation", 
            3: "expression",
            4: "constraint",
            5: "combinatorial"
        }
        
        for t in self.problem_types:
            if problem["type"] == type_map[t]:
                return True
        return False
    
    def run_session(self, num_problems=5):
        print(f"Starting {num_problems} problem custom challenge...")
        print("=" * 60)
        
        available_problems = [p for p in self.problems if p["id"] in self.unlocked and self.filter_problems_by_type(p)]
        
        if not available_problems:
            print("No problems available with your criteria!")
            return 0
        
        for i in range(num_problems):
            problem = random.choice(available_problems)
            
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
                print(f"LEVEL UP! You're getting better!")
            
            print("-" * 40)
        
        self.show_final_results()
        return self.score
    
    def solve_problem(self, problem):
        engine = EnhancedGAEngine(problem)
        best_fitness = 0
        generations = 10
        gen_fitness = []
        
        print("Evolution Progress:")
        print("Generation | Fitness | Algorithm | Progress")
        print("-" * 50)
        
        for gen in range(generations):
            fitness = engine.get_fitness(engine.pop)
            current_best = max(fitness)
            best_fitness = max(best_fitness, current_best)
            gen_fitness.append(current_best)
            
            if random.random() < self.rl_state['llm_effectiveness']:
                print(f"Gen {gen:2d}     | {current_best:6.2f}  | LLM       | ", end="")
                llm_candidates = self.llm_solver.get_candidates(problem, engine.pop, 3)
                engine.pop = engine.next_gen(engine.pop, llm_candidates)
                llm_used = "LLM"
            else:
                print(f"Gen {gen:2d}     | {current_best:6.2f}  | GA        | ", end="")
                engine.pop = engine.next_gen(engine.pop)
                llm_used = "GA"
            
            progress_bar = "█" * int((gen / generations) * 20) + "░" * (20 - int((gen / generations) * 20))
            print(f"[{progress_bar}]")
            
            time.sleep(0.5)
        
        print("-" * 50)
        print(f"Evolution complete! Best fitness: {best_fitness:.2f}")
        
        score = int(best_fitness * 10)
        reward = get_reward([0], [best_fitness])
        self.rl_state = update_rl_state(self.rl_state, reward, problem['id'])
        
        return {
            'best_fitness': best_fitness,
            'score': score,
            'generations': generations
        }
    
    def show_final_results(self):
        print("\n" + "=" * 60)
        print("CUSTOM CHALLENGE COMPLETE - FINAL RESULTS")
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
    arena = CustomArena(1, [1, 2])
    arena.run_session(5)
