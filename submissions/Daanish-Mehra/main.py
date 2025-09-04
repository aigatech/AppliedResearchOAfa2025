import sys
import os
import time

def show_banner():
    print("=" * 70)
    print("AI EVOLUTION CHALLENGE")
    print("=" * 70)
    print("LLMs and Genetic Algorithms compete to solve problems!")
    print("ENHANCED WITH HF MODELS")
    print()

def show_mode_menu():
    print("EVOLUTION MODES:")
    print()
    print("1. Fast Learning Arena")
    print("   - Quick evolution with basic LLM integration")
    print("   - Perfect for beginners")
    print()
    print("2. Real Evolution Arena") 
    print("   - Full LLM integration with Hugging Face")
    print("   - Advanced reinforcement learning")
    print("   - For experienced players")
    print()
    print("3. Custom Challenge")
    print("   - Choose specific problem types")
    print("   - Set custom parameters")
    print()
    print("4. RL Tournament Mode")
    print("   - LLM vs GA")
    print("   - Advanced Q-learning agents")
    print("   - Watch AI agents learn and compete")
    print()
    print("5. Multi-Model Demo")
    print("   - Test different HF models")
    print()
    print("6. Gradio Space")
    print("   - Web interface")
    print()
    print("7. Exit")
    print("=" * 70)

def get_problem_count():
    while True:
        try:
            count = int(input("How many problems do you want to solve? (1-20): "))
            if 1 <= count <= 20:
                return count
            else:
                print("Please enter a number between 1 and 20.")
        except ValueError:
            print("Please enter a valid number.")

def get_round_count():
    while True:
        try:
            count = int(input("How many rounds do you want? (1-10): "))
            if 1 <= count <= 10:
                return count
            else:
                print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Please enter a valid number.")

def get_difficulty():
    print("\nDIFFICULTY LEVELS:")
    print("1. Easy (problems 1-20)")
    print("2. Medium (problems 21-50)")
    print("3. Hard (problems 51-80)")
    print("4. Expert (problems 81-100)")
    print("5. Random (all problems)")
    
    while True:
        choice = input("Enter choice (1-5): ").strip()
        if choice in ["1", "2", "3", "4", "5"]:
            return int(choice)
        print("Invalid choice! Please enter 1-5.")

def get_problem_types():
    print("\nPROBLEM TYPES:")
    print("1. Bitstring (OneMax, LeadingOnes, Trap)")
    print("2. Permutation (N-Queens, TSP, Sudoku)")
    print("3. Expression (Symbolic Regression)")
    print("4. Constraint (Bin Packing, Scheduling)")
    print("5. Combinatorial (Graph Coloring, Max-Cut)")
    print("6. All types")
    
    while True:
        choice = input("Enter choices (e.g., 1,2,3 or 6): ").strip()
        if choice == "6":
            return ["all"]
        try:
            types = [int(x.strip()) for x in choice.split(",")]
            if all(1 <= t <= 5 for t in types):
                return types
        except:
            pass
        print("Invalid choice! Please enter numbers 1-5 separated by commas.")

def show_loading_animation():
    print("\nInitializing AI systems...")
    for i in range(3):
        print("Loading" + "." * (i + 1), end="\r")
        time.sleep(0.5)
    print("Ready!                    ")

def run_multi_model_demo():
    print("Testing Multiple Hugging Face Models...")
    from lib.ai.multi_model_llm import MultiModelLLMSolver
    from lib.data.hf_dataset import HFDatasetManager
    
    llm_solver = MultiModelLLMSolver()
    dataset_manager = HFDatasetManager()
    
    print("\nDataset Statistics:")
    stats = dataset_manager.get_stats()
    print(f"Total problems: {stats['total_problems']}")
    print(f"Types: {list(stats['types'].keys())}")
    print(f"Difficulties: {list(stats['difficulties'].keys())}")
    
    test_problems = dataset_manager.get_random_problems(3)
    
    for i, problem in enumerate(test_problems):
        print(f"\nTesting Problem {i+1}: {problem['name']}")
        print(f"Type: {problem['type']}, Difficulty: {problem['difficulty']}")
        
        import numpy as np
        if problem['type'] == 'bitstring':
            pop = np.random.randint(0, 2, (5, problem['length']))
        elif problem['type'] == 'permutation':
            pop = np.array([np.random.permutation(problem['length']) for _ in range(5)])
        else:
            pop = np.random.rand(5, problem.get('length', 5))
        
        try:
            candidates = llm_solver.get_candidates(problem, pop, 3)
            print(f"Generated {len(candidates)} candidates")
        except Exception as e:
            print(f"LLM failed: {e}")

def run_hf_space():
    print("Starting Hugging Face Space Interface...")
    from gradio_space import create_interface
    demo = create_interface()
    demo.launch(share=True)

def main():
    show_banner()
    
    while True:
        show_mode_menu()
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == "1":
            print("\n" + "=" * 50)
            print("FAST LEARNING ARENA SELECTED")
            print("=" * 50)
            num_problems = get_problem_count()
            show_loading_animation()
            from fast_one import FastLearningArena
            arena = FastLearningArena()
            arena.run_session(num_problems)
            break
            
        elif choice == "2":
            print("\n" + "=" * 50)
            print("REAL EVOLUTION ARENA SELECTED")
            print("=" * 50)
            num_problems = get_problem_count()
            show_loading_animation()
            from real_one import RealEvolutionArena
            arena = RealEvolutionArena()
            arena.run_session(num_problems)
            break
            
        elif choice == "3":
            print("\n" + "=" * 50)
            print("CUSTOM CHALLENGE SELECTED")
            print("=" * 50)
            num_problems = get_problem_count()
            difficulty = get_difficulty()
            problem_types = get_problem_types()
            show_loading_animation()
            from custom_arena import CustomArena
            arena = CustomArena(difficulty, problem_types)
            arena.run_session(num_problems)
            break
            
        elif choice == "4":
            print("\n" + "=" * 50)
            print("RL TOURNAMENT MODE SELECTED")
            print("=" * 50)
            num_rounds = get_round_count()
            show_loading_animation()
            from rl_tournament_arena import RLTournamentArena
            arena = RLTournamentArena()
            arena.run_session(num_rounds)
            break
            
        elif choice == "5":
            print("\n" + "=" * 50)
            print("MULTI-MODEL DEMO SELECTED")
            print("=" * 50)
            run_multi_model_demo()
            break
            
        elif choice == "6":
            print("\n" + "=" * 50)
            print("GRADIO SPACE SELECTED")
            print("=" * 50)
            run_hf_space()
            break
            
        elif choice == "7":
            print("\nThanks for using Evolution Arena!")
            sys.exit(0)
        else:
            print("Invalid choice! Please enter 1-7.")
            print()

if __name__ == "__main__":
    main()
