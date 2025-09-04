import gradio as gr
import json
import numpy as np
from lib.ai.multi_model_llm import MultiModelLLMSolver
from lib.algo.ga import EnhancedGAEngine
from lib.data.hf_dataset import HFDatasetManager

class EvolutionArenaGradio:
    def __init__(self):
        print(" Loading Evolution Arena for Hugging Face Spaces...")
        self.llm_solver = MultiModelLLMSolver()
        self.dataset_manager = HFDatasetManager()
        
    def run_evolution(self, problem_name, num_generations, population_size, use_llm):
        try:
            # Find problem in dataset
            problems_data = self.dataset_manager.dataset.filter(lambda x: x["name"] == problem_name)
            if len(problems_data) == 0:
                return "Problem not found!", "", ""
            
            problem = problems_data[0]
            
            # Convert to format expected by GA engine
            problem_dict = {
                "name": problem["name"],
                "type": problem["type"],
                "length": problem["length"],
                "description": problem["description"],
                "difficulty": problem["difficulty"]
            }
            
            # Run evolution
            ga_engine = EnhancedGAEngine(problem_dict)
            ga_engine.size = population_size
            
            results = []
            fitness_history = []
            
            for gen in range(num_generations):
                # Evaluate fitness
                fitness = ga_engine.get_fitness(ga_engine.pop)
                best_fitness = np.max(fitness)
                avg_fitness = np.mean(fitness)
                
                fitness_history.append({
                    "generation": gen,
                    "best_fitness": best_fitness,
                    "avg_fitness": avg_fitness
                })
                
                results.append(f"Gen {gen}: Best={best_fitness:.2f}, Avg={avg_fitness:.2f}")
                
                # Generate next generation
                if use_llm and gen % 3 == 0:  # Use LLM every 3 generations
                    llm_candidates = self.llm_solver.get_candidates(problem_dict, ga_engine.pop, 3)
                    ga_engine.pop = ga_engine.next_gen(ga_engine.pop, llm_candidates)
                    results.append(f"  → Used LLM to generate candidates")
                else:
                    ga_engine.pop = ga_engine.next_gen(ga_engine.pop)
            
            # Final results
            final_fitness = ga_engine.get_fitness(ga_engine.pop)
            best_solution = ga_engine.pop[np.argmax(final_fitness)]
            
            summary = f"""
## Evolution Complete! 

**Problem**: {problem['name']} ({problem['difficulty']})
**Type**: {problem['type']}
**Generations**: {num_generations}
**Population Size**: {population_size}
**Used LLM**: {'Yes' if use_llm else 'No'}

**Final Best Fitness**: {np.max(final_fitness):.2f}
**Best Solution**: {best_solution[:10]}... (truncated)

**Problem Description**: {problem['description']}
            """
            
            return "\n".join(results), summary, str(fitness_history)
            
        except Exception as e:
            return f"Error: {str(e)}", "", ""
    
    def get_problem_info(self, problem_name):
        try:
            problems_data = self.dataset_manager.dataset.filter(lambda x: x["name"] == problem_name)
            if len(problems_data) == 0:
                return "Problem not found!"
            
            problem = problems_data[0]
            return f"""
**Name**: {problem['name']}
**Type**: {problem['type']}
**Difficulty**: {problem['difficulty']}
**Length**: {problem['length']}
**Complexity Score**: {problem['complexity_score']}
**Tags**: {', '.join(problem['tags'])}

**Description**: {problem['description']}
            """
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_dataset_stats(self):
        stats = self.dataset_manager.get_stats()
        return f"""
# Evolution Arena Dataset Stats 

**Total Problems**: {stats['total_problems']}
**Average Complexity**: {stats['avg_complexity']:.2f}
**Average Length**: {stats['avg_length']:.1f}

## Problem Types:
{chr(10).join([f"- {k}: {v}" for k, v in stats['types'].items()])}

## Difficulty Distribution:
{chr(10).join([f"- {k}: {v}" for k, v in stats['difficulties'].items()])}
        """

def create_interface():
    app = EvolutionArenaGradio()
    
    # Get list of problem names for dropdown
    problem_names = app.dataset_manager.dataset["name"]
    
    with gr.Blocks(title=" Evolution Arena - Hugging Face Space", theme=gr.themes.Soft()) as demo:
        gr.Markdown("#  Evolution Arena - LLM-Guided Genetic Algorithms")
        gr.Markdown("**Powered by Multiple Hugging Face Models & Datasets**")
        
        with gr.Tab("🎮 Run Evolution"):
            with gr.Row():
                with gr.Column(scale=1):
                    problem_dropdown = gr.Dropdown(
                        choices=problem_names,
                        label="Select Problem",
                        value=problem_names[0] if problem_names else None
                    )
                    num_generations = gr.Slider(5, 50, value=20, step=1, label="Generations")
                    population_size = gr.Slider(5, 50, value=20, step=5, label="Population Size")
                    use_llm = gr.Checkbox(label="Use LLM Assistance", value=True)
                    
                    run_btn = gr.Button(" Start Evolution", variant="primary")
                
                with gr.Column(scale=2):
                    evolution_log = gr.Textbox(
                        label="Evolution Progress",
                        lines=15,
                        max_lines=20
                    )
            
            with gr.Row():
                summary_output = gr.Markdown(label="Results Summary")
                fitness_data = gr.Textbox(label="Fitness History (JSON)", visible=False)
        
        with gr.Tab("📋 Problem Info"):
            with gr.Row():
                with gr.Column(scale=1):
                    info_problem_dropdown = gr.Dropdown(
                        choices=problem_names,
                        label="Select Problem",
                        value=problem_names[0] if problem_names else None
                    )
                    info_btn = gr.Button("📖 Get Info")
                
                with gr.Column(scale=2):
                    problem_info_output = gr.Markdown()
        
        with gr.Tab(" Dataset Stats"):
            stats_btn = gr.Button("�� Show Dataset Statistics")
            stats_output = gr.Markdown()
        
        # Event handlers
        run_btn.click(
            app.run_evolution,
            inputs=[problem_dropdown, num_generations, population_size, use_llm],
            outputs=[evolution_log, summary_output, fitness_data]
        )
        
        info_btn.click(
            app.get_problem_info,
            inputs=[info_problem_dropdown],
            outputs=[problem_info_output]
        )
        
        stats_btn.click(
            app.get_dataset_stats,
            outputs=[stats_output]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
