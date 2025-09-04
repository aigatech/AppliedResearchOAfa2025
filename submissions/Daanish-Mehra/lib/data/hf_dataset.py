from datasets import Dataset, DatasetDict
import json
import numpy as np

class HFDatasetManager:
    def __init__(self, problems_file="lib/problems/probs.json"):
        self.problems_file = problems_file
        self.dataset = None
        self.load_problems_as_dataset()
    
    def load_problems_as_dataset(self):
        print("Converting problems to Hugging Face Dataset format...")
        
        with open(self.problems_file, 'r') as f:
            problems = json.load(f)
        
        dataset_dict = {
            "name": [],
            "type": [],
            "difficulty": [],
            "length": [],
            "description": [],
            "fitness_description": [],
            "tags": [],
            "complexity_score": []
        }
        
        for problem in problems:
            dataset_dict["name"].append(problem["name"])
            dataset_dict["type"].append(problem["type"])
            dataset_dict["difficulty"].append(problem["difficulty"])
            dataset_dict["length"].append(problem.get("length", 0))
            dataset_dict["description"].append(problem["description"])
            dataset_dict["fitness_description"].append(problem.get("fitness_description", ""))
            
            # Add tags based on problem type
            tags = [problem["type"], problem["difficulty"]]
            if "OneMax" in problem["name"]:
                tags.append("counting")
            elif "Queen" in problem["name"]:
                tags.append("constraint")
            elif "TSP" in problem["name"]:
                tags.append("routing")
            elif "Sudoku" in problem["name"]:
                tags.append("puzzle")
            
            dataset_dict["tags"].append(tags)
            
            # Calculate complexity score
            complexity = self.calculate_complexity(problem)
            dataset_dict["complexity_score"].append(complexity)
        
        self.dataset = Dataset.from_dict(dataset_dict)
        print(f" Created HF Dataset with {len(self.dataset)} problems")
        print(f"  - Types: {set(self.dataset['type'])}")
        print(f"  - Difficulties: {set(self.dataset['difficulty'])}")
        
    def calculate_complexity(self, problem):
        base_score = {
            "easy": 1,
            "medium": 2, 
            "hard": 3,
            "expert": 4
        }.get(problem["difficulty"], 2)
        
        length = problem.get("length", 10)
        length_factor = min(length / 50.0, 2.0)
        
        type_factor = {
            "bitstring": 1.0,
            "permutation": 1.5,
            "expression": 2.0,
            "constraint": 2.5,
            "combinatorial": 3.0
        }.get(problem["type"], 1.0)
        
        return round(base_score * length_factor * type_factor, 2)
    
    def filter_by_difficulty(self, difficulty):
        return self.dataset.filter(lambda x: x["difficulty"] == difficulty)
    
    def filter_by_type(self, problem_type):
        return self.dataset.filter(lambda x: x["type"] == problem_type)
    
    def get_random_problems(self, n=10):
        indices = np.random.choice(len(self.dataset), min(n, len(self.dataset)), replace=False)
        return self.dataset.select(indices)
    
    def save_to_hub(self, repo_name="evolution-arena-problems"):
        print(f"Saving dataset to Hugging Face Hub: {repo_name}")
        try:
            self.dataset.push_to_hub(repo_name)
            print(" Dataset saved to HF Hub!")
        except Exception as e:
            print(f" Failed to save to HF Hub: {e}")
    
    def get_stats(self):
        stats = {
            "total_problems": len(self.dataset),
            "types": dict(self.dataset.to_pandas()["type"].value_counts()),
            "difficulties": dict(self.dataset.to_pandas()["difficulty"].value_counts()),
            "avg_complexity": np.mean(self.dataset["complexity_score"]),
            "avg_length": np.mean([x for x in self.dataset["length"] if x > 0])
        }
        return stats
