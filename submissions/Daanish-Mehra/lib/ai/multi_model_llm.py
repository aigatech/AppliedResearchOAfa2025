from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
import torch
import numpy as np
import re
import random

class MultiModelLLMSolver:
    def __init__(self):
        self.models = {
            "bitstring": {
                "model_name": "google/flan-t5-base",
                "tokenizer": None,
                "model": None
            },
            "permutation": {
                "model_name": "microsoft/DialoGPT-small", 
                "tokenizer": None,
                "model": None
            },
            "expression": {
                "model_name": "facebook/blenderbot-400M-distill",
                "tokenizer": None,
                "model": None
            },
            "default": {
                "model_name": "google/flan-t5-small",
                "tokenizer": None,
                "model": None
            }
        }
        self.load_models()
    
    def load_models(self):
        print("Loading multiple Hugging Face models...")
        for model_type, config in self.models.items():
            try:
                if "flan-t5" in config["model_name"]:
                    config["tokenizer"] = AutoTokenizer.from_pretrained(config["model_name"])
                    config["model"] = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])
                else:
                    config["tokenizer"] = AutoTokenizer.from_pretrained(config["model_name"])
                    config["model"] = AutoModelForCausalLM.from_pretrained(config["model_name"])
                print(f" Loaded {config['model_name']} for {model_type}")
            except Exception as e:
                print(f" Failed to load {config['model_name']}: {e}")
                config["tokenizer"] = None
                config["model"] = None
    
    def get_candidates(self, problem, pop, max_candidates=5):
        problem_type = problem["type"]
        model_config = self.models.get(problem_type, self.models["default"])
        
        if model_config["model"] is None:
            return self.make_random(problem, max_candidates)
        
        try:
            prompt = self.make_prompt(problem, pop, max_candidates)
            candidates = self.generate_with_model(model_config, prompt, problem, max_candidates)
            
            while len(candidates) < max_candidates:
                random_candidates = self.make_random(problem, max_candidates - len(candidates))
                candidates.extend(random_candidates)
            return candidates[:max_candidates]
        except Exception as e:
            print(f"LLM generation failed with {model_config['model_name']}: {e}")
            return self.make_random(problem, max_candidates)
    
    def generate_with_model(self, model_config, prompt, problem, max_candidates):
        tokenizer = model_config["tokenizer"]
        model = model_config["model"]
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            if "flan-t5" in model_config["model_name"]:
                outputs = model.generate(inputs.input_ids, max_new_tokens=200, do_sample=True, temperature=0.7)
            else:
                outputs = model.generate(inputs.input_ids, max_new_tokens=200, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.parse_candidates(text, problem, max_candidates)
    
    def make_prompt(self, problem, pop, max_candidates=5):
        name = problem["name"]
        typ = problem["type"]
        pop_str = str(pop[:2])
        length = problem.get("length", 5)
        
        if typ == "bitstring":
            return f"Generate {max_candidates} bitstring solutions for {name}. Length: {length}. Current: {pop_str}. Format: [1,0,1,0]"
        elif typ == "permutation":
            return f"Generate {max_candidates} permutation solutions for {name}. Length: {length}. Current: {pop_str}. Format: [2,0,1,3]"
        else:
            return f"Generate {max_candidates} solutions for {name} {typ} problem. Length: {length}. Current: {pop_str}. Format: [1.0, 2.0, 0.5]"
    
    def parse_candidates(self, text, problem, max_candidates):
        problem_type = problem["type"]
        if problem_type == "bitstring":
            return self.parse_bitstring_candidates(text, problem, max_candidates)
        elif problem_type == "permutation":
            return self.parse_permutation_candidates(text, problem, max_candidates)
        else:
            return self.parse_continuous_candidates(text, problem, max_candidates)
    
    def parse_bitstring_candidates(self, text, problem, max_candidates):
        candidates = []
        length = problem["length"]
        lines = text.split('\n')
        for line in lines:
            if '[' in line and ']' in line:
                try:
                    start = line.find('[')
                    end = line.find(']') + 1
                    candidate_str = line[start:end]
                    candidate = eval(candidate_str)
                    if isinstance(candidate, list) and len(candidate) == length:
                        if all(x in [0, 1] for x in candidate):
                            candidates.append(candidate)
                except:
                    continue
        return candidates
    
    def parse_permutation_candidates(self, text, problem, max_candidates):
        candidates = []
        length = problem["length"]
        lines = text.split('\n')
        for line in lines:
            if '[' in line and ']' in line:
                try:
                    start = line.find('[')
                    end = line.find(']') + 1
                    candidate_str = line[start:end]
                    candidate = eval(candidate_str)
                    if isinstance(candidate, list) and len(candidate) == length:
                        if set(candidate) == set(range(length)):
                            candidates.append(candidate)
                except:
                    continue
        return candidates
    
    def parse_continuous_candidates(self, text, problem, max_candidates):
        candidates = []
        length = problem.get("length", 5)
        lines = text.split('\n')
        for line in lines:
            if '[' in line and ']' in line:
                try:
                    start = line.find('[')
                    end = line.find(']') + 1
                    candidate_str = line[start:end]
                    candidate = eval(candidate_str)
                    if isinstance(candidate, list) and len(candidate) == length:
                        candidates.append(candidate)
                except:
                    continue
        return candidates
    
    def make_random(self, problem, num_candidates):
        candidates = []
        problem_type = problem["type"]
        if problem_type == "bitstring":
            length = problem["length"]
            for _ in range(num_candidates):
                candidate = np.random.randint(0, 2, length).tolist()
                candidates.append(candidate)
        elif problem_type == "permutation":
            length = problem["length"]
            for _ in range(num_candidates):
                candidate = list(range(length))
                random.shuffle(candidate)
                candidates.append(candidate)
        elif problem_type == "expression":
            for _ in range(num_candidates):
                candidate = np.random.rand(5).tolist()
                candidates.append(candidate)
        else:
            length = problem.get("length", 5)
            for _ in range(num_candidates):
                candidate = np.random.rand(length).tolist()
                candidates.append(candidate)
        return candidates
