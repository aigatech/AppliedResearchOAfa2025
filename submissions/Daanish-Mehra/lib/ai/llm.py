from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
import re
import random

class EnhancedLLMSolver:
    def __init__(self):
        self.model_name = "google/flan-t5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
    
    def get_candidates(self, problem, pop, max_candidates=5):
        try:
            prompt = self.make_prompt(problem, pop, max_candidates)
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.model.generate(inputs.input_ids, max_new_tokens=200)
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            candidates = self.parse_candidates(text, problem, max_candidates)
            if isinstance(candidates, np.ndarray):
                candidates = candidates.tolist()
            while len(candidates) < max_candidates:
                random_candidates = self.make_random(problem, max_candidates - len(candidates))
                candidates.extend(random_candidates)
            return candidates[:max_candidates]
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return self.make_random(problem, max_candidates)
    
    def make_prompt(self, problem, pop, max_candidates=5):
        name = problem["name"]
        typ = problem["type"]
        pop_str = str(pop[:2])
        length = problem.get("length", 5)
        return f"solve {name} {typ} problem, current pop: {pop_str}, make {max_candidates} new solutions"
    
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
        return np.array(candidates) if candidates else np.array([]).reshape(0, length)
    
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
        return np.array(candidates) if candidates else np.array([]).reshape(0, length)
    
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
        return np.array(candidates) if candidates else np.array([]).reshape(0, length)
    
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
