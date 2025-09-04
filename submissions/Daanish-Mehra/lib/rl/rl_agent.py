import numpy as np
import random
import json
from collections import deque

class RL_agent:
    def __init__(self, agent_type="llm"):
        self.agent_type = agent_type
        self.experience_buffer = deque(maxlen=1000)
        self.q_table = {}
        self.epsilon = 0.3
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.performance_history = []
        
    def get_state_key(self, problem_info, current_fitness, generation):
        problem_type = problem_info["type"]
        difficulty = problem_info["difficulty"]
        fitness_bucket = int(current_fitness / 5) * 5
        gen_bucket = int(generation / 2) * 2
        return f"{problem_type}_{difficulty}_{fitness_bucket}_{gen_bucket}"
    
    def get_action(self, state_key, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in available_actions}
        
        q_values = self.q_table[state_key]
        best_action = max(q_values, key=q_values.get)
        return best_action
    
    def update_q_value(self, state_key, action, reward, next_state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0.0
        
        current_q = self.q_table[state_key][action]
        
        if next_state_key in self.q_table:
            max_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0
        else:
            max_next_q = 0
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
    
    def store_experience(self, state, action, reward, next_state, done):
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        self.experience_buffer.append(experience)
    
    def calculate_reward(self, old_fitness, new_fitness, problem_info):
        fitness_improvement = new_fitness - old_fitness
        
        if self.agent_type == "llm":
            base_reward = fitness_improvement * 10
            if fitness_improvement > 0:
                base_reward += 5
            else:
                base_reward -= 2
        else:
            base_reward = fitness_improvement * 8
            if fitness_improvement > 0:
                base_reward += 3
            else:
                base_reward -= 1
        
        if problem_info["difficulty"] == "expert":
            base_reward *= 1.5
        elif problem_info["difficulty"] == "hard":
            base_reward *= 1.2
        
        return base_reward
    
    def decay_epsilon(self):
        self.epsilon = max(0.05, self.epsilon * 0.995)
    
    def get_performance_stats(self):
        if not self.performance_history:
            return {"avg_reward": 0, "total_episodes": 0, "success_rate": 0}
        
        recent_rewards = self.performance_history[-50:]
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        success_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
        
        return {
            "avg_reward": avg_reward,
            "total_episodes": len(self.performance_history),
            "success_rate": success_rate,
            "epsilon": self.epsilon
        }
    
    def save_model(self, filename):
        model_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'performance_history': self.performance_history[-100:]
        }
        with open(filename, 'w') as f:
            json.dump(model_data, f)
    
    def load_model(self, filename):
        try:
            with open(filename, 'r') as f:
                model_data = json.load(f)
            self.q_table = model_data['q_table']
            self.epsilon = model_data['epsilon']
            self.performance_history = model_data['performance_history']
        except FileNotFoundError:
            pass
