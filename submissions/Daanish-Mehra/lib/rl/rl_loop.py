import numpy as np

def get_reward(old_fit, new_fit):
    old_sum = sum(old_fit)
    new_sum = sum(new_fit)
    return new_sum - old_sum

def update_rl_state(rl_state, reward, problem_id):
    rl_state['total_reward'] += reward
    rl_state['episodes'] += 1
    rl_state['avg_reward'] = rl_state['total_reward'] / rl_state['episodes']
    
    if reward > 0:
        rl_state['llm_effectiveness'] = min(1.0, rl_state['llm_effectiveness'] + 0.01)
    else:
        rl_state['llm_effectiveness'] = max(0.1, rl_state['llm_effectiveness'] - 0.01)
    
    if problem_id not in rl_state['problem_rewards']:
        rl_state['problem_rewards'][problem_id] = []
    rl_state['problem_rewards'][problem_id].append(reward)
    
    return rl_state

def init_rl_state():
    return {
        'total_reward': 0,
        'episodes': 0,
        'avg_reward': 0,
        'llm_effectiveness': 0.1,
        'problem_rewards': {}
    }
