# Evolution Arena - Hugging Face Powered

A genetic algorithm game that leverages **multiple Hugging Face models**, datasets, and Spaces for LLM-guided optimization.

## 🤖 Hugging Face Integration

### Multiple Models
- **google/flan-t5-base** - For bitstring problems
- **microsoft/DialoGPT-small** - For permutation problems  
- **facebook/blenderbot-400M-distill** - For expression problems
- **google/flan-t5-small** - Default fallback model

### HF Datasets
- Converts 100+ optimization problems to HF Dataset format
- Includes metadata, difficulty scores, and tags
- Shareable and discoverable on Hugging Face Hub

### HF Spaces
- Full Gradio web interface
- Interactive problem selection and visualization
- Real-time evolution progress tracking

## 🚀 Quick Start

```bash
pip install -r reqs.txt
python3 main.py
```

### Available Modes

1. **Fast Learning Arena** - Basic optimization
2. **Real Evolution Arena** - Full LLM integration  
3. **Custom Challenge** - Interactive problem selection
4. **Multi-Model Demo** - Test different HF models
5. **Gradio Space** - Web interface

## 📁 Project Structure

```
lib/
├── ai/
│   ├── llm.py              # Original single model
│   └── multi_model_llm.py  # Multiple HF models
├── data/
│   └── hf_dataset.py       # HF Dataset management
├── algo/
│   └── ga.py               # Genetic Algorithm
├── rl/
│   ├── rl_agent.py         # RL agent
│   └── rl_loop.py          # RL utilities
└── problems/
    └── probs.json          # 100 optimization problems
```

## 🎮 Usage Examples

### Multi-Model Testing
```bash
python3 main.py
# Choose option 4
```

### Gradio Space
```bash
python3 main.py  
# Choose option 5
# Or directly: python3 gradio_space.py
```

### Dataset Stats
```python
from lib.data.hf_dataset import HFDatasetManager
manager = HFDatasetManager()
print(manager.get_stats())
```

## 🔧 Dependencies

Core Hugging Face:
- transformers
- datasets  
- huggingface_hub
- gradio

Scientific:
- torch, numpy, scipy
- matplotlib, pandas
- deap, networkx

## 🎯 Features

- 100+ optimization problems
- Multiple specialized HF models
- HF Dataset integration
- Interactive web interface
- Real-time visualization
- RL-guided algorithm selection

## �� Problems

Bitstring, permutation, expression, constraint, and combinatorial optimization problems across easy to expert difficulty levels.

That's it.
