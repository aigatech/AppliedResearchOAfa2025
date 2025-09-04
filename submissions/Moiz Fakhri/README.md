# Simple Task Planner Agent

**Submission for AI@GT Applied Research Fall 2025**

## What it does

This project uses the `google/flan-t5-base` model from Hugging Face to act as a simple planning agent. It takes a high-level goal as input and breaks it down into actionable steps, demonstrating the core "planning and reasoning" capability needed for more complex AI agents.

The agent can handle various types of goals including:
- Household tasks (e.g., "Do the laundry")
- Hobby projects (e.g., "Plant a small herb garden in a pot")
- Academic tasks (e.g., "Prepare for a job interview")
- Technical tasks (e.g., "Restart computer")

## How to run it

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Jupyter Lab**
   ```bash
   jupyter lab
   ```

5. **Open the notebook**
   - Navigate to `simple_task_planner.ipynb`
   - Run all cells in order (Shift+Enter or click Run)

### First Run Notes
- The model will be automatically downloaded from Hugging Face on first run
- This may take a few minutes depending on your internet connection
- The model is approximately 990MB in size

### Usage Examples

The notebook includes three test cases:
1. **Household Chore**: "Do the laundry"
2. **Hobby Task**: "Plant a small herb garden in a pot"  
3. **Academic Task**: "Prepare for a job interview"

You can also test with your own goals by calling:
```python
generate_plan("Your custom goal here")
```

## Technical Details

- **Model**: `google/flan-t5-base` (990MB, CPU-friendly)
- **Framework**: Hugging Face Transformers
- **Architecture**: Text-to-text generation
- **Parameters**: Optimized for concise, contextual responses
- **Hardware**: Runs on CPU (no GPU required)

## Project Structure

```
submissions/Moiz Fakhri/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
└── simple_task_planner.ipynb   # Main implementation
```

## Dependencies

- `transformers` - Hugging Face transformers library
- `torch` - PyTorch for model inference
- `sentencepiece` - Tokenizer for the model
- `jupyterlab` - Jupyter notebook environment

## Troubleshooting

**Model download issues**: Ensure you have a stable internet connection for the initial model download.

**Memory issues**: The model requires approximately 2GB of RAM. Close other applications if needed.

**Slow performance**: The model runs on CPU. For faster inference, consider using a GPU-enabled environment.

## About the Implementation

This project demonstrates:
- Integration with Hugging Face transformers library
- Text-to-text generation for planning tasks
- Prompt engineering for better model outputs
- Parameter optimization for quality results
- Contextual understanding of different goal types

The implementation uses optimized parameters to reduce repetitive outputs and improve the relevance of generated plans.