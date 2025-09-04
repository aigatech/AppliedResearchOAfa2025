# Distilling DialoGPT-Medium Knowledge into Lightweight DistilGPT-2 Architecture


# Project Overview

This project applied knowledge distillation using gradient descent optimization and a KL-Divergence Loss Function.

The student model (DistilGPT-2) learned from a larger teacher model (DialoGPT-medium) to exhibit similar language generation capabilities while being much more efficient.

Teacher Model: Microsoft's DialoGPT-medium (larger, pre-trained conversational model) **350MB**

Student Model: DistilGPT-2 (smaller, faster model, not pre-trained) **80MB**

# Features

- Temperature Scaling: Softens probability distributions for better knowledge transfer

- Attention Masking: Properly handles variable-length sequences

- Gradient Descent Optimization: custom training loop with AdamW optimizer, learning rate scheduling

- Loss Function: Combines distillation loss (KL divergence) with standard cross-entropy loss

- Comparative Analysis: Shows before/after outputs to demonstrate learning

- CPU Compatible: Runs efficiently on CPU (no GPU required)

# How to Run 

**1.** `cd /submission/Eshaan_Patel`

**2.**`pip install -r requirements.txt`

**3.**`python knowledge_distillation.py`


# Expected Output

- Training progress with loss metrics (Total Loss, KL Divergence, Cross-Entropy)
- Comparative text generation showing teacher vs student outputs

# Example Results

The student model learns to generate coherent text similar to the teacher:
**Prompt:** "The future of artificial intelligence is"

**Teacher:** "The future of artificial intelligence is bright, with applications in healthcare, education, and sustainable technology..."
**Student:** "The future of artificial intelligence is promising, offering solutions for complex problems and improving human lives..."