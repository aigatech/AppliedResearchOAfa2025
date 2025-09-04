# Distilling DialoGPT-Medium Knowledge into Lightweight DistilGPT-2 Architecture


# Project Overview

This project applied knowledge distillation using gradient descent optimization and a KL-Divergence Loss Function: . 
The student model (DistilGPT-2) learned from a larger teacher model (DialoGPT-medium) to exhibit similar language generation capabilities while being much more efficient.

Teacher Model: Microsoft's DialoGPT-medium (larger, pre-trained conversational model)
Student Model: DistilGPT-2 (smaller, faster model, not pre-trained)

# Features

Temperature Scaling: Softens probability distributions for better knowledge transfer

Attention Masking: Properly handles variable-length sequences

Gradient Descent Optimization: Uses modern techniques like gradient clipping and warmup scheduling

Loss Function: Combines distillation loss (KL divergence) with standard cross-entropy loss

Comparative Analysis: Shows before/after outputs to demonstrate learning

CPU Compatible: Runs efficiently on CPU (no GPU required)

#Features

`cd /submission/Eshaan_Patel`

`pip install -r requirements.txt`

`python knowledge_distillation.py`

