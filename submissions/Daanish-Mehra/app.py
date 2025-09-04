#!/usr/bin/env python3
"""
Evolution Arena - Hugging Face Space
A genetic algorithm game powered by multiple HF models
"""

from gradio_space import create_interface

if __name__ == "__main__":
    # This is the entry point for Hugging Face Spaces
    demo = create_interface()
    demo.launch()
