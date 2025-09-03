
# Four-Chord Generator

***Four-Chord Generator** is a tool to help songwriters generate random chords for their songs.*

## Features
- **Random Sequence Generator** - *Generates a random four-chord sequence for your `INTRO`, `VERSE`, `CHORUS`, `SOLO`, or `BRIDGE`.*
- **Random Song Generator** - *Generates a song from scratch by allowing you to pick one of four random `INTRO` sequences and generating sequences for the rest of the sections.*

## How to run
Pretty straightforward - run `main.py` directly in the terminal, as I was unfortunately unable to create a GUI given time limitations.

## Process
*Most of this project's time and work was spent creating the model that powers it, [ChordGPT](https://huggingface.co/xiejoshua/chordgpt), which is essentially GPT-2 fine-tuned on a large dataset of chords. However, due to technical and time limitations, I was only able to use 25% of the dataset of chords ([Chordonomicon](https://huggingface.co/datasets/ailsntua/Chordonomicon)) and I was only able to train it for 3 epochs, meaning the loss is pretty high and the model's output suffers from repetition.*
