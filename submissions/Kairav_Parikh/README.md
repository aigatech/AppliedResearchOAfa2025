# SpeedQuiz

## What it does
Generates multiple-choice quizzes from PDFs using AI. Extracts text, creates questions, quizzes you in terminal, and saves results.

## Setup
```bash
pip3 install transformers torch PyPDF2 sentencepiece
```

## How to Run
1. Put your PDF in the same folder as `speedquiz.py`
2. Run: `python3 speedquiz.py`
3. Enter PDF path when prompted

## Example
```
Enter PDF path: sample.pdf

Question 1: What is photosynthesis?
A) Process plants use to make food
B) Cell division
C) None of the above  
D) Cannot be determined

Your answer: A
✓ Correct!

Final Score: 2/3 (66.7%)
Results saved!
```

## Features
- PDF text extraction
- AI question generation
- Multiple choice format
- Automatic grading
- Quiz history tracking