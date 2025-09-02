# 💡📝📖 Flashcard Generator 

A Python application that generates study flashcards from any text input using Hugging Face's `google/gemma-3-270m` model.

## Features

- **Hugging Face**: Uses Google's Gemma 3 270M model for intelligent content extraction
- **Automatic Formatting**: Converts any text into exactly 3 Q/A flashcards
- **Fallback Mechanism**: Multiple parsing strategies ensure you always get usable output
- **Gated Model Support**: Handles Hugging Face authentication automatically

## How It Works

1. **Input**: Paste any text (articles, notes, paragraphs, etc.)
2. **Processing**: The AI model analyzes the content and attempts to create structured Q/A pairs
3. **Fallback System**: If the model doesn't follow strict formatting, the fallbacks will extract key information
4. **Output**: Always delivers exactly 3 flashcards in Q/A format

## Installation & Setup

### 1. Create Virtual Environment
```bash
python3 -m venv ~/.venvs/flashcards
source ~/.venvs/flashcards/bin/activate
```

### 2. Install Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Authenticate with Hugging Face
```bash
huggingface-cli login
```
- Create/login to your Hugging Face account
- Request access to `google/gemma-3-270m` model
- Follow the authentication prompts

### 4. Set Environment Variables (if using macOS)
```bash
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=
```

## Run the Program

```bash
python main.py
```

When prompted, paste your source text and press Enter. The program will generate 3 flashcards. Source text can be of any subject.

## Example

### Input
```
The Georgia Institute of Technology is a public research university and institute of technology in Atlanta, Georgia, United States. Established in 1885, it has the largest student enrollment of the University System of Georgia institutions and satellite campuses in Savannah, Georgia, and Metz, France.
```

### Output
```
1) Q: What is The Georgia Institute of Technology?
   A: The Georgia Institute of Technology is a public research university and institute of technology in Atlanta, Georgia, United States.

2) Q: Where are Georgia Tech's satellite campuses located?
   A: Satellite campuses are located in Savannah, Georgia, and Metz, France.

3) Q: What distinction does The Georgia Institute of Technology hold within the University System of Georgia?
   A: The Georgia Institute of Technology has the largest student enrollment.
```

## Technical Details

- **Model**: `google/gemma-3-270m` (270M parameters)
- **Framework**: Hugging Face Transformers with PyTorch backend

## Troubleshooting

### "Gated repo" Error
- Ensure you've requested access to the model on Hugging Face
- Run `huggingface-cli login` and authenticate
- Verify with `huggingface-cli whoami`

### Model Not Following Format
- This can happen due to small size of the 270M model
- The fallback system ensures you always get 3 flashcards
- Questions may be generic but answers are extracted from your text

## Future Developments
- Use a larger model that can create varied question outputs
- Create an intuitive frontend

## Requirements

- Python 3.9+
- Hugging Face account with model access
- Disk space for model cache
