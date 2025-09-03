# 💡📝📖 Flashcard Generator 📖📝💡

With the rise of short-form content, knowledge aquisition is more difficult than ever 😭. With this program 🤩, any chunk of text can be condensed into just 3 simple flashcards 🌩️ that capture the crux of the information 📓.

And...since doomscrolling is so visually engaging 👀, the program (by choice) will generate a matching doodle 🖌️ for each flashcard to help you retain the info even better 🤓!

## Features ✨

- **Multiple AI Models**: Choose between speed vs. quality with 3 different models
- **Automatic Formatting**: Converts any text into exactly 3 Q/A flashcards
- **Performance Benchmarking**: Compare model speeds and quality
- **CLI Interface**: Command-line arguments for advanced usage
- **Fallback Mechanism**: Multiple parsing strategies ensure you always get usable output
- **Gated Model Support**: Handles Hugging Face authentication automatically
- **Flashcard Doodle Mode**: Generate a doodle image per flashcard using Stable Diffusion

## How It Works 💡

1. **Input**: Paste any text (articles, notes, paragraphs, etc.)
2. **Processing**: The AI model analyzes the content and attempts to create structured Q/A pairs
3. **Fallback System**: If the model doesn't follow strict formatting, the fallbacks will extract key information
4. **Output**: Always delivers exactly 3 flashcards in Q/A format

## Installation & Setup 🔧

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
- Request access to `google/gemma-3-270m` model (required for "balanced" mode)
- Follow the authentication prompts

### 4. Set Environment Variables (if using macOS)
```bash
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=
```

## Quick Start 📝

Try it immediately with the fast model (no authentication required):
```bash
python main.py --model fast --text "insert text here."
```

## Usage 📝

### Basic Usage
```bash
python main.py
```

### Advanced Usage
```bash
# Use fast model for quick generation
python main.py --model fast

# Use quality model for better results
python main.py --model quality

# Process text directly from command line
python main.py --text "Your text here" --model balanced

# List available models
python main.py --list-models

# Run benchmark on all models
python main.py --benchmark
```

### Available Models
- **fast**: DistilGPT-2 - Ultra-fast generation, basic quality
- **balanced**: Gemma 3 270M - Good balance of speed and quality (default)
- **quality**: DialoGPT Medium - Higher quality, slower generation

When prompted, paste your source text and press Enter. The program will generate 3 flashcards. Source text can be of any subject.

## Flashcard Doodle Mode 🎨

Generate a doodle image per flashcard using Stable Diffusion (`runwayml/stable-diffusion-v1-5`).

```bash
python main.py --model fast --text "your text here" --doodle
```

### Behavior
- For each flashcard's answer, an image is generated with Stable Diffusion using the answer text as the prompt
- Images are saved as `flashcard_1.png`, `flashcard_2.png`, `flashcard_3.png`
- The console output includes the saved filename per flashcard
- Progress is printed: `🎨 Generating doodle for flashcard X...`

### Requirements
- Uses `diffusers` with `runwayml/stable-diffusion-v1-5`
- If the model is gated, request access and ensure you're logged in with `huggingface-cli login`

## Example

### Input
```
The Georgia Institute of Technology is a public research university and institute of technology in Atlanta, Georgia, United States. Established in 1885, it has the largest student enrollment of the University System of Georgia institutions and satellite campuses in Savannah, Georgia, and Metz, France.
```

### Output
```
1) Q: What is The Georgia Institute of Technology?
   A: The Georgia Institute of Technology is a public research university and institute of technology in Atlanta, Georgia, United States.
   🖼️ [Doodle saved at flashcard_1.png]

2) Q: Where are Georgia Tech's satellite campuses located?
   A: Satellite campuses are located in Savannah, Georgia, and Metz, France.
   🖼️ [Doodle saved at flashcard_2.png]

3) Q: What distinction does The Georgia Institute of Technology hold within the University System of Georgia?
   A: The Georgia Institute of Technology has the largest student enrollment.
   🖼️ [Doodle saved at flashcard_3.png]
```

## Technical Details 📝

- **Models**: 
  - DistilGPT-2 (82M parameters) - Fast
  - Gemma 3 270M (270M parameters) - Balanced  
  - DialoGPT Medium (345M parameters) - Quality
  - Stable Diffusion v1-5 for doodles (`runwayml/stable-diffusion-v1-5`)
- **Framework**: Hugging Face Transformers and Diffusers with PyTorch backend

## Troubleshooting 🔧

### "Gated repo" Error
- Only affects the "balanced" model (Gemma 3 270M) and possibly Stable Diffusion
- Ensure you've requested access on Hugging Face and run `huggingface-cli login`
- Verify with `huggingface-cli whoami`
- Alternative: Use `--model fast` or `--model quality` (no auth required)

### Model Not Following Format
- Smaller models (fast/balanced) may not follow strict Q/A formatting
- The fallback system ensures you always get 3 flashcards
- Try `--model quality` for better formatting adherence

## Future Developments 💡
- Add more model options (Llama, Mistral, etc.)
- Implement question type selection (multiple choice, fill-in-blank)
- Add web interface for easier usage
- Support for PDF and URL inputs
