# Visual Language Learning Tool

A multimodal language learning application that combines translation, text-to-speech, and AI-generated images to create an immersive learning experience.

## What it does

This tool helps language learners by providing three key features in one interface:

- **Translation**: Converts English text to Spanish, French, German, or Hindi
- **Audio Pronunciation**: Generates native-sounding speech in the target language
- **Visual Context**: Creates AI-generated images to reinforce vocabulary learning

## Purpose

Unlike most translation apps, this tool creates a learning experience with visual + auditory learning:

- **Beyond human capability**: Instantly generates contextual images for any concept
- **Combines multiple AI pipelines**: Translation → Speech → Vision working together
- **Educational focus**: Designed specifically for language learning, not just translation
- **Visual reinforcement**: Images help cement vocabulary in long-term memory
- **Native language support**: Includes authentic Hindi pronunciation and other target languages

## How it works

The application uses three AI pipelines working together, with each pipeline selecting from 4 different models based on the target language:

### Translation Pipeline (`utils/translator.py`)

- **Models**: 4 Helsinki-NLP Opus-MT models (one for each target language)
  - `opus-mt-en-fr` (English → French)
  - `opus-mt-en-de` (English → German)
  - `opus-mt-en-es` (English → Spanish)
  - `opus-mt-en-hi` (English → Hindi)
- **Purpose**: High-quality neural machine translation from English to target languages
- **Components**: MarianMTModel + AutoTokenizer for each language pair

### Text-to-Speech Pipeline (`utils/speak.py`)

- **Models**: 4 Facebook MMS-TTS models (one for each target language)
  - `facebook/mms-tts-fra` (French pronunciation)
  - `facebook/mms-tts-deu` (German pronunciation)
  - `facebook/mms-tts-spa` (Spanish pronunciation)
  - `facebook/mms-tts-hin` (Hindi pronunciation - native support)
- **Purpose**: Generate natural-sounding pronunciation in target languages
- **Components**: VitsModel + VitsTokenizer for realistic speech synthesis

### Image Generation Pipeline (`utils/drawer.py`)

- **Model**: FLUX.1-dev via Hugging Face Inference API
- **Purpose**: Create visual representations to reinforce vocabulary learning
- **Components**: Together AI provider for high-quality image generation

## Supported Languages

- English → Spanish, French, German, or Hindi (Translation + Audio + Images)

## How to run it

### Prerequisites

- Python 3.8 or higher
- Internet connection for model downloads
- Hugging Face account and token

### Installation

1. Clone and navigate to the project directory

```bash
cd your-project-folder
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Set up Hugging Face token
   - Get a token from [Hugging Face Settings](https://huggingface.co/settings/tokens)
   - Add your token to `main.py` by replacing the existing token on line 12:
   ```python
   os.environ["HF_TOKEN"] = "your_token_here"
   ```

### Running the Application

1. Start the Gradio app

```bash
python main.py
```

2. Open your browser and navigate to the URL shown in terminal (usually `http://127.0.0.1:7860`)

3. Use the interface:
   - Enter English text in the input box
   - Select your target language from the dropdown
   - Click "Translate" to get the translation
   - Click "Audio + Visual Learning" to get pronunciation and visual representation!

## Example Usage

**Input**: "The cat is sleeping on the sofa"  
**Select**: English → Spanish  
**Output**:

- Translation: "El gato está durmiendo en el sofá"
- Audio: Plays Spanish pronunciation
- Image: AI-generated picture of a cat sleeping on a sofa
