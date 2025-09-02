import argparse
import os
import re
import sys
import time
from typing import List, Tuple, Dict, Any

# Mitigate macOS mutex crash in tokenizers and torch by disabling threads
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Force CPU and avoid GPU/MPS acceleration which can be unstable on some macOS setups
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:
    import torch  # type: ignore
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
except Exception:
    torch = None  # type: ignore

from transformers import pipeline


# Model configurations for different speed/quality tradeoffs
MODEL_CONFIGS = {
    "fast": {
        "model": "distilgpt2",
        "name": "DistilGPT-2 (Fast)",
        "description": "Ultra-fast generation, basic quality",
        "max_tokens": 80,
        "temperature": 0.7,
        "gated": False
    },
    "balanced": {
        "model": "google/gemma-3-270m",
        "name": "Gemma 3 270M (Balanced)",
        "description": "Good balance of speed and quality",
        "max_tokens": 120,
        "temperature": 0.5,
        "gated": True
    },
    "quality": {
        "model": "microsoft/DialoGPT-medium",
        "name": "DialoGPT Medium (Quality)",
        "description": "Higher quality, slower generation",
        "max_tokens": 150,
        "temperature": 0.3,
        "gated": False
    }
}


STRICT_FORMAT_GUIDE = (
    "Format strictly as three numbered items like this and nothing else:"\
    "\n\n"
    "1) Q: <question>\n   A: <answer>\n"
    "2) Q: <question>\n   A: <answer>\n"
    "3) Q: <question>\n   A: <answer>\n\n"
    "Rules: Exactly 3 items. No extra bullets. No explanations."
)


def build_prompt(source_text: str) -> str:
    return (
        "You are a helpful assistant that creates concise study flashcards.\n"
        "Read the provided text and generate exactly 3 flashcards.\n"
        "Each flashcard must be a short question and a short, factual answer grounded only in the provided text.\n"
        + STRICT_FORMAT_GUIDE + "\n\n"
        f"Text:\n{source_text}\n"
    )


def build_stricter_prompt(source_text: str) -> str:
    return (
        "STRICT MODE: Output must match the exact format. Generate 3 flashcards only.\n"
        + STRICT_FORMAT_GUIDE + "\n\n"
        f"Text:\n{source_text}\n"
    )


def parse_flashcards(generated_text: str) -> List[Tuple[str, str]]:
    # Extract pairs of Q:/A: lines, in order
    cards: List[Tuple[str, str]] = []
    lines = [line.strip() for line in generated_text.splitlines() if line.strip()]
    current_q = None
    for line in lines:
        if line.startswith("Q:"):
            current_q = line[2:].strip(" \t:-")
        elif line.startswith("A:") and current_q is not None:
            a = line[2:].strip(" \t:-")
            if a:
                cards.append((current_q, a))
            current_q = None
        if len(cards) == 3:
            break
    return cards


def extract_numbered_items_as_cards(text: str) -> List[Tuple[str, str]]:
    # Heuristic: take first 3 numbered items and make question/answer pairs
    items = re.findall(r"(?:^|\n)\s*(?:\d+\)|-\s+|•\s+)(.+?)\s*(?=(?:\n\s*(?:\d+\)|-\s+|•\s+)|$))", text, flags=re.S)
    cards: List[Tuple[str, str]] = []
    for raw in items[:3]:
        sentence = " ".join(raw.strip().split())
        if not sentence:
            continue
        q = "What is a key point from the text?"
        a = sentence
        cards.append((q, a))
    return cards


def fallback_from_source_text(source_text: str) -> List[Tuple[str, str]]:
    # Build 3 concise Q/A from the user's text directly
    # Split into sentences
    chunks = re.split(r"(?<=[.!?])\s+|\n+", source_text)
    sentences = [" ".join(s.strip().split()) for s in chunks if s and len(s.strip()) > 0]
    # Prefer non-trivial sentences
    selected = [s for s in sentences if len(s) > 20][:3]
    if len(selected) < 3:
        selected.extend(sentences[: 3 - len(selected)])
    selected = selected[:3] if selected else [source_text.strip()][:1]

    # Generate varied questions based on content
    question_templates = [
        "What is mentioned about this topic?",
        "What are the key details?",
        "What information is provided?",
        "What does the text say about this?",
        "What are the main points?",
        "What is described here?"
    ]

    cards: List[Tuple[str, str]] = []
    for i, s in enumerate(selected):
        if not s:
            continue
        # Use different question templates
        q = question_templates[i % len(question_templates)]
        a = s
        cards.append((q, a))
        if len(cards) == 3:
            break
    
    # Ensure exactly 3
    while len(cards) < 3:
        q = question_templates[len(cards) % len(question_templates)]
        cards.append((q, "<no additional content>"))
    return cards


def print_gated_repo_help() -> None:
    print(
        "\nThis model is gated on Hugging Face and requires access/authentication.\n"
        "To resolve:\n"
        "1) Create/login to a Hugging Face account.\n"
        "2) Request access on the model page: google/gemma-3-270m.\n"
        "3) Authenticate locally either by:\n"
        "   - Running: huggingface-cli login\n"
        "   - Or set an env var: export HF_TOKEN=your_token_here\n"
        "4) Re-run this script.\n"
    )


def generate_with_settings(generator, prompt: str, config: Dict[str, Any]) -> str:
    outputs = generator(
        prompt,
        max_new_tokens=config["max_tokens"],
        temperature=config["temperature"],
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1,
        eos_token_id=None,
        repetition_penalty=1.08,
    )
    return outputs[0]["generated_text"][len(prompt):].strip() if outputs else ""


def generate_flashcards(text: str, model_mode: str = "balanced") -> Tuple[List[Tuple[str, str]], float]:
    """Generate flashcards using specified model mode. Returns (cards, generation_time)."""
    config = MODEL_CONFIGS[model_mode]
    user_token = os.environ.get("HF_TOKEN") if config["gated"] else None

    print(f"Using {config['name']}: {config['description']}")
    
    start_time = time.time()
    
    try:
        generator = pipeline(
            "text-generation",
            model=config["model"],
            tokenizer=config["model"],
            token=user_token,
            framework="pt",
            device=-1,  # force CPU
        )
        print("Device set to use cpu")
    except Exception as e:  # Handle gated/unauthorized errors gracefully
        message = str(e).lower()
        if "gated" in message or "401" in message or "unauthorized" in message or "forbidden" in message:
            print(f"Error: Access to model '{config['model']}' is gated or unauthorized.")
            print_gated_repo_help()
            return [("Authorization required", "Please grant access and authenticate (see instructions above).")], 0.0
        raise

    # First attempt
    prompt = build_prompt(text)
    try:
        generated = generate_with_settings(generator, prompt, config)
    except Exception as e:
        print("A low-level runtime error occurred while generating text.")
        print("Tips:")
        print("  - export TOKENIZERS_PARALLELISM=false")
        print("  - export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1")
        print("  - Try a clean virtualenv and reinstall dependencies (transformers, torch)")
        raise

    cards = parse_flashcards(generated)

    # Retry once with stricter prompt and lower temperature if parsing failed
    if not cards:
        strict_prompt = build_stricter_prompt(text)
        strict_config = config.copy()
        strict_config["temperature"] = 0.1
        generated = generate_with_settings(generator, strict_prompt, strict_config)
        cards = parse_flashcards(generated)

    # Heuristic fallback if still nothing
    if not cards:
        cards = extract_numbered_items_as_cards(generated)

    # Final fallback: derive from source text to guarantee 3 items
    if not cards:
        cards = fallback_from_source_text(text)

    generation_time = time.time() - start_time
    return cards, generation_time


def print_model_options() -> None:
    """Print available model options."""
    print("Available models:")
    for mode, config in MODEL_CONFIGS.items():
        print(f"  {mode:8} - {config['name']}: {config['description']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate flashcards from text using AI models")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="balanced",
                       help="Model mode: fast (speed), balanced (default), quality")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark on all models")
    parser.add_argument("--text", type=str, help="Text to process (if not provided, will prompt)")
    
    args = parser.parse_args()
    
    if args.list_models:
        print_model_options()
        return
    
    print("💡📝📖 Flashcard Generator 📖📝💡")
    print(f"Selected model: {MODEL_CONFIGS[args.model]['name']}")
    print()
    
    if args.benchmark:
        run_benchmark()
        return
    
    # Get source text
    if args.text:
        source_text = args.text
    else:
        print("Enter or paste the source text, then press Enter.\n")
        try:
            source_text = input("> ")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return

    if not source_text.strip():
        print("No text provided. Exiting.")
        return

    print(f"\nGenerating 3 flashcards using {MODEL_CONFIGS[args.model]['name']}...\n")
    cards, generation_time = generate_flashcards(source_text, args.model)

    for idx, (q, a) in enumerate(cards, start=1):
        print(f"{idx}) Q: {q}")
        print(f"   A: {a}\n")
    
    print(f"⏱️  Generation time: {generation_time:.2f} seconds")


def run_benchmark() -> None:
    """Run benchmark on all models with sample text."""
    sample_text = "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms and has a large standard library."
    
    print("🚀 Running benchmark on all models...\n")
    
    results = []
    for mode, config in MODEL_CONFIGS.items():
        print(f"Testing {config['name']}...")
        try:
            cards, generation_time = generate_flashcards(sample_text, mode)
            results.append((mode, config['name'], generation_time, len(cards)))
            print(f"✅ Completed in {generation_time:.2f}s\n")
        except Exception as e:
            print(f"❌ Failed: {e}\n")
            results.append((mode, config['name'], float('inf'), 0))
    
    # Print benchmark results
    print("📊 Benchmark Results:")
    print("-" * 60)
    print(f"{'Model':<20} {'Time (s)':<10} {'Cards':<8} {'Status'}")
    print("-" * 60)
    
    for mode, name, time_taken, card_count in results:
        status = "✅ Success" if time_taken != float('inf') else "❌ Failed"
        print(f"{name:<20} {time_taken:<10.2f} {card_count:<8} {status}")
    
    # Find fastest and best quality
    successful_results = [(mode, name, time_taken) for mode, name, time_taken, _ in results if time_taken != float('inf')]
    if successful_results:
        fastest = min(successful_results, key=lambda x: x[2])
        print(f"\n🏆 Fastest: {fastest[1]} ({fastest[2]:.2f}s)")


if __name__ == "__main__":
    main()
