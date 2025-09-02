import os
import re
import sys
from typing import List, Tuple

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


def generate_with_settings(generator, prompt: str, temperature: float) -> str:
    outputs = generator(
        prompt,
        max_new_tokens=120,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1,
        eos_token_id=None,
        repetition_penalty=1.08,
    )
    return outputs[0]["generated_text"][len(prompt):].strip() if outputs else ""


def generate_flashcards(text: str) -> List[Tuple[str, str]]:
    user_token = os.environ.get("HF_TOKEN")

    try:
        generator = pipeline(
            "text-generation",
            model="google/gemma-3-270m",
            tokenizer="google/gemma-3-270m",
            token=user_token,
            framework="pt",
            device=-1,  # force CPU
        )
        print("Device set to use cpu")
    except Exception as e:  # Handle gated/unauthorized errors gracefully
        message = str(e).lower()
        if "gated" in message or "401" in message or "unauthorized" in message or "forbidden" in message:
            print("Error: Access to model 'google/gemma-3-270m' is gated or unauthorized.")
            print_gated_repo_help()
            return [("Authorization required", "Please grant access and authenticate (see instructions above).")]
        raise

    # First attempt
    prompt = build_prompt(text)
    try:
        generated = generate_with_settings(generator, prompt, temperature=0.3)
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
        generated = generate_with_settings(generator, strict_prompt, temperature=0.1)
        cards = parse_flashcards(generated)

    # Heuristic fallback if still nothing
    if not cards:
        cards = extract_numbered_items_as_cards(generated)

    # Final fallback: derive from source text to guarantee 3 items
    if not cards:
        cards = fallback_from_source_text(text)

    return cards


def main() -> None:
    print("Flashcard Generator (Gemma 3 270M)")
    print("Enter or paste the source text, then press Enter.\n")
    try:
        source_text = input("> ")
    except KeyboardInterrupt:
        print("\nCancelled.")
        return

    if not source_text.strip():
        print("No text provided. Exiting.")
        return

    print("\nGenerating 3 flashcards...\n")
    cards = generate_flashcards(source_text)

    for idx, (q, a) in enumerate(cards, start=1):
        print(f"{idx}) Q: {q}")
        print(f"   A: {a}\n")


if __name__ == "__main__":
    main()
