from transformers import pipeline
import re

# Load summarization pipeline
summarizer = pipeline("summarization", model="t5-small")

def chunk_text(text, max_chunk_size=450):
    """
    Splits text into smaller chunks so T5-small can summarize safely.
    Splits at sentence boundaries.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def is_relevant(bullet, text, threshold=0.3):
    """
    Returns True if at least `threshold` fraction of words in the bullet appear in text.
    """
    bullet_words = set(bullet.lower().split())
    text_words = set(text.lower().split())
    overlap = bullet_words & text_words
    return len(overlap) / max(1, len(bullet_words)) >= threshold

def make_notes(text):
    """
    Summarizes any text into bullet points, handling long texts by chunking.
    Removes duplicate bullets and bullets unlikely to be relevant.
    """
    if len(text.split()) < 10:
        return f"- {text.capitalize()}"

    bullets = []
    for chunk in chunk_text(text):
        summary_list = summarizer(
            chunk,
            max_length=120,
            min_length=40,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        summary = summary_list[0]['summary_text'] if summary_list else chunk
        chunk_bullets = [b.strip() for b in summary.split('. ') if b.strip()]
        bullets.extend(chunk_bullets)

    # Remove duplicates
    bullets = list(dict.fromkeys(bullets))
    # Capitalize bullets
    bullets = [f"- {b.capitalize().rstrip('.')}" for b in bullets]
    # Filter bullets by word overlap with original text
    bullets = [b for b in bullets if is_relevant(b[2:], text)]  # remove "- " before checking

    return "\n".join(bullets)


if __name__ == "__main__":
    print("Welcome to AI@GT Note Taker!")
    print("Enter your text (type 'quit' to exit):\n")

    while True:
        user_input = input(">> ")
        if user_input.lower() == "quit":
            break
        notes_output = make_notes(user_input)
        print("\nYour Notes:\n")
        print(notes_output)
        print("\n" + "-"*40 + "\n")
