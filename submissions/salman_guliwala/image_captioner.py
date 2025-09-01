import os
import faiss
import numpy as np
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
import gradio as gr

# Paths
CORPUS_PATH = "corpus.txt"
FAISS_INDEX_PATH = "faiss.index"

# Load BLIP model & processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load sentiment analysis pipeline
sentiment_analyzer = pipeline(
    "sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Load SentenceTransformer embedder for retrieval
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load text generation model and tokenizer (T5-small)
tokenizer = AutoTokenizer.from_pretrained("t5-small")
text_generator = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Load or build corpus and FAISS index
def load_corpus_and_index(corpus_path=CORPUS_PATH, index_path=FAISS_INDEX_PATH):
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"{corpus_path} not found. Please add a corpus.txt file.")
    with open(corpus_path, encoding="utf-8") as f:
        corpus_lines = [line.strip() for line in f if line.strip()]
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        embeddings = embedding_model.encode(corpus_lines, convert_to_tensor=False, show_progress_bar=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype("float32"))
        faiss.write_index(index, index_path)
    return corpus_lines, index

try:
    corpus, faiss_index = load_corpus_and_index()
except Exception as e:
    print(f"⚠️ {e}")
    corpus = []
    faiss_index = None

def retrieve_facts(query, top_k=3):
    if faiss_index is None or not corpus:
        return []
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    distances, indices = faiss_index.search(np.array(query_embedding).astype("float32"), top_k)
    return [corpus[idx] for idx in indices[0] if idx < len(corpus)]

def rewrite_caption(base_caption, retrieved_facts):
    if retrieved_facts:
        input_text = f"rewrite: {"\n".join(f"{i}. {s}" for i, s in enumerate(retrieved_facts, start=1))}"
    else:
        input_text = base_caption
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = text_generator.generate(**inputs, max_length=100, num_beams=4)
    rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return rewritten

def analyze_image(image):
    try:
        if image is None:
            return "❌ Please upload an image first!"
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(pixel_values=inputs["pixel_values"], max_length=50, num_beams=4)
        caption = processor.decode(output[0], skip_special_tokens=True)

        facts = retrieve_facts(caption)
        enriched_caption = rewrite_caption(caption, facts)

        sentiment_result = sentiment_analyzer(enriched_caption)[0]
        label = sentiment_result["label"]
        score = sentiment_result["score"]

        if label == "LABEL_2":
            mood = "Positive 😊"
            interp = "This image has a positive/happy vibe!"
        elif label == "LABEL_0":
            mood = "Negative 😔"
            interp = "This image seems to have a negative/sad mood."
        else:
            mood = "Neutral 😐"
            interp = "This image has a neutral mood."

        result = f"## 📸 Image Analysis Results\n\n"
        result += f"**📝 Caption:**\n{enriched_caption}\n\n"
        result += f"**🎭 Sentiment:** {mood} ({score:.1%} confidence)\n\n"
        result += f"**💭 Interpretation:** {interp}"
        return result

    except Exception as e:
        return f"❌ Error processing image: {e}\n\nPlease try uploading a different image."

demo = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(type="pil", label="📤 Upload Your Image"),
    outputs=gr.Markdown(label="🔍 Analysis Results"),
    title="🖼️ BLIP Captioner with RAG and Sentiment Analysis",
    description="""
    ### How it works:
    1. Upload an image.
    2. BLIP generates an initial caption.
    3. Relevant facts retrieved from corpus using FAISS semantic search.
    4. Caption rewritten with retrieved facts via T5-small.
    5. Sentiment analysis on enriched caption.
    """,
    theme=gr.themes.Soft(),
    flagging_mode=None,
)

if __name__ == "__main__":
    print("🚀 Starting BLIP Captioner with RAG...")
    demo.launch(share=True, server_name="0.0.0.0")
