import fitz
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading PDF file {pdf_path}: {e}")
        return None

class HFEmbedder:
    def __init__(self, model_name="intfloat/e5-small", chunk_size=256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.chunk_size = chunk_size

    def encode(self, text):
        words = text.split()
        chunks = [" ".join(words[i:i + self.chunk_size]) for i in range(0, len(words), self.chunk_size)]
        chunk_embeddings = []
        for chunk in chunks:
            inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1)  # mean pooling
                emb = F.normalize(emb, p=2, dim=1)
                chunk_embeddings.append(emb)

        final_embedding = torch.mean(torch.stack(chunk_embeddings), dim=0)
        return F.normalize(final_embedding, p=2, dim=1)




def process_pdfs(pdf_paths, embedder):
    embeddings = []
    metadata = []
    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        if text:
            embedding = embedder.encode(text)
            embeddings.append(embedding)
            metadata.append({"path": pdf_path, "text": text})
            print(f"Processed: {pdf_path}")
        else:
            print(f"Skipped: {pdf_path}")
    return embeddings, metadata

def search_documents(query, embeddings, metadata, embedder, top_k=1):
    query_embedding = embedder.encode(query)
    scores = [F.cosine_similarity(query_embedding, e).item() for e in embeddings]
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    results = []
    for idx in sorted_idx[:top_k]:
        results.append({"path": metadata[idx]["path"], "score": scores[idx], "preview": metadata[idx]["text"][:500]+"..."})
    return results




if __name__ == "__main__":
    print("=================PDF Semantic Search==============\n")

    pdf_files = [#add your pdf files here
                ]

    embedder = HFEmbedder(model_name="intfloat/e5-small")
    embeddings, metadata = process_pdfs(pdf_files, embedder)
    print("\nEmbeddings created for all PDFs.\n")

    while True:
        query = input("Enter your search query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        results = search_documents(query, embeddings, metadata, embedder, top_k=3)
        print("\nTop matching documents:")
        for i, res in enumerate(results, start=1):
            print(f"{i}. Path: {res['path']}")
            print(f"   Similarity: {res['score']*100:.2f}%")
            print(f"   Preview: {res['preview']}\n")
