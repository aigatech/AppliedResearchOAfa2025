from sentence_transformers import SentenceTransformer, util
st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def pick_best(headlines, goal="increase signups for a local service"):
    sims = [util.cos_sim(st.encode(h), st.encode(goal)).item() for h in headlines]
    return headlines[int(max(range(len(sims)), key=lambda i: sims[i]))]