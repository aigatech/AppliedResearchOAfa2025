import wikipediaapi
from sentence_transformers import SentenceTransformer, util
import heapq
import streamlit as st

wiki = wikipediaapi.Wikipedia(user_agent='PJainWikiRaceBot', language='en')
model = SentenceTransformer('all-MiniLM-L6-v2')
MAX_LINKS = 10

def page_exists(title):
    return wiki.page(title).exists()

def get_links(title):
    p = wiki.page(title)
    return list(p.links.keys())

def rank_links(links, target_emb):
    embs = model.encode(links, convert_to_tensor=True)
    scores = util.cos_sim(target_emb, embs)[0]
    ranked = sorted(zip(links, list(scores)), key=lambda x: x[1], reverse=True)
    return ranked

def find_path(start, end, output):
    if not page_exists(start):
        output(f'Start page missing: {start}')
        return None
    if not page_exists(end):
        output(f'End page missing: {end}')
        return None

    target_emb = model.encode([end], convert_to_tensor=True)
    visited = set()
    queue = []
    heapq.heappush(queue, (0, [start]))

    while queue:
        score, path = heapq.heappop(queue)
        current = path[-1]

        output(f'Checking path: {" -> ".join(path)} (similarity: {score:.4f})')

        if current in visited:
            continue
        visited.add(current)

        if current == end:
            return path

        links = get_links(current)
        ranked = rank_links(links, target_emb)[:MAX_LINKS]

        for l, s in ranked:
            if l not in visited:
                heapq.heappush(queue, (s, path + [l]))

    return None


st.title('Wikipedia Path Finder')

col1, col2 = st.columns(2)
with col1:
    start_page = st.text_input('Start page', value='Taylor Swift')
with col2:
    end_page = st.text_input('End page', value='Quantum mechanics')

max_links = st.slider('Number of top links to consider per page:', 5, 20, 10)

MAX_LINKS = max_links

if st.button('Find Path'):
    log_area = st.empty()
    logs = []

    def log_callback(msg):
        if 'missing' in msg:
            st.error(msg)
        if 'Checking path' in msg:
            logs.clear()
        logs.append(msg)
        log_area.text('\n'.join(logs))

    path = find_path(start_page, end_page, output=log_callback)
    if path:
        st.success('Path found: ' + ' -> '.join(path))
    else:
        st.error('No path found.')
