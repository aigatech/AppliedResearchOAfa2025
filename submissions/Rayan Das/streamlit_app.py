"""
Semantic FAQ Search Engine - Streamlit Web App
"""

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


def parse_faq_text(faq_text):
    # Parse Q: and A: format into question-answer pairs
    if not faq_text.strip():
        return []
    
    pairs = re.split(r'\n\s*\n', faq_text.strip())
    faq_data = []
    
    for pair in pairs:
        lines = pair.strip().split('\n')
        question = ""
        answer = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q:') or line.startswith('q:'):
                question = line[2:].strip()
            elif line.startswith('A:') or line.startswith('a:'):
                answer = line[2:].strip()
            elif question and not answer:
                question += " " + line
            elif answer:
                answer += " " + line
        
        if question and answer:
            faq_data.append({'question': question, 'answer': answer})
    
    return faq_data


@st.cache_resource
def load_model():
    # Load sentence transformer model (cached for performance)
    return SentenceTransformer('all-MiniLM-L6-v2')


def search_faq(model, question_embeddings, faq_data, query, top_k=3):
    # Find most similar FAQ answers using cosine similarity
    if not faq_data or len(question_embeddings) == 0:
        return []
    
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, question_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:  # Minimum similarity threshold
            results.append({
                'question': faq_data[idx]['question'],
                'answer': faq_data[idx]['answer'],
                'similarity': float(similarities[idx])
            })
    
    return results


def get_sample_faq():
    return """Q: What is your return policy?
A: We offer a 30-day return policy for all items. Items must be in original condition with tags attached. Returns are free within the US.

Q: How long does shipping take?
A: Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days. International shipping takes 7-14 business days.

Q: Do you offer international shipping?
A: Yes, we ship to over 50 countries worldwide. International shipping costs vary by destination and are calculated at checkout.

Q: What payment methods do you accept?
A: We accept all major credit cards (Visa, MasterCard, American Express), PayPal, Apple Pay, and Google Pay.

Q: How can I track my order?
A: Once your order ships, you'll receive a tracking number via email. You can also track your order in your account dashboard.

Q: What is your customer service hours?
A: Our customer service team is available Monday through Friday, 9 AM to 6 PM EST. You can reach us via email, phone, or live chat.

Q: Do you have a mobile app?
A: Yes, our mobile app is available for both iOS and Android devices. Download it from the App Store or Google Play Store.

Q: How do I change my password?
A: Go to your account settings and click on 'Change Password'. You'll need to enter your current password and create a new one.

Q: What if my item arrives damaged?
A: If your item arrives damaged, please contact us within 48 hours with photos. We'll arrange for a replacement or full refund.

Q: Do you offer student discounts?
A: Yes, we offer a 10% student discount. Verify your student status through our partner verification service to receive the discount."""


def main():
    st.set_page_config(page_title="FAQ Search", layout="wide")
    
    # Load AI model
    with st.spinner("Loading model..."):
        model = load_model()
    
    # Two-column layout: FAQ input (left) and question search (right)
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("FAQ Data")
        
        if st.button("Load Sample FAQ"):
            st.session_state.faq_text = get_sample_faq()
        
        faq_text = st.text_area(
            "Enter your FAQ. Every question must start with Q: , every answer must start with A: , and there must be an empty line between Q&A pairs.",
            height=400,
            value=st.session_state.get('faq_text', ''),
            placeholder="Q: What is your return policy?\nA: We offer a 30-day return policy for all items.\n\nQ: How long does shipping take?\nA: Standard shipping takes 3-5 business days."
        )
        
        faq_data = parse_faq_text(faq_text)
        if faq_data:
            st.success(f"{len(faq_data)} FAQ entries loaded")
    
    with col2:
        st.header("Ask a Question")
        
        user_question = st.text_input("Your question:", placeholder="How do I get my money back?")
        
        if st.button("Search", type="primary"):
            if user_question and faq_data:
                with st.spinner("Searching..."):
                    # Generate embeddings and search for similar questions
                    questions = [item['question'] for item in faq_data]
                    question_embeddings = model.encode(questions)
                    results = search_faq(model, question_embeddings, faq_data, user_question, top_k=3)
                
                if results:
                    st.success("Found relevant answers:")
                    for i, result in enumerate(results, 1):
                        st.write(f"**{i}. {result['question']}**")
                        st.write(f"*Similarity: {result['similarity']:.3f}*")
                        st.write(result['answer'])
                        st.write("---")
                else:
                    st.warning("No relevant answers found.")
            elif not user_question:
                st.error("Please enter a question.")
            elif not faq_data:
                st.error("Please enter FAQ data first.")


if __name__ == "__main__":
    main()